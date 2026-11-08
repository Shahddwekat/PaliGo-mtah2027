"""Possession labelling with a minimum-duration HMM decoded by Viterbi (v5: calibrated tracking + Smart Tagging + restarts + team shape).

    python solution/hmm_possession.py                    # -> evaluation/results/hmm_v5.csv
    python solution/hmm_possession.py --no-tags          # v1 behaviour (tracking + kickoff only)
    python solution/hmm_possession.py --out my_team.csv
    python solution/hmm_possession.py --tag-report       # unsupervised check of Smart Tagging

Idea
----
The baseline decides every frame on its own and then smooths. Here a possession is a *state* that
persists until there is enough evidence that it changed:

  * three labels H / A / D, each split into MIN_FRAMES sub-states, so no label can last less than
    MIN_FRAMES frames (kills flicker without a smoothing window that blurs transitions);
  * fixed per-frame switching probabilities (calibrated on other matches, see calibration/);
  * per-frame evidence ("emissions") from the tracking only:
        - team:  which team's nearest player is closer to the ball, weighted by how close it is
                 (a ball in flight 15 m from everyone says nothing about who has it);
        - alive: ball outside the pitch or motionless -> dead,
                 players collectively slow -> dead (via an unsupervised 2-component mixture on the
                 mean player speed, fitted on this match without any labels);
  * when the ball is missing there is no team evidence, so the current state simply continues;
  * each half is decoded separately, non-causally (the whole half is seen before deciding);
  * before the tagged 2nd-half "Kick Off (Start)" the ball is dead by definition;
  * v2: attacking-phase Smart Tagging intervals add a small, capped nudge toward the tagged team;
  * v3: tracking constants calibrated on Metrica's open sample games (other matches);
  * v4: Smart Tagging SET PIECES -- dead ball just before each tagged restart, then the tagged team
        has it (throw-ins are tagged N/A, so only the dead part applies). Calibrated on the open
        games with simulated set-piece tags;
  * v5: a team-shape classifier (who is moving forward, spread out, pressing the ball) trained on
        23 external matches only -- Metrica's 3 and SkillCorner's 20 open A-League games -- adds its
        evidence every frame, including the ~41% of frames where the ball is missing.

Nothing here is fitted on, or read from, the competition match's ground truth. Every constant is in
PARAMS with the reason it has that value. The public 5-minute window is for checking the file, not for tuning these.
"""

import argparse
import json
import os
import sys
import time
import warnings
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, DATA_DIR, half_video_blocks, require, source_frames  # noqa: E402

FPS = GAMES[GAME]["fps"]
H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])

# Params. v3: the tracking constants were calibrated on three OTHER matches (Metrica's open sample
# games, labels derived from their own event data, tracking degraded to look like the automatic
# ATD) -- see calibration/README.md. Values marked [cal] came out of that search; the rest kept their
# reasoned v1 values because changing them did not help there. Never tuned on the public window.
PARAMS = {
    # [cal] Shortest run of any label, in frames: 75 = 3 s (v5). Shorter runs are touches and duels,
    # which almost never match a true possession at IoU >= 0.3 and often split a real one.
    "min_frames": 75,
    # [cal] Prior mean seconds before a team's spell ends. Lower than the ~12 s a coach would quote
    # because, together with the 2 s minimum above, it sets how cheaply a clear change is accepted.
    "alive_mean_s": 4.0,
    # Of the spells that end, the share that end in a turnover rather than a stoppage.
    "turnover_share": 0.6,
    # [cal] Mean seconds of a stoppage (throw-ins short, goal kicks / fouls / subs long).
    "dead_mean_s": 25.0,
    # --- team evidence (ball visible and inside the pitch) ---
    "team_scale_m": 1.0,        # [cal] distance gap (m) that counts as "clearly closer"
    "team_max": 0.4,            # cap on |log-odds| per frame; frames are strongly correlated
    "team_reach_m": 1.0,        # within this distance the nearest player is fully trusted
    "team_decay_m": 1.5,        # [cal] trust falls fast beyond the reach: only touches count
    # --- alive / dead evidence ---
    "out_margin": 0.005,        # normalised; ball beyond the line by this much is out
    "out_weight": 1.0,          # ball outside the pitch -> dead
    "moving_mps": 2.0,          # ball faster than this inside the pitch -> alive ...
    "moving_weight": 0.0,       # [cal] ... but it did not help: a carried dead ball moves too
    "still_mps": 0.5,           # ball slower than this -> probably placed for a restart
    "still_weight": 0.3,
    # [cal] "Players slow" = slow component of a 2-Gaussian mixture on log mean player speed,
    # FITTED ON THE EXTERNAL MATCHES (Metrica 3 + SkillCorner dev 15, calibration/), fixed here.
    "speed_mix": {"mu_log": [0.2126, 0.8117], "sd_log": [0.5382, 0.3638], "w": [0.4551, 0.5449]},
    "speed_weight": 0.05,        # players-slow mixture posterior (logit) times this, ...
    "speed_clip": 0.2,          # ... clipped
    "missing_weight": 0.0,      # [cal] ball not seen says nothing about dead vs alive
    "ball_speed_halfwin": 3,    # frames either side when smoothing ball speed
    "player_speed_halfwin": 12, # frames either side when smoothing mean player speed
    # --- v4: restarts ---
    # Ball (nearly) still within this distance of a touch/goal line, or beyond it: a throw-in,
    # corner or goal kick being taken. [cal] Gave nothing on the open games, so off.
    "edge_m": 1.5,
    "edge_still_weight": 0.0,
    # Smart Tagging SET PIECES: the tagged interval starts ~3.4 s before the restart itself
    # (measured on the two 'Kick Off (Start)' tags against the published kickoff frames, 88 and
    # 82 frames). Before the restart the ball is dead; after it, the tagged team has it.
    "sp_pre_s": 3.4,
    "sp_dead_weight": 0.4,      # [cal] per-frame push toward D over [tag start, restart)
    "sp_team_s": 3.0,           # seconds after the restart that the tagged team gets a push
    "sp_team_weight": 0.4,      # [cal] the team taking a restart has the ball
    # --- v5: team-shape classifier (solution/shape_model.json, trained on external matches) ---
    "shape_weight": 0.03,       # [cal] per-frame weight of its log-probabilities
    # --- Smart Tagging team evidence (v2) ---
    # Per-frame log-odds toward the tagged side, at most this much in total. 0.05/frame over a
    # 10 s phase is ~12 nats: enough to settle an ambiguous stretch, while ~1 s of clear tracking
    # evidence (0.4/frame) still overrides it.
    "tag_weight": 0.05,
    # Phase boundaries are approximate and phases are not possessions, so ignore this many
    # seconds at each end of every interval.
    "tag_trim_s": 1.0,
}

# Codes whose Team tag is the side IN POSSESSION, with a reliability factor. Chosen from the
# unsupervised --tag-report (agreement with the nearest-player team, no labels involved):
#   BUILD UP 76%, PROGRESSION 70% -> 1.0;  ATTACKING TRANSITION 63%, BALL IN FINAL THIRD 62%,
#   LONG BALL 62% -> 0.5.  Box codes (52-56%) are near chance and left out. DEFENDING* and
#   DEFENSIVE TRANSITION are exact mirrors (same intervals, Team = defending side) of the
#   attacking codes, so using them too would count the same evidence twice.
TAG_TEAM_CODES = {
    "build up": 1.0,
    "progression": 1.0,
    "attacking transition": 0.5,
    "ball in final third": 0.5,
    "long ball": 0.5,
}

FIELD_SEP, GROUP_SEP = ":", ";"


# -- reading the tracking ------------------------------------------------------------------------

def team_of_each_track(meta_path):
    """0 for home slots, 1 for away; first <Team> in the metadata is home (as in baseline.py)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("expected 2 teams in {}, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([H if t == team_ids[0] else A for t in slots], dtype=np.int8)


def _floats(text):
    return np.array([v for v in text.replace(GROUP_SEP, ",").split(",") if v.strip()],
                    dtype=np.float32)


def parse_atd(raw_path, n_frames, n_slots):
    """Arrays (n_frames, n_slots) of player x, y, speed and (n_frames,) ball x, y. NaN = unseen."""
    px = np.full((n_frames, n_slots), np.nan, np.float32)
    py = np.full((n_frames, n_slots), np.nan, np.float32)
    ps = np.full((n_frames, n_slots), np.nan, np.float32)
    bx = np.full(n_frames, np.nan, np.float32)
    by = np.full(n_frames, np.nan, np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1          # ATD counts from 1, submissions from 0
            if not 0 <= frame < n_frames:
                continue
            players = _floats(parts[1])
            k = min(len(players) // 3, n_slots)
            if k:
                players = players[:k * 3].reshape(k, 3)
                px[frame, :k], py[frame, :k], ps[frame, :k] = players.T
            ball = _floats(parts[2])
            if len(ball) >= 2:
                bx[frame], by[frame] = ball[0], ball[1]
    return px, py, ps, bx, by


def load_atd(cache=True):
    """Parse once, then reuse a local cache under data/ (gitignored, never published)."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    n_frames = source_frames("tactical")
    team = team_of_each_track(meta_path)
    cache_path = os.path.join(DATA_DIR, "cache", "atd_arrays.npz")
    stamp = np.array([os.path.getsize(raw_path), int(os.path.getmtime(raw_path))])
    if cache and os.path.isfile(cache_path):
        z = np.load(cache_path)
        if np.array_equal(z["stamp"], stamp):
            return team, z["px"], z["py"], z["ps"], z["bx"], z["by"]
    t0 = time.time()
    print("parsing {} (one-off, ~1-3 min) ...".format(os.path.basename(raw_path)))
    arrays = parse_atd(raw_path, n_frames, len(team))
    print("  done in {:.0f} s".format(time.time() - t0))
    if cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, stamp=stamp, px=arrays[0], py=arrays[1], ps=arrays[2],
                 bx=arrays[3], by=arrays[4])
    return (team,) + arrays


# -- features ------------------------------------------------------------------------------------

def nan_mean_window(x, half):
    """Centred mean over +/- half frames, ignoring NaN. NaN where the window has no value."""
    ok = ~np.isnan(x)
    vals = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cnt = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    n = cnt[hi] - cnt[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (vals[hi] - vals[lo]) / np.maximum(n, 1), np.nan)


def fit_speed_mixture(values, iters=100):
    """2-Gaussian mixture on log mean player speed. Used only OFFLINE, on external matches
    (calibration/), to produce PARAMS["speed_mix"]; never fitted on the competition match."""
    ok = ~np.isnan(values) & (values > 0)
    x = np.log(values[ok].astype(np.float64))
    mu = np.percentile(x, [25, 75]).astype(np.float64)
    sd = np.array([x.std(), x.std()])
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        ll = -0.5 * ((x[:, None] - mu) / sd) ** 2 - np.log(sd) + np.log(w)
        ll -= ll.max(1, keepdims=True)
        r = np.exp(ll)
        r /= r.sum(1, keepdims=True)
        nk = r.sum(0)
        w = nk / nk.sum()
        mu = (r * x[:, None]).sum(0) / nk
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk) + 1e-3
    order = np.argsort(mu)                                  # slow component first
    return {"mu_log": mu[order].round(4).tolist(), "sd_log": sd[order].round(4).tolist(),
            "w": w[order].round(4).tolist()}


def slow_posterior(values, mix):
    """P(slow) for each value under a FIXED mixture (PARAMS["speed_mix"])."""
    mu, sd, w = (np.asarray(mix[k], np.float64) for k in ("mu_log", "sd_log", "w"))
    out = np.full(len(values), np.nan)
    ok = ~np.isnan(values) & (values > 0)
    x = np.log(values[ok].astype(np.float64))
    ll = -0.5 * ((x[:, None] - mu) / sd) ** 2 - np.log(sd) + np.log(w)
    ll -= ll.max(1, keepdims=True)
    r = np.exp(ll)
    out[ok] = r[:, 0] / r.sum(1)
    return out


def frame_features(team, px, py, ps, bx, by, graded, p=PARAMS, pitch=None):
    length, width = pitch or GAMES[GAME]["pitch"]
    seen = ~np.isnan(bx) & ~np.isnan(by)
    m = p["out_margin"]
    out = seen & ((bx < -m) | (bx > 1 + m) | (by < -m) | (by > 1 + m))

    # Nearest home / away player to the ball, in metres.
    dx = (px - bx[:, None]) * length
    dy = (py - by[:, None]) * width
    dist = np.sqrt(dx * dx + dy * dy)
    dist = np.where(np.isnan(dist), np.inf, dist)
    d_home = dist[:, team == H].min(1)
    d_away = dist[:, team == A].min(1)

    # Ball speed (m/s) from consecutive positions, lightly smoothed.
    step = np.hypot(np.diff(bx) * length, np.diff(by) * width) * FPS
    ball_speed = nan_mean_window(np.concatenate(([np.nan], step)), p["ball_speed_halfwin"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # halftime frames have no players
        mean_speed = np.nanmean(ps, axis=1)
    mean_speed = nan_mean_window(mean_speed, p["player_speed_halfwin"])
    slow = np.full(len(bx), np.nan)
    slow_graded = slow_posterior(mean_speed[graded], p["speed_mix"])
    mix = {"mu_mps": np.exp(p["speed_mix"]["mu_log"]).round(2).tolist(),
           "weights": p["speed_mix"]["w"]}
    slow[graded] = slow_graded
    # Distance (m) from the ball to the nearest boundary line; negative outside the pitch.
    edge = np.minimum(np.minimum(bx, 1 - bx) * length, np.minimum(by, 1 - by) * width)
    return {"seen": seen, "out": out, "d_home": d_home, "d_away": d_away,
            "ball_speed": ball_speed, "slow": slow, "mixture": mix, "edge": edge}


def emissions(f, p=PARAMS):
    """(n, 3) log-scores for H, A, D. Only differences between states matter."""
    n = len(f["seen"])
    inside = f["seen"] & ~f["out"]

    alive = np.zeros(n)
    alive[f["out"]] -= p["out_weight"]
    bs = f["ball_speed"]
    with np.errstate(invalid="ignore"):
        alive[inside & (bs > p["moving_mps"])] += p["moving_weight"]
        alive[f["seen"] & (bs < p["still_mps"])] -= p["still_weight"]
        at_edge = f["seen"] & (f["edge"] < p["edge_m"]) & (bs < p["still_mps"])
    alive[at_edge] -= p["edge_still_weight"]
    alive[~f["seen"]] -= p["missing_weight"]
    slow = np.clip(f["slow"], 1e-3, 1 - 1e-3)
    speed_term = np.clip(-p["speed_weight"] * np.log(slow / (1 - slow)),
                         -p["speed_clip"], p["speed_clip"])
    alive += np.nan_to_num(speed_term)

    d_min = np.minimum(f["d_home"], f["d_away"])
    with np.errstate(invalid="ignore"):
        trust = np.exp(-np.maximum(0.0, d_min - p["team_reach_m"]) / p["team_decay_m"])
        gap = np.clip((f["d_away"] - f["d_home"]) / p["team_scale_m"], -1, 1)
    team = np.where(inside & np.isfinite(d_min), trust * gap * p["team_max"], 0.0)
    team = np.nan_to_num(team)

    e = np.empty((n, 3))
    e[:, H] = alive / 2 + team / 2
    e[:, A] = alive / 2 - team / 2
    e[:, D] = -alive / 2
    return e


# -- v5: team-shape features + a classifier trained on external matches -------------------------

SHAPE_FEATURES = (
    # per team, in that team's own attacking direction (T = home, then away)
    "d{T}", "n3{T}", "n8{T}", "cx{T}", "vx{T}", "spread{T}", "ballx{T}", "ball_minus_cx{T}",
)
NEUTRAL_FEATURES = ("seen", "out", "still", "ball_speed", "edge", "slow", "n_players")


def attack_sign(team, px, blocks):
    """+1 if home attacks toward x=1 in each block (half), from where each team stands on
    average over that half: a team's mean position is on its own side of the pitch."""
    sign = np.ones(px.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for first, last in blocks:
            sl = slice(first, last + 1)
            home = np.nanmean(px[sl][:, team == H])
            away = np.nanmean(px[sl][:, team == A])
            sign[sl] = 1.0 if home < away else -1.0
    return sign


def shape_features(team, px, py, bx, by, f, blocks, pitch=None):
    """(n, k) per-frame features that describe each team's shape and movement around the ball.
    They exist even when the ball is missing (players are tracked), which is where the
    nearest-player cue alone is blind."""
    length, width = pitch or GAMES[GAME]["pitch"]
    n = len(bx)
    sgn = attack_sign(team, px, blocks)
    seen = f["seen"]
    X = (px - 0.5) * length * sgn[:, None]           # +x = toward the away goal for home
    Y = (py - 0.5) * width
    BX = (bx - 0.5) * length * sgn
    BY = (by - 0.5) * width
    VX = np.full_like(X, np.nan)
    VX[FPS:] = (X[FPS:] - X[:-FPS])                  # metres per second, over 1 s
    cols = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for T, name, s in ((H, "home", 1.0), (A, "away", -1.0)):
            m = team == T
            x, y, vx = X[:, m] * s, Y[:, m], VX[:, m] * s   # in this team's attack frame
            bxt = BX * s
            d = np.sqrt((x - bxt[:, None]) ** 2 + (y - BY[:, None]) ** 2)
            cols["d" + name] = np.where(seen, np.log1p(np.nan_to_num(np.nanmin(d, 1), nan=50.0)),
                                        0.0)
            cols["n3" + name] = np.where(seen, np.nansum(d < 3, 1), 0.0)
            cols["n8" + name] = np.where(seen, np.nansum(d < 8, 1), 0.0)
            cx = np.nanmean(x, 1)
            cols["cx" + name] = cx / 10.0
            cols["vx" + name] = nan_mean_window(np.nanmean(vx, 1), FPS // 2)
            cols["spread" + name] = (np.nanstd(x, 1) + np.nanstd(y, 1)) / 10.0
            cols["ballx" + name] = np.where(seen, bxt / 10.0, 0.0)
            cols["ball_minus_cx" + name] = np.where(seen, (bxt - cx) / 10.0, 0.0)
    bs = np.nan_to_num(f["ball_speed"], nan=0.0)
    cols["seen"] = seen.astype(float)
    cols["out"] = f["out"].astype(float)
    cols["still"] = (seen & (bs < 0.5)).astype(float)
    cols["ball_speed"] = np.where(seen, np.clip(bs, 0, 30) / 10.0, 0.0)
    cols["edge"] = np.where(seen, np.clip(np.nan_to_num(f["edge"], nan=0.0), -5, 10) / 10.0, 0.0)
    slow = np.nan_to_num(f["slow"], nan=0.5)
    cols["slow"] = slow
    cols["n_players"] = np.sum(~np.isnan(px), 1) / 22.0
    names = feature_names()
    out = np.column_stack([np.nan_to_num(np.asarray(cols[k], float), nan=0.0) for k in names])
    return out.astype(np.float32)


def feature_names():
    per_team = [t.format(T=s) for s in ("home", "away") for t in SHAPE_FEATURES]
    return per_team + list(NEUTRAL_FEATURES)


def swap_teams(Xf):
    """Same features with home and away exchanged (used to train a team-symmetric model)."""
    k = len(SHAPE_FEATURES)
    out = Xf.copy()
    out[:, :k], out[:, k:2 * k] = Xf[:, k:2 * k], Xf[:, :k]
    return out


def learned_emissions(Xf, model, weight):
    """log P(H/A/D | features) from the multinomial logistic model, times weight."""
    z = (Xf - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = z @ np.asarray(model["coef"]).T + np.asarray(model["intercept"])
    logits -= logits.max(1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    return weight * logp


# -- decoding ------------------------------------------------------------------------------------

def transition_logs(p=PARAMS):
    """3x3 log transition matrix between labels, applied when a run is old enough to end."""
    leave_alive = 1.0 / (p["alive_mean_s"] * FPS)
    leave_dead = 1.0 / (p["dead_mean_s"] * FPS)
    t = np.zeros((3, 3))
    for team, other in ((H, A), (A, H)):
        t[team, other] = leave_alive * p["turnover_share"]
        t[team, D] = leave_alive * (1 - p["turnover_share"])
        t[team, team] = 1 - leave_alive
    t[D, H] = t[D, A] = leave_dead / 2
    t[D, D] = 1 - leave_dead
    return np.log(t)


def viterbi_min_duration(e, logt, k):
    """Best label path where every run lasts >= k frames. e: (n, 3) log-emissions."""
    n = len(e)
    neg = -np.inf
    delta = np.full((3, k), neg)
    delta[:, k - 1] = np.log(1 / 3) + e[0]
    from_label = np.zeros((n, 3), np.int8)   # for sub-state 0: which label we came from
    stayed = np.zeros((n, 3), bool)          # for sub-state k-1: came from k-1 (True) or k-2
    switch = logt.copy()
    np.fill_diagonal(switch, neg)
    stay = np.diag(logt)
    for t in range(1, n):
        last = delta[:, k - 1]
        cand = last[:, None] + switch                  # [from, to]
        best_from = cand.argmax(0)
        new = np.empty_like(delta)
        new[:, 0] = cand[best_from, np.arange(3)]
        from_label[t] = best_from
        if k > 1:
            new[:, 1:k - 1] = delta[:, 0:k - 2]
            keep = last + stay
            grow = delta[:, k - 2]
            stayed[t] = keep >= grow
            new[:, k - 1] = np.where(stayed[t], keep, grow)
        else:
            keep = last + stay
            stayed[t] = keep >= new[:, 0]
            new[:, 0] = np.maximum(keep, new[:, 0])
        delta = new + e[t][:, None]

    label, sub = np.unravel_index(np.argmax(delta), delta.shape)
    path = np.empty(n, np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = label
        if t == 0:
            break
        if k == 1:
            if not stayed[t, label]:
                label = from_label[t, label]
        elif sub == k - 1:
            sub = k - 1 if stayed[t, label] else k - 2
        elif sub == 0:
            label, sub = from_label[t, label], k - 1
        else:
            sub -= 1
    return path


# -- Smart Tagging -------------------------------------------------------------------------------

def load_tags():
    try:
        import load_smart_tagging
        return load_smart_tagging.load()
    except (IOError, OSError) as err:
        print("Smart Tagging not used: {}".format(str(err).splitlines()[0]))
        return []


def kickoff_starts(tags):
    """{half_number: start_frame} of each 'Kick Off (Start)' tag."""
    out = {}
    for ev in tags:
        if (ev["code"] or "").strip().lower() != "set pieces":
            continue
        if (ev["tags"].get("Type") or "").strip().lower() != "kick off (start)":
            continue
        half = 1 if (ev["tags"].get("Half") or "").startswith("1") else 2
        out[half] = min(out.get(half, ev["start_frame"]), ev["start_frame"])
    return out


def tag_team_evidence(tags, n, team_names, p=PARAMS):
    """(n,) log-odds for H over A from attacking-phase tags. Positive = home in possession."""
    score = np.zeros(n)
    trim = int(round(p["tag_trim_s"] * FPS))
    used = 0
    for ev in tags:
        factor = TAG_TEAM_CODES.get((ev["code"] or "").strip().lower())
        if factor is None:
            continue
        name = (ev["tags"].get("Team") or "").strip()
        sign = 1.0 if name == team_names[0] else -1.0 if name == team_names[1] else 0.0
        if not sign:
            continue
        lo, hi = max(ev["start_frame"] + trim, 0), min(ev["end_frame"] - trim, n)
        if hi > lo:
            score[lo:hi] += sign * factor * p["tag_weight"]
            used += 1
    return np.clip(score, -p["tag_weight"], p["tag_weight"]), used


def setpiece_evidence(tags, n, team_names, p=PARAMS):
    """From SET PIECES tags: (dead, team) per-frame scores. dead > 0 pushes toward D before the
    restart; team > 0 pushes toward home just after it (throw-ins are N/A: no team push)."""
    dead = np.zeros(n)
    team = np.zeros(n)
    pre = int(round(p["sp_pre_s"] * FPS))
    post = int(round(p["sp_team_s"] * FPS))
    used = 0
    for ev in tags:
        if (ev["code"] or "").strip().lower() != "set pieces":
            continue
        start = max(ev["start_frame"], 0)
        restart = min(start + pre, n)
        if restart <= start:
            continue
        used += 1
        dead[start:restart] = p["sp_dead_weight"]
        name = (ev["tags"].get("Team") or "").strip()
        sign = 1.0 if name == team_names[0] else -1.0 if name == team_names[1] else 0.0
        if sign:
            team[restart:min(restart + post, n)] = sign * p["sp_team_weight"]
    return dead, team, used


def apply_setpieces(e, tags, team_names, p=PARAMS):
    dead, team, used = setpiece_evidence(tags, len(e), team_names, p)
    e[:, H] += (team - dead) / 2
    e[:, A] += (-team - dead) / 2
    e[:, D] += dead / 2
    return used


def tag_report(tags, f, team_names):
    """For each code: how often the tag's Team matches the nearest-player team. No labels used."""
    near = np.where(f["d_home"] < f["d_away"], H, A)
    usable = f["seen"] & ~f["out"] & np.isfinite(np.minimum(f["d_home"], f["d_away"]))
    rows = {}
    for ev in tags:
        name = (ev["tags"].get("Team") or "N/A").strip()
        side = H if name == team_names[0] else A if name == team_names[1] else None
        if side is None:
            continue
        sl = slice(max(ev["start_frame"], 0), max(ev["end_frame"], 0))
        u = usable[sl]
        agree, total = int(np.sum(near[sl][u] == side)), int(np.sum(u))
        r = rows.setdefault(ev["code"], [0, 0, 0])
        r[0] += agree
        r[1] += total
        r[2] += 1
    print("\nSmart Tagging vs nearest-player team (unsupervised consistency check)")
    print("  {:<32} {:>5} {:>9} {:>8}".format("code", "n", "frames", "agree"))
    for code, (agree, total, count) in sorted(rows.items(), key=lambda kv: -kv[1][2]):
        print("  {:<32} {:>5} {:>9} {:>7.0%}".format(code, count, total,
                                                   agree / total if total else float("nan")))
    print("  ~>60% : Team = side in possession;  ~<40% : Team = the other side (e.g. defending)")


# -- main ----------------------------------------------------------------------------------------

SHAPE_MODEL = os.path.join(HERE, "shape_model.json")


def load_shape_model(path=SHAPE_MODEL):
    model = json.load(open(path))
    if model["features"] != feature_names():
        raise ValueError("{} was trained on different features".format(path))
    return model


def run(out_path, do_tag_report=False, cache=True, use_tags=True, use_shape=True):
    meta_path = require(["atd_meta"])[0]
    names = tuple(t.findtext("Name") for t in ET.parse(meta_path).getroot().iter("Team"))
    team, px, py, ps, bx, by = load_atd(cache)
    n = len(bx)
    blocks = half_video_blocks(GAME)
    graded = np.zeros(n, bool)
    for _, first, last in blocks:
        graded[first:last + 1] = True

    f = frame_features(team, px, py, ps, bx, by, graded)
    print("player-speed mixture (fixed, from external matches): means {} m/s, weights {}".format(
        f["mixture"]["mu_mps"], f["mixture"]["weights"]))
    e = emissions(f)
    if use_shape and PARAMS["shape_weight"]:
        Xf = shape_features(team, px, py, bx, by, f, [(a, b) for _, a, b in blocks])
        e += learned_emissions(Xf, load_shape_model(), PARAMS["shape_weight"])
        print("team-shape evidence from {} (trained on external matches only)".format(
            os.path.basename(SHAPE_MODEL)))

    tags = load_tags()
    kick = kickoff_starts(tags)
    if do_tag_report and tags:
        tag_report(tags, f, names)
    if use_tags and tags:
        tag_score, used = tag_team_evidence(tags, n, names)
        e[:, H] += tag_score / 2
        e[:, A] -= tag_score / 2
        print("team evidence from {} Smart Tagging phases, on {:.0%} of graded frames".format(
            used, np.mean(tag_score[graded] != 0)))
    if use_tags and tags and (PARAMS["sp_dead_weight"] or PARAMS["sp_team_weight"]):
        print("restart evidence from {} SET PIECES tags".format(apply_setpieces(e, tags, names)))

    logt = transition_logs()
    labels = np.full(n, "X", dtype="<U1")
    for half, first, last in blocks:
        eh = e[first:last + 1].copy()
        start = kick.get(half)
        if start is not None and start > first:
            pre = min(start, last + 1) - first     # ball dead before the tagged kickoff
            eh[:pre, H] = eh[:pre, A] = -1e9
        path = viterbi_min_duration(eh, logt, PARAMS["min_frames"])
        labels[first:last + 1] = LETTERS[path]
        spells = np.sum((path[1:] != path[:-1]) & (path[1:] != D)) + int(path[0] != D)
        print("half {}: {} frames, kickoff tag at {}, {} possessions predicted".format(
            half, last - first + 1, start, spells))

    folder = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(folder, exist_ok=True)
    with open(out_path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    counts = {k: int(np.sum(labels == k)) for k in "HADX"}
    print("wrote {}  H {H}  A {A}  D {D}  X {X}".format(os.path.relpath(out_path), **counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "evaluation", "results", "hmm_v5.csv"))
    parser.add_argument("--tag-report", action="store_true",
                        help="print how Smart Tagging team tags line up with the tracking")
    parser.add_argument("--no-tags", action="store_true",
                        help="v1 behaviour: ignore Smart Tagging team evidence")
    parser.add_argument("--no-shape", action="store_true",
                        help="v4 behaviour: no team-shape classifier")
    parser.add_argument("--no-cache", action="store_true", help="re-parse the ATD every run")
    args = parser.parse_args()
    run(args.out, args.tag_report, cache=not args.no_cache, use_tags=not args.no_tags,
        use_shape=not args.no_shape)
