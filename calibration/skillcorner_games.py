"""SkillCorner open data (20 A-League 2024/25 matches, MIT licence) -> the same arrays as metrica_games.

Source: https://github.com/SkillCorner/opendata  (credit: SkillCorner & PySport).
Tracking is broadcast tracking at 10 fps; it is resampled here to the 25 fps the model works at.

Labels come from SkillCorner's own dynamic events, by rule:
  * the team of the latest player_possession has the ball until the next one starts;
  * after a possession whose game_interruption_after is set (throw-in, goal kick, corner, free
    kick, goal), the ball is dead until the next possession starts (the restart);
  * X outside each period's played span (first to last possession).
"""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# git clone https://github.com/SkillCorner/opendata calibration/skillcorner-opendata  (+ LFS files)
ROOT = os.environ.get("SKILLCORNER_DATA") or os.path.join(HERE, "skillcorner-opendata", "data")
TRACKING = os.environ.get("SKILLCORNER_TRACKING") or os.path.join(ROOT, "matches")
CACHE = os.path.join(HERE, "cache")
SRC_FPS, FPS = 10, 25
H, A, D, X = "H", "A", "D", "X"
KIND = {"throw_in": "THROW IN", "goal_kick": "GOAL KICK", "free_kick": "FREE KICK",
        "corner": "CORNER KICK", "goal": "KICK OFF"}


def match_ids():
    return sorted(os.listdir(os.path.join(ROOT, "matches")))


def _tracking_path(mid):
    for p in (os.path.join(TRACKING, "{}.jsonl".format(mid)),
              os.path.join(TRACKING, mid, "{}_tracking_extrapolated.jsonl".format(mid))):
        if os.path.isfile(p):
            return p
    raise IOError("no tracking for match {}".format(mid))


def _load_10fps(mid):
    meta = json.load(open(os.path.join(ROOT, "matches", mid, "{}_match.json".format(mid))))
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    home = meta["home_team"]["id"]
    slot = {}
    team = []
    for p in meta["players"]:
        slot[p["id"]] = len(team)
        team.append(0 if p["team_id"] == home else 1)
    rows = [json.loads(line) for line in open(_tracking_path(mid))]
    n = len(rows)
    px = np.full((n, len(team)), np.nan, np.float32)
    py = np.full_like(px, np.nan)
    bx = np.full(n, np.nan, np.float32)
    by = np.full(n, np.nan, np.float32)
    period = np.zeros(n, np.int8)
    for i, r in enumerate(rows):
        period[i] = r["period"] or 0
        b = r["ball_data"]
        if b["x"] is not None:
            bx[i] = b["x"] / length + 0.5
            by[i] = b["y"] / width + 0.5
        for pl in r["player_data"]:
            j = slot.get(pl["player_id"])
            if j is not None:
                px[i, j] = pl["x"] / length + 0.5
                py[i, j] = pl["y"] / width + 0.5
    frames = np.array([r["frame"] for r in rows])
    return meta, np.array(team, np.int8), px, py, bx, by, period, frames, (length, width)


def _labels_10fps(mid, frames, period, home_id):
    ev = pd.read_csv(os.path.join(ROOT, "matches", mid, "{}_dynamic_events.csv".format(mid)),
                     low_memory=False)
    pp = ev[ev["event_type"] == "player_possession"].sort_values("frame_start")
    lab = np.full(len(frames), X, dtype="<U1")
    restarts = []
    pos = {f: i for i, f in enumerate(frames)}
    for per, pe in pp.groupby("period"):
        pe = pe.reset_index(drop=True)
        starts = pe["frame_start"].to_numpy()
        ends = pe["frame_end"].to_numpy()
        teams = np.where(pe["team_id"].to_numpy() == home_id, H, A)
        for i in range(len(pe)):
            nxt = starts[i + 1] if i + 1 < len(pe) else ends[i] + 1
            for f in range(starts[i], nxt):
                if f in pos:
                    lab[pos[f]] = teams[i]
            stop = pe["game_interruption_after"].iloc[i]
            if isinstance(stop, str) and i + 1 < len(pe):
                for f in range(ends[i], nxt):
                    if f in pos:
                        lab[pos[f]] = D
            before = pe["game_interruption_before"].iloc[i]
            if isinstance(before, str):
                kind = next((v for k, v in KIND.items() if before.startswith(k)), None)
                if kind:
                    restarts.append((starts[i], teams[i], kind))
    lab[period == 0] = X
    return lab, restarts


def _resample(a, n_out):
    """10 fps -> 25 fps by linear interpolation; NaN wherever either neighbour is NaN."""
    t = np.arange(n_out) * SRC_FPS / FPS
    lo = np.minimum(np.floor(t).astype(int), len(a) - 1)
    hi = np.minimum(lo + 1, len(a) - 1)
    w = (t - lo).reshape((-1,) + (1,) * (a.ndim - 1))
    return ((1 - w) * a[lo] + w * a[hi]).astype(np.float32)


def load_game(mid):
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, "sc_{}.npz".format(mid))
    if os.path.isfile(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    meta, team, px, py, bx, by, period, frames, pitch = _load_10fps(mid)
    lab10, restarts = _labels_10fps(mid, frames, period, meta["home_team"]["id"])
    n = int(len(frames) * FPS / SRC_FPS)
    near = np.minimum(np.round(np.arange(n) * SRC_FPS / FPS).astype(int), len(frames) - 1)
    PX, PY = _resample(px, n), _resample(py, n)
    BX, BY = _resample(bx, n), _resample(by, n)
    ps = np.full_like(PX, np.nan)
    ps[1:] = np.hypot(np.diff(PX, axis=0) * pitch[0], np.diff(PY, axis=0) * pitch[1]) * FPS
    k = np.ones(5) / 5
    for j in range(ps.shape[1]):
        col = ps[:, j]
        ok = ~np.isnan(col)
        sm = np.convolve(np.where(ok, col, 0), k, "same") / np.maximum(
            np.convolve(ok.astype(float), k, "same"), 1e-9)
        ps[:, j] = np.where(ok, sm, np.nan)
    f2i = {f: i for i, f in enumerate(frames)}
    r_idx = [int(round(f2i[f] * FPS / SRC_FPS)) for f, _, _ in restarts if f in f2i]
    out = dict(team=team, px=PX, py=PY, ps=ps.astype(np.float32), bx=BX, by=BY,
               labels=lab10[near], periods=period[near], pitch=np.array(pitch),
               restart_idx=np.array(r_idx, int),
               restart_team=np.array([t for f, t, _ in restarts if f in f2i], "<U1"),
               restart_kind=np.array([k for f, _, k in restarts if f in f2i], "<U16"))
    np.savez(cache, **out)
    return out


if __name__ == "__main__":
    import sys
    for mid in (sys.argv[1:] or match_ids()):
        z = load_game(mid)
        lab = z["labels"]
        g = lab != X
        miss = np.isnan(z["bx"])
        segs = np.sum((lab[1:] != lab[:-1]) & np.isin(lab[1:], [H, A]))
        print("{}: {} frames, graded {:.0%}, D {:.0%}, ~{} possessions, {} restarts; ball missing "
              "{:.0%} graded ({:.0%} alive, {:.0%} dead); players seen {:.1f}".format(
                  mid, len(lab), g.mean(), (lab[g] == D).mean(), segs, len(z["restart_idx"]),
                  miss[g].mean(), miss[np.isin(lab, [H, A])].mean(), miss[lab == D].mean(),
                  np.mean(np.sum(~np.isnan(z["px"][g]), 1))), flush=True)
