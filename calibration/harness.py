"""Calibrate hmm_possession PARAMS on Metrica's open sample games (other matches only).

The open tracking is cleaner than the competition ATD: there the ball is missing almost exactly when
it is dead, so "ball missing" would look like a perfect dead-ball signal. To calibrate for the
automatic tracking instead, every game is degraded into several *scenarios* before scoring:

  * the dead-ball stretches get a visible ball (held where play stopped, carried to the restart
    spot, held there) -- what an automatic tracker sees during a stoppage;
  * the ball is then removed in random runs, at separate rates for live and dead play, so that
    missingness is no longer a label in disguise; the scenarios bracket how it might really split;
  * position noise, and player slots dropped in runs (the ATD sees ~20.5 of 22 players).

A parameter set is scored by the official spellF1 (evaluation/possession_metrics.py), averaged over
the scenarios. Games 1 and 2 are used to choose; game 3 is held out to check the choice holds.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "evaluation"))
sys.path.insert(0, os.path.join(REPO, "solution"))
import possession_metrics as pm  # noqa: E402
import hmm_possession as hp  # noqa: E402
from metrica_games import load_game  # noqa: E402

try:
    import numba
except ImportError:          # pure-python fallback (same result, slower)
    numba = None

FPS = 25
PITCH = (105.0, 68.0)
CODE = {"H": 0, "A": 1, "D": 2, "X": 3}

# (name, live-play missing rate, dead-ball missing rate, mean missing run in seconds)
SCENARIOS = [
    ("even-short", 0.41, 0.41, 1.0),
    ("even-long", 0.41, 0.41, 3.0),
    ("deadheavy-short", 0.33, 0.60, 1.0),
    ("deadheavy-long", 0.33, 0.60, 3.0),
    ("liveheavy-short", 0.46, 0.30, 1.0),
    ("liveheavy-long", 0.46, 0.30, 3.0),
]


def runs_mask(n, rate, mean_run, rng):
    """Boolean mask with ~rate True in runs of mean length mean_run frames (alternating renewal)."""
    if rate <= 0:
        return np.zeros(n, bool)
    mean_gap = mean_run * (1 - rate) / rate
    out = np.zeros(n, bool)
    i, on = 0, rng.random() < rate
    while i < n:
        length = max(1, int(rng.exponential(mean_run if on else mean_gap)))
        if on:
            out[i:i + length] = True
        i += length
        on = not on
    return out


def label_runs(lab, value):
    m = lab == value
    edges = np.diff(np.concatenate(([0], m.astype(int), [0])))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def degrade(z, scenario, seed):
    _, live_rate, dead_rate, mean_s = scenario
    rng = np.random.default_rng(seed)
    lab = z["labels"]
    n = len(lab)
    bx, by = z["bx"].astype(np.float64).copy(), z["by"].astype(np.float64).copy()
    px, py = z["px"].astype(np.float64).copy(), z["py"].astype(np.float64).copy()

    # 1. a visible ball through every stoppage: held, carried to the restart, held.
    for a, b in label_runs(lab, "D"):
        base = max(0, a - 75)
        before = np.where(~np.isnan(bx[base:a]))[0]
        after = np.where(~np.isnan(bx[b:b + 75]))[0]
        if not len(before) and not len(after):
            continue
        p0 = (bx[base + before[-1]], by[base + before[-1]]) if len(before) else None
        p1 = (bx[b + after[0]], by[b + after[0]]) if len(after) else None
        p0, p1 = p0 or p1, p1 or p0
        L = b - a
        t = np.clip((np.arange(L) / max(L - 1, 1) - 0.3) / 0.4, 0, 1)
        bx[a:b] = p0[0] + (p1[0] - p0[0]) * t
        by[a:b] = p0[1] + (p1[1] - p0[1]) * t
    # fill the few live-play gaps of the clean data by interpolation (they are re-removed below)
    ok = ~np.isnan(bx)
    if ok.any():
        idx = np.arange(n)
        bx = np.interp(idx, idx[ok], bx[ok])
        by = np.interp(idx, idx[ok], by[ok])

    # 2. noise
    bx += rng.normal(0, 0.3 / PITCH[0], n)
    by += rng.normal(0, 0.3 / PITCH[1], n)
    px += rng.normal(0, 0.15 / PITCH[0], px.shape)
    py += rng.normal(0, 0.15 / PITCH[1], py.shape)

    # 3. remove the ball in runs, at different rates in live and dead play
    run = mean_s * FPS
    live_mask = runs_mask(n, live_rate, run, rng)
    dead_mask = runs_mask(n, dead_rate, run, rng)
    miss = np.where(lab == "D", dead_mask, live_mask)
    bx[miss] = np.nan
    by[miss] = np.nan

    # 4. drop player slots in runs (~7%)
    for j in range(px.shape[1]):
        drop = runs_mask(n, 0.07, 3 * FPS, rng)
        px[drop, j] = np.nan
        py[drop, j] = np.nan
    ps = z["ps"].copy()
    ps[np.isnan(px)] = np.nan
    return (z["team"], px.astype(np.float32), py.astype(np.float32), ps,
            bx.astype(np.float32), by.astype(np.float32))


def blocks_of(lab, periods):
    out = []
    for per in np.unique(periods):
        idx = np.where((periods == per) & (lab != "X"))[0]
        if len(idx):
            out.append((int(idx[0]), int(idx[-1])))
    return out


if numba is not None:
    @numba.njit(cache=True)
    def _viterbi_nb(e, logt, k):
        n = e.shape[0]
        neg = -np.inf
        delta = np.full((3, k), neg)
        for l in range(3):
            delta[l, k - 1] = np.log(1.0 / 3.0) + e[0, l]
        from_label = np.zeros((n, 3), np.int8)
        stayed = np.zeros((n, 3), np.bool_)
        new = np.empty((3, k))
        for t in range(1, n):
            for to in range(3):
                best, bi = neg, 0
                for fr in range(3):
                    if fr == to:
                        continue
                    v = delta[fr, k - 1] + logt[fr, to]
                    if v > best:
                        best, bi = v, fr
                new[to, 0] = best
                from_label[t, to] = bi
            for l in range(3):
                keep = delta[l, k - 1] + logt[l, l]
                if k > 1:
                    for i in range(k - 2, 0, -1):
                        new[l, i] = delta[l, i - 1]
                    grow = delta[l, k - 2]
                    stayed[t, l] = keep >= grow
                    new[l, k - 1] = keep if keep >= grow else grow
                else:
                    stayed[t, l] = keep >= new[l, 0]
                    if keep > new[l, 0]:
                        new[l, 0] = keep
            for l in range(3):
                for i in range(k):
                    delta[l, i] = new[l, i] + e[t, l]
        best, label, sub = neg, 0, 0
        for l in range(3):
            for i in range(k):
                if delta[l, i] > best:
                    best, label, sub = delta[l, i], l, i
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
                label = from_label[t, label]
                sub = k - 1
            else:
                sub -= 1
        return path

    def viterbi(e, logt, k):
        return _viterbi_nb(np.ascontiguousarray(e, np.float64), logt, int(k))
else:
    viterbi = hp.viterbi_min_duration


TEAM_NAMES = ("Home", "Away")
TAG_TITLE = {"THROW IN": "Throw-in", "CORNER KICK": "Corner Kick", "GOAL KICK": "Goal Kick",
             "FREE KICK": "Free Kick", "KICK OFF": "Kick Off", "PENALTY": "Penalty"}


def simulate_setpiece_tags(z, rng, recall=0.6, false_rate=0.1):
    """Set-piece tags like Smart Tagging's, from this game's own restart events.

    What is copied from the competition's tagging is only its published shape: about 59 set-piece
    tags for a match with ~90-100 restarts (so ~60% recall), an interval that starts ~3.4 s before
    the restart (both tagged kickoffs), lasts ~5 s, and has Team = N/A on every throw-in. Jitter of
    ~0.6 s on the start and some false tags at random moments make it imperfect.
    """
    out = []
    for i, team, kind in zip(z["restart_idx"], z["restart_team"], z["restart_kind"]):
        if rng.random() > recall:
            continue
        start = int(i - 85 + rng.normal(0, 15))
        end = int(start + 125 + rng.normal(0, 25))
        name = "N/A" if kind == "THROW IN" else TEAM_NAMES[0 if team == "H" else 1]
        out.append({"code": "SET PIECES", "start_frame": start, "end_frame": end,
                    "tags": {"Team": name, "Type": TAG_TITLE.get(str(kind), str(kind))}})
    graded = np.where(z["labels"] != "X")[0]
    for _ in range(int(false_rate * len(out))):
        start = int(rng.choice(graded))
        out.append({"code": "SET PIECES", "start_frame": start, "end_frame": start + 125,
                    "tags": {"Team": TEAM_NAMES[int(rng.integers(2))], "Type": "Free Kick"}})
    return out


class Case:
    """One (game, scenario): degraded tracking, its features and the event-derived truth."""

    def __init__(self, g, z, scenario, seed, p):
        self.name = "g{}/{}".format(g, scenario[0])
        self.game = g
        self.lab = z["labels"]
        self.gt = pm.label_codes(self.lab)
        self.blocks = blocks_of(self.lab, z["periods"])
        graded = self.lab != "X"
        arrays = degrade(z, scenario, seed)
        self.f = hp.frame_features(*arrays, graded, p=p, pitch=PITCH)
        self.raw = arrays
        self.tags = simulate_setpiece_tags(z, np.random.default_rng(seed + 7))


def predict(case, p):
    e = hp.emissions(case.f, p)
    hp.apply_setpieces(e, case.tags, TEAM_NAMES, p)
    logt = hp.transition_logs(p)
    pred = np.full(len(case.lab), "X", dtype="<U1")
    for a, b in case.blocks:
        pred[a:b + 1] = hp.LETTERS[viterbi(e[a:b + 1], logt, p["min_frames"])]
    return pred


def score(case, pred):
    r = pm.evaluate(case.gt, pm.label_codes(pred))
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"] if "n_pred_spells" in r else None


def baseline_pred(case, window=50):
    """The organisers' reference: nearest player, majority vote over +/- window."""
    f = case.f
    codes = np.where(~f["seen"] | ~np.isfinite(np.minimum(f["d_home"], f["d_away"])), -1,
                     np.where(f["d_home"] <= f["d_away"], 0, 1))
    hc = np.concatenate(([0], np.cumsum(codes == 0)))
    ac = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    out = np.where(h + a > 0, np.where(h >= a, "H", "A"), "D")
    out[case.lab == "X"] = "X"
    return out


def build_cases(p, games=(1, 2, 3)):
    cases = []
    for g in games:
        z = load_game(g)
        for si, sc in enumerate(SCENARIOS):
            cases.append(Case(g, z, sc, seed=1000 * g + si, p=p))
    return cases


def evaluate_params(cases, p):
    rows = []
    for c in cases:
        s, m, _ = score(c, predict(c, p))
        rows.append((c.game, s, m))
    rows = np.array(rows)
    train = rows[rows[:, 0] != 3]
    test = rows[rows[:, 0] == 3]
    return {"train_spell": train[:, 1].mean(), "train_macro": train[:, 2].mean(),
            "test_spell": test[:, 1].mean() if len(test) else np.nan,
            "test_macro": test[:, 2].mean() if len(test) else np.nan}
