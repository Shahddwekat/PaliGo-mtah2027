"""Stage 3 (v5): re-tune the HMM settings together with the learned team-shape evidence.

Each development case gets out-of-fold learned evidence (from a classifier that did not see its
match), so the search does not reward the classifier for memorising its own training matches.
"""
import json
import os
import pickle
import time

import numpy as np

import harness as h
import hmm_possession as hp
import suite
import train_shape as ts

GRID = {
    "shape_weight": [0.02, 0.03, 0.05],
    "min_frames": [25, 37, 50, 75],
    "alive_mean_s": [3.0, 4.0, 6.0, 9.0],
    "dead_mean_s": [8.0, 15.0, 25.0],
    "turnover_share": [0.5, 0.6, 0.7],
    "team_max": [0.2, 0.4, 0.8],
    "sp_dead_weight": [0.1, 0.2, 0.4],
    "sp_team_weight": [0.4, 0.8, 1.2],
    "still_weight": [0.0, 0.15, 0.3, 0.6],
    "out_weight": [0.5, 1.0, 2.0],
    "speed_weight": [0.05, 0.1, 0.2],
}


def attach_oof(dev, folds=3):
    games = sorted({c.game for c in dev})
    fold_of = {g: i % folds for i, g in enumerate(games)}
    for k in range(folds):
        model = ts.fit([c for c in dev if fold_of[c.game] != k])
        for c in dev:
            if fold_of[c.game] == k:
                c.L = hp.learned_emissions(c.Xf, model, 1.0).astype(np.float32)


def predict(case, p):
    e = hp.emissions(case.f, p) + p["shape_weight"] * case.L
    hp.apply_setpieces(e, case.tags, h.TEAM_NAMES, p)
    logt = hp.transition_logs(p)
    pred = np.full(len(case.lab), "X", dtype="<U1")
    for a, b in case.blocks:
        pred[a:b + 1] = hp.LETTERS[h.viterbi(e[a:b + 1], logt, p["min_frames"])]
    return pred


if __name__ == "__main__":
    p = dict(hp.PARAMS, shape_weight=0.03)
    path = os.path.join(h.HERE, "cache", "suite_dev_oof.pkl")
    if os.path.isfile(path):
        dev = pickle.load(open(path, "rb"))
    else:
        dev = suite.build("dev", p)
        attach_oof(dev)
        pickle.dump(dev, open(path, "wb"))
    best = suite.score_cases(dev, p, predict)["spell"]
    print("start (v4 settings + shape 0.03): %.4f" % best, flush=True)
    t = time.time()
    for sweep in range(2):
        changed = False
        for key, values in GRID.items():
            sc = {v: best if v == p[key] else
                  suite.score_cases(dev, dict(p, **{key: v}), predict)["spell"] for v in values}
            v = max(sc, key=sc.get)
            if v != p[key] and sc[v] > best + 0.002:
                p[key], best, changed = v, sc[v], True
                print("sweep %d %-16s -> %-6s dev %.4f" % (sweep, key, v, best), flush=True)
        if not changed:
            break
    json.dump({k: p[k] for k in GRID}, open(os.path.join(h.HERE, "search_v5.json"), "w"), indent=1)
    print("final dev %.4f" % best, json.dumps({k: p[k] for k in GRID}), "took %.0fs" % (time.time() - t))
