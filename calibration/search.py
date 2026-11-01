"""Coordinate search over PARAMS on games 1-2 (train); game 3 reported as held-out check."""
import json
import os
import pickle
import sys
import time

import harness as h
import hmm_possession as hp

GRID = {
    "min_frames": [6, 12, 25, 37, 50],
    "alive_mean_s": [6.0, 8.0, 12.0, 18.0, 25.0],
    "turnover_share": [0.4, 0.5, 0.6, 0.7, 0.8],
    "dead_mean_s": [8.0, 15.0, 25.0],
    "team_scale_m": [1.0, 2.0, 3.0, 5.0],
    "team_max": [0.2, 0.4, 0.8, 1.5],
    "team_reach_m": [0.5, 1.0, 2.0, 3.0],
    "team_decay_m": [1.5, 3.0, 6.0, 10.0],
    "out_weight": [0.5, 1.0, 2.0, 4.0],
    "moving_mps": [1.0, 2.0, 4.0],
    "moving_weight": [0.0, 0.3, 0.6, 1.0],
    "still_mps": [0.3, 0.5, 1.0],
    "still_weight": [0.0, 0.3, 0.6, 1.0],
    "speed_weight": [0.0, 0.1, 0.3, 0.6],
    "speed_clip": [0.2, 0.5, 1.0],
    "missing_weight": [0.0, 0.05, 0.15, 0.3],
}
CASES = os.path.join(h.HERE, "cache", "cases.pkl")
MIN_GAIN = 0.002   # ignore changes smaller than this: keep the reasoned default


def main(out_path, sweeps=3):
    if os.path.isfile(CASES):
        cases = pickle.load(open(CASES, "rb"))
    else:
        cases = h.build_cases(dict(hp.PARAMS))
        pickle.dump(cases, open(CASES, "wb"))
    train = [c for c in cases if c.game != 3]
    p = dict(hp.PARAMS)
    best = h.evaluate_params(cases, p)
    log = [("start", None, best)]
    print("start", {k: round(v, 4) for k, v in best.items()}, flush=True)
    for sweep in range(sweeps):
        changed = False
        for key, values in GRID.items():
            scores = {}
            for v in values:
                q = dict(p, **{key: v})
                if v == p[key]:
                    scores[v] = best["train_spell"]
                    continue
                scores[v] = h.evaluate_params(train, q)["train_spell"]
            v_best = max(scores, key=scores.get)
            if v_best != p[key] and scores[v_best] > best["train_spell"] + MIN_GAIN:
                p[key] = v_best
                best = h.evaluate_params(cases, p)
                changed = True
                log.append((key, v_best, best))
                print("sweep {} {:<16} -> {:<6} train {:.4f} test {:.4f}".format(
                    sweep, key, v_best, best["train_spell"], best["test_spell"]), flush=True)
        if not changed:
            break
    json.dump({"params": p, "final": best, "log": [(k, v, s) for k, v, s in log]},
              open(out_path, "w"), indent=1, default=float)
    print("final", {k: round(v, 4) for k, v in best.items()})
    print(json.dumps(p, indent=1))


if __name__ == "__main__":
    t = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(h.HERE, "search_result.json"))
    print("took {:.0f} s".format(time.time() - t))
