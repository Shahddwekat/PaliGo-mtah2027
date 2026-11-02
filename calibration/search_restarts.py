"""Stage 2 (v4): restart / set-piece settings, on top of the stage-1 values already in PARAMS.

Same method as search.py: games 1-2 choose, game 3 is held out, gains under 0.002 are ignored.
"""
import json, os, pickle, time, harness as h, hmm_possession as hp
CASES = os.path.join(h.HERE, "cache", "cases.pkl")
GRID = {
    "sp_dead_weight": [0.0, 0.1, 0.2, 0.4, 0.8, 1.5],
    "sp_team_weight": [0.0, 0.1, 0.2, 0.4, 0.8],
    "sp_team_s": [1.0, 2.0, 3.0, 5.0],
    "edge_still_weight": [0.0, 0.2, 0.4, 0.8, 1.5],
    "edge_m": [0.75, 1.5, 3.0],
    "still_weight": [0.0, 0.15, 0.3, 0.6],
    "still_mps": [0.3, 0.5, 1.0],
    "out_weight": [0.5, 1.0, 2.0],
    "dead_mean_s": [8.0, 15.0, 25.0],
    "min_frames": [25, 37, 50, 75],
    "alive_mean_s": [4.0, 6.0, 9.0],
}
if os.path.isfile(CASES):
    cases = pickle.load(open(CASES, "rb"))
else:
    cases = h.build_cases(dict(hp.PARAMS)); pickle.dump(cases, open(CASES, "wb")); train = [c for c in cases if c.game != 3]
p = dict(hp.PARAMS); best = h.evaluate_params(cases, p); t = time.time()
print("start", {k: round(v, 4) for k, v in best.items()}, flush=True)
for sweep in range(3):
    changed = False
    for key, values in GRID.items():
        sc = {v: (best["train_spell"] if v == p[key] else h.evaluate_params(train, dict(p, **{key: v}))["train_spell"]) for v in values}
        v = max(sc, key=sc.get)
        if v != p[key] and sc[v] > best["train_spell"] + 0.002:
            p[key] = v; best = h.evaluate_params(cases, p); changed = True
            print("sweep %d %-18s -> %-6s train %.4f test %.4f" % (sweep, key, v, best["train_spell"], best["test_spell"]), flush=True)
    if not changed: break
print("final", {k: round(v, 4) for k, v in best.items()})
json.dump({"params": p, "final": best}, open(os.path.join(h.HERE, "search_restarts.json"), "w"), indent=1, default=float)
print(json.dumps({k: p[k] for k in GRID}, indent=1)); print("took %.0fs" % (time.time() - t))
