"""Development / locked-test suite over all external matches (Metrica 3 + SkillCorner 20).

Split is fixed by rule, before looking at any result: every 4th SkillCorner match (by sorted id)
is LOCKED for the final v4-vs-v5 decision and must not be scored before then. Everything else is
the development set.
"""
import os
import pickle

import numpy as np

import harness as h
import metrica_games
import skillcorner_games

SC_IDS = skillcorner_games.match_ids()
LOCKED = [m for i, m in enumerate(SC_IDS) if i % 4 == 3]
DEV_SC = [m for m in SC_IDS if m not in LOCKED]
SC_SCENARIOS_PER_GAME = 2


def _games(which):
    if which == "dev":
        return [("m%d" % g, lambda g=g: metrica_games.load_game(g), list(range(6)))
                for g in (1, 2, 3)] + [
            ("sc" + m, lambda m=m: skillcorner_games.load_game(m),
             [(2 * i + k) % 6 for k in range(SC_SCENARIOS_PER_GAME)])
            for i, m in enumerate(DEV_SC)]
    if which == "locked":
        return [("sc" + m, lambda m=m: skillcorner_games.load_game(m),
                 [(2 * i + k) % 6 for k in range(SC_SCENARIOS_PER_GAME)])
                for i, m in enumerate(LOCKED)]
    raise ValueError(which)


def build(which, p, cache=True):
    path = os.path.join(h.HERE, "cache", "suite_{}.pkl".format(which))
    if cache and os.path.isfile(path):
        return pickle.load(open(path, "rb"))
    cases = []
    for name, load, scen in _games(which):
        z = load()
        for si in scen:
            seed = sum(map(ord, name)) * 100 + si          # stable across runs
            c = h.Case(name, z, h.SCENARIOS[si], seed=seed, p=p)
            team, px, py, ps, bx, by = c.raw
            c.Xf = h.hp.shape_features(team, px, py, bx, by, c.f, c.blocks, pitch=h.PITCH)
            c.raw = None                                      # keep the pickle small
            cases.append(c)
    pickle.dump(cases, open(path, "wb"))
    return cases


def score_cases(cases, p, predict=None):
    predict = predict or h.predict
    by_game = {}
    for c in cases:
        s, m, _ = h.score(c, predict(c, p))
        by_game.setdefault(c.game, []).append((s, m))
    g = {k: np.mean(v, 0) for k, v in by_game.items()}
    metrica = [v[0] for k, v in g.items() if k.startswith("m")]
    sc = [v[0] for k, v in g.items() if k.startswith("sc")]
    return {"spell": float(np.mean([v[0] for v in g.values()])),
            "macro": float(np.mean([v[1] for v in g.values()])),
            "metrica": float(np.mean(metrica)) if metrica else float("nan"),
            "skillcorner": float(np.mean(sc)) if sc else float("nan"),
            "per_game": {k: round(float(v[0]), 4) for k, v in g.items()}}
