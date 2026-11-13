"""v6 candidate: team-shape classifier WITH memory (context features), vs v5, same CV folds."""
import json
import os
import pickle
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

import harness as h
import hmm_possession as hp
import search_v51
import suite

CODE = {"H": 0, "A": 1, "D": 2}


def xy(cases, step=5):
    X, y = [], []
    for c in cases:
        keep = np.where(c.lab != "X")[0][::step]
        Xc = hp.context_features(c.Xf)[keep]
        lab = np.array([CODE[v] for v in c.lab[keep]])
        X += [Xc, hp.swap_teams_context(Xc)]
        y += [lab, np.where(lab == 0, 1, np.where(lab == 1, 0, 2))]
    return np.vstack(X), np.concatenate(y)


def fit(cases, C=1.0):
    X, y = xy(cases)
    mean, scale = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(max_iter=3000, C=C)
    clf.fit((X - mean) / scale, y)
    return {"features": hp.context_feature_names(), "classes": ["H", "A", "D"],
            "mean": mean.round(6).tolist(), "scale": scale.round(6).tolist(),
            "coef": clf.coef_.round(6).tolist(), "intercept": clf.intercept_.round(6).tolist()}


if __name__ == "__main__":
    dev = pickle.load(open(os.path.join(h.HERE, "cache", "suite_dev_oof51.pkl"), "rb"))
    p = dict(hp.PARAMS)
    games = sorted({c.game for c in dev})
    fold_of = {g: i % 3 for i, g in enumerate(games)}       # same folds as v5
    L5 = {id(c): c.L for c in dev}
    L6 = {}
    for k in range(3):
        model = fit([c for c in dev if fold_of[c.game] != k])
        for c in dev:
            if fold_of[c.game] == k:
                L6[id(c)] = hp.learned_emissions(hp.context_features(c.Xf), model, 1.0).astype(
                    np.float32)
        print("fold", k, "trained", flush=True)

    def show(tag, r):
        print("%-22s dev %.4f  metrica %.4f  skillcorner %.4f" % (
            tag, r["spell"], r["metrica"], r["skillcorner"]), flush=True)

    for c in dev:
        c.L = L5[id(c)]
    show("v5 (reference)", suite.score_cases(dev, p, search_v51.predict))
    for c in dev:
        c.L = L6[id(c)]
    for w in tuple(float(x) for x in os.environ.get("V6_WEIGHTS", "0.02,0.03,0.05,0.08").split(",")):
        show("v6 w=%s" % w, suite.score_cases(dev, dict(p, shape_weight=w), search_v51.predict))
    if not os.environ.get("V6_NOFINAL"):
        json.dump(fit(dev), open(os.path.join(h.HERE, "shape_model_v6.json"), "w"))
    print("final v6 model written", flush=True)
