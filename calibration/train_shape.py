"""Train the v5 team-shape classifier on external matches only, with leave-games-out CV.

  * features: hmm_possession.shape_features (computed on the degraded external tracking)
  * labels:   H / A / D from each external match's own events
  * model:    multinomial logistic regression, team-symmetric (every frame is also added with
              home and away swapped and H<->A swapped), frames subsampled every 5th
  * CV:       3 folds by game over the development set; the weight of the learned evidence and
              how it is combined with v4's hand-made evidence are chosen on the held-out folds
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

import harness as h
import hmm_possession as hp
import suite

CODE = {"H": 0, "A": 1, "D": 2}


def training_xy(cases, step=5):
    X, y = [], []
    for c in cases:
        keep = np.where(c.lab != "X")[0][::step]
        Xf = c.Xf[keep]
        lab = np.array([CODE[v] for v in c.lab[keep]])
        X += [Xf, hp.swap_teams(Xf)]
        y += [lab, np.where(lab == 0, 1, np.where(lab == 1, 0, 2))]
    return np.vstack(X), np.concatenate(y)


def fit(cases):
    X, y = training_xy(cases)
    mean, scale = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit((X - mean) / scale, y)
    return {"features": hp.feature_names(), "classes": ["H", "A", "D"],
            "mean": mean.round(6).tolist(), "scale": scale.round(6).tolist(),
            "coef": clf.coef_.round(6).tolist(), "intercept": clf.intercept_.round(6).tolist()}


def make_predict(model, w, mode):
    def predict(case, p):
        base = hp.emissions(case.f, p)
        learned = hp.learned_emissions(case.Xf, model, w)
        if mode == "sum":
            e = base + learned
        else:                                    # "replace": learned evidence only
            e = learned.copy()
        hp.apply_setpieces(e, case.tags, h.TEAM_NAMES, p)
        logt = hp.transition_logs(p)
        pred = np.full(len(case.lab), "X", dtype="<U1")
        for a, b in case.blocks:
            pred[a:b + 1] = hp.LETTERS[h.viterbi(e[a:b + 1], logt, p["min_frames"])]
        return pred
    return predict


def cv(dev, p, weights=(0.02, 0.05, 0.1, 0.2, 0.4), modes=("sum", "replace"), folds=3):
    games = sorted({c.game for c in dev})
    fold_of = {g: i % folds for i, g in enumerate(games)}
    res = {(m, w): [] for m in modes for w in weights}
    res[("v4", 0)] = []
    for k in range(folds):
        train = [c for c in dev if fold_of[c.game] != k]
        test = [c for c in dev if fold_of[c.game] == k]
        model = fit(train)
        res[("v4", 0)].append(suite.score_cases(test, p)["spell"])
        for m in modes:
            for w in weights:
                res[(m, w)].append(suite.score_cases(test, p, make_predict(model, w, m))["spell"])
        print("fold", k, "done", flush=True)
    return {k: float(np.mean(v)) for k, v in res.items()}


if __name__ == "__main__":
    p = dict(hp.PARAMS)
    dev = suite.build("dev", p)
    scores = cv(dev, p)
    for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        print("{:<16} CV spellF1 {:.4f}".format("%s w=%s" % k, v))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(h.HERE, "shape_model.json")
    json.dump(fit(dev), open(out, "w"))
    print("final model (all development matches) ->", out)
