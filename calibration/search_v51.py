"""Stage 4 (v5.1 candidates): ball-speed gate and boundary timing, on the development matches
with out-of-fold team-shape evidence. A change is kept only if it helps BOTH data sources
(Metrica and SkillCorner), so it is not an artefact of one provider's labelling style."""
import json
import os
import pickle

import numpy as np

import harness as h
import hmm_possession as hp
import search_v5
import suite


def predict(case, p):
    e = hp.emissions(case.f, p) + p["shape_weight"] * case.L
    hp.apply_setpieces(e, case.tags, h.TEAM_NAMES, p)
    logt = hp.transition_logs(p)
    pred = np.full(len(case.lab), "X", dtype="<U1")
    for a, b in case.blocks:
        path = h.viterbi(e[a:b + 1], logt, p["min_frames"])
        pred[a:b + 1] = hp.LETTERS[hp.shift_boundaries(path, p["shift_frames"])]
    return pred


def show(tag, r):
    print("%-34s dev %.4f  metrica %.4f  skillcorner %.4f" % (
        tag, r["spell"], r["metrica"], r["skillcorner"]), flush=True)


if __name__ == "__main__":
    p = dict(hp.PARAMS)
    path = os.path.join(h.HERE, "cache", "suite_dev_oof51.pkl")
    if os.path.isfile(path):
        dev = pickle.load(open(path, "rb"))
    else:
        dev = suite.build("dev", p, cache=False)
        search_v5.attach_oof(dev)
        pickle.dump(dev, open(path, "wb"))
    base = suite.score_cases(dev, p, predict)
    show("v5 (reference)", base)
    for g0 in (3.0, 5.0, 8.0, 12.0):
        for g1 in (2.0, 4.0, 8.0):
            show("gate %s / %s" % (g0, g1), suite.score_cases(
                dev, dict(p, gate_rel_mps=g0, gate_decay_mps=g1), predict))
    for kind in ("HA", "AD", "DA"):
        for k in (-25, -12, -6, 6, 12, 25):
            sh = dict(p["shift_frames"], **{kind: k})
            show("shift %s %+d" % (kind, k), suite.score_cases(dev, dict(p, shift_frames=sh),
                                                               predict))
