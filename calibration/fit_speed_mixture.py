"""Fit PARAMS["speed_mix"] (players-slow mixture) on the external development matches only."""
import json
import warnings

import numpy as np

import hmm_possession as hp
import metrica_games
import skillcorner_games
import suite

vals = []
for z in [metrica_games.load_game(g) for g in (1, 2, 3)] + \
        [skillcorner_games.load_game(m) for m in suite.DEV_SC]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms = np.nanmean(z["ps"], 1)
    ms = hp.nan_mean_window(ms, 12)[z["labels"] != "X"]
    vals.append(ms[~np.isnan(ms) & (ms > 0)])
print(json.dumps(hp.fit_speed_mixture(np.concatenate(vals))))
