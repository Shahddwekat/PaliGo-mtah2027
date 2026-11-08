# Calibration and training on external matches

Every tuned constant in `solution/hmm_possession.py` (`PARAMS`) and the team-shape classifier
`solution/shape_model.json` come from here, from **other matches only**. Nothing in this folder
reads the competition match, its ground truth, its video, or anything derived from them.

## External data

| Source | Matches | Licence | Used for |
|---|---|---|---|
| [Metrica Sports sample data](https://github.com/metrica-sports/sample-data) | 3 | public sample data | development |
| [SkillCorner open data](https://github.com/SkillCorner/opendata) (A-League 2024/25, credit SkillCorner & PySport) | 20 | MIT | 15 development, **5 locked test** |

The split was fixed by rule before any result was seen: every 4th SkillCorner match by sorted id is
locked (`suite.py`) and was scored exactly once, for the final v4-vs-v5 decision.

## Method

1. **Labels from each match's own events.** Metrica (`metrica_games.py`): the team of the latest
   possession event has the ball; dead from a ball-out / shot out / foul until the next restart, and
   before every restart. SkillCorner (`skillcorner_games.py`): the team of the latest player
   possession; dead after a possession that ends in a game interruption until the restart. Its
   10 fps broadcast tracking is resampled to 25 fps.
2. **Make the tracking look like the competition's** (`harness.py`). In the open data the ball is
   missing almost exactly when play is dead. Each match is degraded into scenarios: a visible ball
   through stoppages, then the ball removed in runs (~41% of frames, live/dead split and run length
   varied), position noise, player slots dropped. Set-piece tags are simulated from each match's
   restarts with the published shape of the competition's tagging (~60% of restarts, interval
   starting ~3.4 s before the restart, Team = N/A on throw-ins, jitter, ~10% false tags).
3. **Score** with the official `evaluation/possession_metrics.py` spellF1.

## Stages

| Stage | Script | What it set |
|---|---|---|
| 1 (v3) | `search.py` | tracking-evidence weights, switching rates, minimum duration |
| 2 (v4) | `search_restarts.py` | set-piece weights |
| — | `fit_speed_mixture.py` | the "players slow" mixture, now **fixed from external matches** instead of fitted on the competition match |
| 3 (v5) | `train_shape.py` | team-shape classifier (multinomial logistic regression, team-symmetric), 3-fold leave-matches-out CV |
| 3 (v5) | `search_v5.py` | HMM settings re-tuned with out-of-fold classifier evidence; where Metrica and SkillCorner disagreed, the value best for the weaker of the two was kept |
| — | `errors.py` | error breakdown (wrong team, merged, split, boundaries, missed/invented stoppages) |

## Results (spellF1)

| | development (18 matches) | **locked test (5 matches)** |
|---|---|---|
| organisers' baseline (nearest player, ±50 vote) | 0.20 | 0.20 |
| v4 | 0.45 | 0.45 |
| **v5** | 0.50 (out-of-fold) | **0.51** |

v5 beat v4 on all 5 locked matches.

## Run it

```
git clone https://github.com/metrica-sports/sample-data calibration/metrica-sample-data
git clone https://github.com/SkillCorner/opendata calibration/skillcorner-opendata   # needs git-lfs
pip install kloppy==3.18.0 numba scikit-learn
python calibration/fit_speed_mixture.py
python calibration/train_shape.py solution/shape_model.json
python calibration/search_v5.py
```

## Limits

- Event-derived labels follow each provider's conventions, which need not match the organisers'
  possession definition exactly; Metrica's are finer-grained than SkillCorner's.
- The degradation models the automatic tracking; it does not reproduce it.
- The open matches have no Smart Tagging: attacking-phase tag weights keep their reasoned values,
  and set-piece weights assume the real tags are about as reliable as the simulated ones.
