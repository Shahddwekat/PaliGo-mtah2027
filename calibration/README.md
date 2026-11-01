# Calibration on other matches

The constants in `solution/hmm_possession.py` (`PARAMS`) were chosen here, on **Metrica Sports'
open sample games** — three other, anonymised matches with tracking and event data
(<https://github.com/metrica-sports/sample-data>). Nothing in this folder reads the competition
match, its ground truth or anything derived from them.

## What it does

1. **Labels from events** (`metrica_games.py`). Each open game's own event data is turned into
   per-frame `H` / `A` / `D` by rule: the team of the latest possession event has the ball; the ball
   is dead from a ball-out, a shot that goes out or in, or a foul, until the next restart.
2. **Make the tracking look like the competition's** (`harness.py`). The open tracking is cleaner:
   its ball is missing almost exactly when play is dead, which would make "ball missing" a perfect
   dead-ball cue. So each game is degraded into six scenarios: a visible ball through stoppages,
   then the ball removed in runs (~41% of frames overall, with the split between live and dead play
   and the run length varied across scenarios), position noise, and player slots dropped in runs.
   Sanity check: the organisers' baseline (nearest player, ±50-frame vote) scores spellF1 0.21–0.23
   on these cases, against 0.2433 officially on the competition match.
3. **Search** (`search.py`). Coordinate search over `PARAMS`, scored with the official
   `evaluation/possession_metrics.py` spellF1, averaged over the scenarios of **games 1 and 2**.
   **Game 3 is held out** and only reported. A change is kept only if it gains more than 0.002.

## Run it

```
git clone https://github.com/metrica-sports/sample-data calibration/metrica-sample-data
pip install kloppy==3.18.0 numba        # numba only makes it faster
python calibration/search.py
```

The result is written to `calibration/search_result.json`; the values were copied into `PARAMS`.

## Limits

- The event-derived labels follow Metrica's event conventions, which need not match the
  organisers' possession definition exactly.
- The degradation is a model of the automatic tracking, not the real thing. The scenarios bracket
  plausible cases rather than reproduce one.
- The open games have no Smart Tagging, so the tag weights (`tag_weight`, `tag_trim_s`) are not
  calibrated here; they stay at their reasoned values.
