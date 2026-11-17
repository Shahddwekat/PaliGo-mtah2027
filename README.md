# PaliGo : ball possession from tracking data

Who has the ball, every frame of a football match  Barcelona (`H`), Athletic Club (`A`), or nobody
(`D`, dead ball) — computed automatically from tracking data and automatic event tags.

Built by **Team PaliGo** (Shahd, Dana, Sara, Rahaf) for **More than a Hack 2027**
(Mobile World Capital Barcelona · Barça Innovation Hub · Metrica Sports).

> **Status:** submitted to the Initial Challenge on 17 November 2026.
> The exact submitted version is the git tag
> [`initial-challenge-submission`](../../tree/initial-challenge-submission).

## Results

Scored with the official metric (spellF1: each possession is an interval, matched to the truth by
overlap) on open matches from other competitions, with the tracking degraded to look like the
challenge data:

| | 18 development matches | 5 held-out test matches |
|---|---|---|
| Organisers' baseline (nearest player + smoothing) | 0.20 | 0.20 |
| **PaliGo v6** | **0.56** | **0.55** |

On the challenge match itself we have no labels, so no score: the model finds 217 possessions,
15.7 s on average.

## How it works

1. **Possession is a state that lasts.** A hidden Markov model with three states (H / A / D)
   only switches when the evidence is stronger than the cost of switching, and nothing shorter
   than 3 seconds is labelled. Viterbi decodes each half as a whole.
2. **Evidence every frame:**
   - who is touching the ball, and whether it is out of play or standing still;
   - a **team-shape classifier with memory**  which team is moving forward, spreading out and
     pressing over the surrounding seconds, and who was nearest the ball when it was last seen;
     it keeps working in the ~41% of frames where the ball is not tracked;
   - automatic **Smart Tagging**: attacking phases and set pieces (restarts).
3. **Trained only on other matches.** Every tuned value and the classifier come from 23 open
   matches (Metrica Sports and SkillCorner), never from the challenge match, its answers or its
   video.

Full details: [`PaliGo.md`](PaliGo.md) (technical document) and
[`calibration/README.md`](calibration/README.md) (how every value was chosen).

## Run it

Python 3.10+.

```
git clone https://github.com/Shahddwekat/PaliGo-mtah2027.git
cd PaliGo-mtah2027
python -m venv .venv
.venv\Scripts\activate            # Linux / macOS: source .venv/bin/activate
pip install -r requirements-starter.txt
```

Put the challenge data (available to registered participants; **not in this repo**) in
`data/tactical/`, then:

```
python solution/hmm_possession.py --out evaluation/results/PaliGo.csv
python evaluation/validate_submission.py evaluation/results/PaliGo.csv
```

The output is deterministic: the same file, byte for byte, on every computer.

## Repository layout

| Path | What |
|---|---|
| `solution/hmm_possession.py` | the model: features, evidence, Viterbi decoding, CSV writer |
| `solution/shape_model.json` | the team-shape classifier (trained on external matches) |
| `calibration/` | training and tuning on open data from other matches |
| `PaliGo.md` | technical document |
| `starter/`, `evaluation/`, `games.py`, challenge `*.md` | the organisers' challenge kit |
| `data/` | empty here (git-ignored); the challenge data goes here |

## Roadmap

Ideas we did not have time for in the Initial Challenge, and directions for the next stage:

- **Untagged restarts** : recognise throw-ins, corners and goal kicks from player positions, not
  only from tags.
- **More training matches** : e.g. the DFL/IDSSE open Bundesliga data.
- **Where the tracker loses the ball** : model it, instead of simulating it.
- **From possession to analysis** : possession chains, territory and pressing maps, and a simple
  tool that turns these labels into something a coach can read.



## Credits

Open data: [Metrica Sports sample data](https://github.com/metrica-sports/sample-data) and
[SkillCorner open data](https://github.com/SkillCorner/opendata) (MIT; credit SkillCorner and
PySport). Challenge kit and data: Mobile World Capital Barcelona, Barça Innovation Hub and Metrica
Sports; the kit keeps its original licence (`LICENSE`).
