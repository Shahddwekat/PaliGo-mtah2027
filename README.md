# PaliGo — More than a Hack 2027, Initial Challenge

Frame-by-frame ball possession (`H` / `A` / `D` / `X`) for Barcelona vs Athletic Club, LaLiga
2024-25, from the automatic tracking data and Smart Tagging.

**Team PaliGo:** Qais, Shahd, Dana.
**Technical document:** [`PaliGo.md`](PaliGo.md).

## Approach in one paragraph

Possession is treated as a **state that persists** until the evidence says it changed. A
three-state hidden Markov model (H / A / D, minimum 3 s per run) is decoded with Viterbi over each
half. Every frame contributes evidence from the tracking (who is touching the ball, ball out or
still, players slowing down), from a **team-shape classifier with memory** (who is moving forward,
spreading out, pressing over the surrounding seconds, and who was nearest the ball when it was
last and next seen — it works even when the ball is missing), and from the automatic Smart Tagging
(attacking phases and set pieces). Everything that was trained or tuned was trained or tuned on
**23 external matches only** (Metrica's and SkillCorner's open data); nothing was fitted on the
provided match, its ground truth or its video.

## Reproduce the submission

Requirements: Python 3.10+.

```
git clone https://github.com/Shahddwekat/PaliGo-mtah2027.git
cd PaliGo-mtah2027
python -m venv .venv
.venv\Scripts\activate            # Linux / macOS: source .venv/bin/activate
pip install -r requirements-starter.txt
```

Put the competition data (from the enrolment link — **not included in this repo**) in
`data/tactical/`:

```
data/tactical/BAR-ATH_tactical_atd.txt
data/tactical/BAR-ATH_tactical_atd.xml
data/tactical/BAR-ATH_tactical_homography.csv
data/tactical/BAR-ATH_tactical_smart_tagging.xml
```

Then the one command that regenerates the submitted file:

```
python solution/hmm_possession.py --out evaluation/results/PaliGo.csv
python evaluation/validate_submission.py evaluation/results/PaliGo.csv
```

The run is deterministic (same file byte for byte on every run; checked on two computers) and
takes a few minutes the first time (it parses and caches the tracking under `data/cache/`) and a
few seconds after that. It uses only numpy and the standard library.

The validator must end with `Valid.`; the label counts it prints are recorded in `PaliGo.md`.

## Repository layout

| Path | What |
|---|---|
| `solution/hmm_possession.py` | the model: features, evidence, Viterbi decoding, CSV writer |
| `solution/shape_model.json` | the team-shape classifier (v6, with memory), trained on external matches only |
| `calibration/` | how every tuned value was obtained, on external matches — see its README |
| `PaliGo.md` | technical document |
| `starter/`, `evaluation/`, `games.py`, `*.md` | the organisers' challenge kit, unchanged |
| `data/` | empty in the repo (git-ignored); the competition data goes here |

## Rules

- No competition data, derived frame-level data or video is in this repository.
- No hand labels or human-made data on the provided match were used, for any purpose.
- Nothing was trained, fitted or tuned on the ground truth or the video. The public 5-minute
  ground truth was used only to check that the pipeline produces a valid, scorable file.
- The match data was never uploaded to any AI service. `.claude/settings.json` blocks AI coding
  tools from reading `data/`.

## Credits

External open data: [Metrica Sports sample data](https://github.com/metrica-sports/sample-data)
and [SkillCorner open data](https://github.com/SkillCorner/opendata) (MIT; credit SkillCorner and
PySport). Challenge kit and data: Mobile World Capital Barcelona, Barça Innovation Hub and Metrica
Sports. The organisers' kit keeps its original licence (`LICENSE`).
