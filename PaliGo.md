# PaliGo — Technical document

More than a Hack 2027 · Initial Challenge · frame-level ball possession
Team: Shahd, Dana, Sara, Rahaf · Repository: `PaliGo-mtah2027`

## 1. Approach

**Why a state model.** spellF1 scores possessions as intervals. One wrong blip inside a possession
splits it; one missed stoppage merges two. Deciding every frame on its own and smoothing afterwards
(the baseline) trades one error for the other: small windows flicker, large windows blur the
boundaries. We instead treat possession as a **hidden state that persists until the evidence says
it changed**, and decide each half as a whole.

**Model.** A three-state hidden Markov model — `H`, `A`, `D` — decoded exactly with Viterbi, one
half at a time, non-causally (the whole half is seen before deciding).

- **Minimum duration.** Each label is expanded into a chain of sub-states so no run can be shorter
  than 3 s (75 frames). Shorter "possessions" are touches and duels; they almost never match a true
  possession at IoU ≥ 0.3 and often split a real one.
- **Switching costs.** Fixed per-frame transition probabilities: a team keeps the ball for a prior
  mean of 4 s before a change is "cheap", 60% of changes are turnovers and 40% stoppages, and a
  stoppage lasts a prior mean of 25 s. Together with the minimum duration these set how much
  evidence a change needs.
- **Per-frame evidence** (log-scores for H, A, D; only differences matter):
  1. *Who is on the ball* — whether Barcelona's or Athletic's nearest player is closer, trusted
     fully within 1 m and fading fast beyond (a ball in flight 15 m from everyone says nothing).
  2. *Ball state* — ball outside the pitch → dead; ball motionless → probably placed for a restart.
  3. *Players slowing down* — the slow component of a two-Gaussian mixture on the players' mean
     speed, a small push towards dead. The mixture is **fixed from external matches**.
  4. *Team shape with memory* (v5, extended in v6) — a multinomial logistic-regression classifier.
     Per frame: for each team, in its own attacking direction, distance to the ball, players within
     3 m and 8 m, centroid position, forward speed, spread, ball position relative to the team; ball
     visibility, speed and distance to the lines (23 features). v6 adds memory: each team's distance,
     pressure, position, forward speed and ball position averaged over the previous and next 1 s and
     3 s, time since / until the ball is seen, and which team was nearest it when last and next seen
     (67 features). It still has evidence when the ball is missing, because the players are
     tracked. Trained on external matches only (§3); weight 0.12 per frame.
  5. *Smart Tagging, attacking phases* — a small capped push towards the tagged team (§2).
  6. *Smart Tagging, set pieces* — dead ball in the seconds before each tagged restart, then a push
     towards the team taking it (§2).
- **Missing ball.** No ball means no ball-based evidence, not a dead ball: the current state simply
  continues unless other evidence (team shape, tags) says otherwise.

## 2. Assets used

| Asset | Used | How |
|---|---|---|
| Tactical ATD (tracking) | ✅ | All tracking evidence and team-shape features |
| Smart Tagging | ✅ | Attacking phases, SET PIECES, and the 2nd-half `Kick Off (Start)` |
| Tactical homography | ❌ | Not needed: we work in pitch coordinates only |
| Tactical video | ❌ | Not used as input, not used to check output |
| Broadcast video / ATD / homography | ❌ | Not synchronised with the tactical reference |
| Public 5-minute ground truth | ⚠️ | Only to check the pipeline produces a valid, scorable file |

**Which Smart Tagging codes and why.** An automatic check on the provided data (no ground truth
involved) measured, for every code, how often its `Team` agrees with the team of the player
nearest the ball. This showed that each attacking code has a mirror "defending" code over the same
intervals naming the other team. We therefore use only the attacking codes, with full weight for
`BUILD UP` and `PROGRESSION` and half weight for `ATTACKING TRANSITION`, `BALL IN FINAL THIRD` and
`LONG BALL`, and leave out the box codes, whose team agreement was near chance. The push is capped
at 0.05 per frame and ignores the first and last second of each phase.

**Set pieces.** A tagged SET PIECES interval starts about 3.4 s before the restart (seen on both
`Kick Off (Start)` tags against the kickoff times in the challenge documentation). Before the
restart the ball is dead; after it, the tagged team gets a push for 3 s. Throw-ins are tagged
`N/A`, so they contribute only the dead-ball part. Before the second-half kickoff tag the ball is
dead by definition.

## 3. Training data

| Dataset | Matches | Licence | Labels from | Role |
|---|---|---|---|---|
| Metrica Sports sample data | 3 | public sample data | each match's own event data | development |
| SkillCorner open data (A-League 2024/25) | 20 | MIT | each match's own dynamic events | 15 development, 5 locked test |

**None from the provided game.** Labels were derived by rule from each provider's events: the team
of the latest possession event has the ball; the ball is dead from a ball-out, a shot that goes out
or in, or a foul, until the next restart. SkillCorner's 10 fps broadcast tracking was resampled to
25 fps.

**Making it look like the competition tracking.** In the open data the ball is missing almost
exactly when play is dead, which would make "ball missing" a perfect dead-ball signal. Every match
was therefore degraded into scenarios: a visible ball through stoppages, then the ball removed in
runs (~41% of frames, as in the competition data, with the live/dead split and run length varied),
position noise, and player slots dropped. Set-piece tags were simulated from each match's restarts
with the published shape of the competition's tagging (~60% of restarts tagged, throw-ins `N/A`).
Sanity check: the organisers' baseline scores 0.20–0.23 on this simulation, against its official
0.2433 on the competition match.

**How values were chosen.** Coordinate searches on the development matches, scored with the
official spellF1. The team-shape classifier was cross-validated by match (3 folds); the HMM
settings were then re-tuned with out-of-fold classifier evidence. Where Metrica and SkillCorner
preferred different values, we kept the one that was best for the weaker of the two.

## 4. Possession rules implemented

- **Loose balls, passes, deflections.** The team in possession keeps it until the evidence for the
  other team outweighs the switching cost. A pass in flight does not change possession; a ball
  that reaches an opponent and stays changes it.
- **Challenges and duels.** With both teams close to the ball the nearest-player evidence cancels
  out; a change needs sustained evidence, and nothing shorter than 3 s is labelled.
- **Dead ball.** From the ball leaving the pitch, the ball being motionless, players slowing down,
  and set-piece tags, until the restart.
- **Restarts.** The team taking a tagged restart gets the ball; for throw-ins (tagged `N/A`) the
  team comes from the tracking after the throw.
- **Frames without a ball position.** The state continues; team shape and tags still contribute.
- **`X`.** Only outside the two scored halves. Every graded frame gets `H`, `A` or `D`.

## 5. Results

On the external matches (degraded as described), spellF1:

| | development (18 matches) | **locked test (5 matches)** |
|---|---|---|
| organisers' baseline (nearest player, ±50-frame vote) | 0.20 | 0.20 |
| v4 (tracking + tags + set pieces) | 0.45 | 0.45 |
| v5 (+ team shape) | 0.50 (out-of-fold) | 0.51 |
| **v6 (submitted: team shape with memory)** | **0.56** (out-of-fold) | **0.55** |

The 5 locked matches were chosen by a fixed rule before any result was seen. They were scored
twice: once to choose v5 over v4, once to choose v6 over v5. Both times the newer version won on
all five matches (v6 over v5: +0.03 to +0.06 per match), and v6 also improved both external
sources in cross-validation (Metrica 0.48 → 0.53, SkillCorner 0.50 → 0.56).

On the competition match (no score — we have no labels): 217 possessions, mean 15.7 s;
`H` 53362 (26.8%), `A` 31648 (15.9%), `D` 73847 (37.0%), `X` 40462 (20.3%, outside the scored
halves). Reproduction check: these counts must match exactly on any computer.

**Tried and not kept.** Gating the nearest-player evidence by the ball's speed relative to that
player, and shifting the decoded boundaries earlier or later (turnovers, stoppages, restarts):
none improved both Metrica and SkillCorner, so they are off (`calibration/search_v51.py`). A
"ball still near a line" cue for restarts gave nothing and is off.

## 6. Limitations

- **Short possessions.** Nothing shorter than 3 s can be labelled; very short real possessions are
  lost by design.
- **Labelling conventions.** Our external labels follow Metrica's and SkillCorner's event
  conventions, which need not match the organisers' possession definition.
- **Simulation.** The degradation models the automatic tracking; it does not reproduce it.
- **Smart Tagging weights.** The open matches have no Smart Tagging, so the attacking-phase weights
  are reasoned values, and the set-piece weights assume the real tags are about as reliable as our
  simulated ones.
- **Untagged stoppages.** Only about 59 of the match's restarts are tagged; others rely on the
  tracking alone.
- **With more time:** detecting untagged restarts from player configurations, more external
  matches with tracking and events (e.g. the DFL/IDSSE open data), and a better model of where the
  automatic tracker loses the ball.

## 7. Reproduction

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-starter.txt
# competition data in data/tactical/ (not in the repo)
python solution/hmm_possession.py --out evaluation/results/PaliGo.csv
python evaluation/validate_submission.py evaluation/results/PaliGo.csv
```

Deterministic: two runs, and runs on two different computers, give the same file byte for byte.
To redo the calibration and training on the external matches, see `calibration/README.md`.
