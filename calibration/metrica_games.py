"""Metrica open sample games (other matches -- allowed training/validation data) -> model arrays + labels.

Source: https://github.com/metrica-sports/sample-data (3 anonymised games, tracking + events, 25 fps).

For each game this produces exactly what solution/hmm_possession.py consumes for the competition
match -- per-slot team, player x/y/speed, ball x/y, all normalised 0-1 -- plus a frame,label ground
truth derived *by rule* from Metrica's own event data for that game:

  * team in possession = team of the latest possession event (PASS, SHOT, SET PIECE, RECOVERY,
    BALL LOST, BALL OUT -- the team on the ball when it happened). CHALLENGE / CARD / FAULT
    RECEIVED do not change possession.
  * dead ball (D) from the frame the ball leaves play to the restart:
        BALL OUT  end frame  -> start of the next event (throw-in, corner, goal kick ...)
        SHOT that goes out / is a goal -> next event
        FAULT RECEIVED start -> next event (the free kick)
  * X before the first event and after the last event of each period, and between periods.
"""

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# git clone https://github.com/metrica-sports/sample-data calibration/metrica-sample-data
ROOT = os.environ.get("METRICA_DATA") or os.path.join(HERE, "metrica-sample-data", "data")
CACHE = os.path.join(HERE, "cache")
POSSESSION_TYPES = {"PASS", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST", "BALL OUT"}
H, A, D, X = "H", "A", "D", "X"


def _events_csv(path):
    ev = pd.read_csv(path)
    return pd.DataFrame({
        "team": ev["Team"].map({"Home": H, "Away": A}),
        "type": ev["Type"].str.upper(),
        "subtype": ev["Subtype"].fillna("").str.upper(),
        "period": ev["Period"].astype(int),
        "start": ev["Start Frame"].astype(int),
        "end": ev["End Frame"].astype(int),
    })


def _events_json(path, home_team_id):
    rows = []
    for e in json.load(open(path))["data"]:
        subs = e.get("subtypes") or []
        if isinstance(subs, dict):
            subs = [subs]
        rows.append({
            "team": H if e["team"]["id"] == home_team_id else A,
            "type": e["type"]["name"].upper(),
            "subtype": "-".join(s["name"].upper() for s in subs),
            "period": int(e["period"]),
            "start": int(e["start"]["frame"]),
            "end": int((e.get("end") or {}).get("frame") or e["start"]["frame"]),
        })
    return pd.DataFrame(rows)


def labels_from_events(ev, frame_ids, periods):
    """Per tracking frame label. frame_ids are Metrica frame numbers aligned with the events."""
    n = len(frame_ids)
    lab = np.full(n, X, dtype="<U1")
    ev = ev.sort_values(["period", "start"], kind="stable").reset_index(drop=True)
    for period, pe in ev.groupby("period"):
        idx = np.where(periods == period)[0]
        f = frame_ids[idx]
        first, last = pe["start"].min(), max(pe["end"].max(), pe["start"].max())
        # 1. team in possession: latest possession-type event at or before each frame
        poss = pe[pe["type"].isin(POSSESSION_TYPES | {"CARRY"}) & pe["team"].isin([H, A])]
        k = np.searchsorted(poss["start"].to_numpy(), f, side="right") - 1
        team = np.where(k >= 0, poss["team"].to_numpy()[np.maximum(k, 0)], X)
        out = team.astype("<U1")
        # 2. dead ball: from the moment play stops to the next real restart
        restarts = pe[~pe["type"].isin(["CARRY", "CARD", "CHALLENGE", "BALL OUT",
                                         "FAULT RECEIVED"])]["start"].to_numpy()
        for _, e in pe.iterrows():
            if e["type"] == "BALL OUT":
                dead_from = e["end"]
            elif e["type"] == "SHOT" and ("OUT" in e["subtype"] or "GOAL" in e["subtype"]):
                dead_from = e["end"]
            elif e["type"] == "FAULT RECEIVED":
                dead_from = e["start"]
            else:
                continue
            later = restarts[restarts > dead_from]
            dead_to = later[0] if len(later) else last + 1
            out[(f >= dead_from) & (f < dead_to)] = D
        out[(f < first) | (f > last)] = X
        lab[idx] = out
    return lab


def _arrays_from_df(df, slot_cols, slot_team):
    px = np.stack([df[c + "_x"].to_numpy(np.float32) for c in slot_cols], 1)
    py = np.stack([df[c + "_y"].to_numpy(np.float32) for c in slot_cols], 1)
    # Speed from positions (the competition ATD ships it; Metrica sample does not).
    ps = np.full_like(px, np.nan)
    step = np.hypot(np.diff(px, axis=0) * 105.0, np.diff(py, axis=0) * 68.0) * 25.0
    ps[1:] = step
    # light smoothing like a tracker's speed channel
    k = np.ones(5) / 5
    for j in range(ps.shape[1]):
        col = ps[:, j]
        ok = ~np.isnan(col)
        sm = np.convolve(np.where(ok, col, 0), k, "same") / np.maximum(
            np.convolve(ok.astype(float), k, "same"), 1e-9)
        ps[:, j] = np.where(ok, sm, np.nan)
    bx = df["ball_x"].to_numpy(np.float32)
    by = df["ball_y"].to_numpy(np.float32)
    return np.array(slot_team, np.int8), px, py, ps.astype(np.float32), bx, by


def load_game(g):
    """dict(team, px, py, ps, bx, by, labels, periods, pitch) for sample game g (1, 2 or 3)."""
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, "game{}.npz".format(g))
    if os.path.isfile(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    from kloppy import metrica
    d = os.path.join(ROOT, "Sample_Game_{}".format(g))
    if g in (1, 2):
        ds = metrica.load_tracking_csv(
            home_data=os.path.join(d, "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(g)),
            away_data=os.path.join(d, "Sample_Game_{}_RawTrackingData_Away_Team.csv".format(g)),
            coordinates="metrica")
        ev = _events_csv(os.path.join(d, "Sample_Game_{}_RawEventsData.csv".format(g)))
    else:
        ds = metrica.load_tracking_epts(
            meta_data=os.path.join(d, "Sample_Game_3_metadata.xml"),
            raw_data=os.path.join(d, "Sample_Game_3_tracking.txt"), coordinates="metrica")
        ev = _events_json(os.path.join(d, "Sample_Game_3_events.json"),
                          ds.metadata.teams[0].team_id)
    df = ds.to_df()
    cols, teams = [], []
    for ti, team in enumerate(ds.metadata.teams):
        for p in team.players:
            if p.player_id + "_x" in df.columns:
                cols.append(p.player_id)
                teams.append(0 if ti == 0 else 1)
    team, px, py, ps, bx, by = _arrays_from_df(df, cols, teams)
    frame_ids = df["frame_id"].to_numpy()
    periods = df["period_id"].to_numpy()
    labels = labels_from_events(ev, frame_ids, periods)
    out = dict(team=team, px=px, py=py, ps=ps, bx=bx, by=by, labels=labels, periods=periods,
               pitch=np.array([105.0, 68.0]))
    np.savez(cache, **out)
    return out


if __name__ == "__main__":
    for g in (1, 2, 3):
        z = load_game(g)
        lab = z["labels"]
        graded = lab != X
        miss = np.isnan(z["bx"])
        segs = np.sum((lab[1:] != lab[:-1]) & np.isin(lab[1:], [H, A]))
        print("game {}: {} frames, graded {:.0%}, H {:.0%} A {:.0%} D {:.0%} of graded, "
              "~{} possessions; ball missing {:.0%} graded, {:.0%} alive, {:.0%} dead".format(
                  g, len(lab), graded.mean(), (lab[graded] == H).mean(),
                  (lab[graded] == A).mean(), (lab[graded] == D).mean(), segs,
                  miss[graded].mean(), miss[np.isin(lab, [H, A])].mean(), miss[lab == D].mean()))
