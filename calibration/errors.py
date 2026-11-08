"""Where does the model lose spellF1? Error breakdown on the development suite (external only)."""
import collections

import numpy as np

import harness as h


def spells(lab):
    """[(start, end_exclusive, team)] for maximal H/A runs."""
    out, i, n = [], 0, len(lab)
    while i < n:
        if lab[i] in "HA":
            j = i
            while j < n and lab[j] == lab[i]:
                j += 1
            out.append((i, j, lab[i]))
            i = j
        else:
            i += 1
    return out


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def analyse(cases, p, predict=None):
    predict = predict or h.predict
    cnt = collections.Counter()
    start_off, end_off = [], []
    miss_dead_len, false_dead_len = [], []
    team_acc = collections.defaultdict(lambda: [0, 0])
    for c in cases:
        gt = c.lab
        pr = predict(c, p)
        G, P = spells(gt), spells(pr)
        # index pred spells by time for overlap search
        p_starts = np.array([s[0] for s in P])
        for g in G:
            lo = max(0, np.searchsorted(p_starts, g[0]) - 3)
            cand = [q for q in P[lo:lo + 40] if overlap(g, q) > 0]
            best = max(cand, key=lambda q: overlap(g, q) / (max(g[1], q[1]) - min(g[0], q[0])),
                       default=None)
            iou = 0 if best is None else overlap(g, best) / (max(g[1], best[1]) - min(g[0], best[0]))
            cnt["gt_spells"] += 1
            if iou >= 0.5 and best[2] == g[2]:
                cnt["matched@0.5"] += 1
                start_off.append(best[0] - g[0])
                end_off.append(best[1] - g[1])
                continue
            # classify the miss
            glen = g[1] - g[0]
            same = [q for q in cand if q[2] == g[2]]
            other = sum(overlap(g, q) for q in cand if q[2] != g[2])
            cover_same = sum(overlap(g, q) for q in same)
            if not cand or (cover_same + other) < 0.5 * glen:
                cnt["miss: predicted dead/none"] += 1
            elif other > cover_same:
                cnt["miss: wrong team"] += 1
            elif len(same) >= 2:
                cnt["miss: split"] += 1
            elif best is not None and best[2] == g[2] and (best[1] - best[0]) > 1.5 * glen:
                cnt["miss: merged into longer"] += 1
            else:
                cnt["miss: boundaries off"] += 1
            if glen < 3 * 25:
                cnt["  (of misses: short spell < 3 s)"] += 1
        # dead-ball gaps between two spells: missed or invented
        gd = gt == "D"
        pd_ = pr == "D"
        for a, b in h.label_runs(gt, "D"):
            if not pd_[a:b].any():
                miss_dead_len.append(b - a)
        for a, b in h.label_runs(pr, "D"):
            if not gd[a:b].any():
                false_dead_len.append(b - a)
        alive = np.isin(gt, ["H", "A"]) & np.isin(pr, ["H", "A"])
        seen = c.f["seen"]
        for name, m in (("ball visible", alive & seen), ("ball missing", alive & ~seen)):
            team_acc[name][0] += int(np.sum(gt[m] == pr[m]))
            team_acc[name][1] += int(np.sum(m))
    n = cnt["gt_spells"]
    print("true possessions: {}  matched at IoU>=0.5: {:.0%}".format(n, cnt["matched@0.5"] / n))
    for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
        if k.startswith("miss") or k.startswith("  "):
            print("  {:<34} {:>5}  {:.1%} of true possessions".format(k, v, v / n))
    so, eo = np.array(start_off) / 25, np.array(end_off) / 25
    print("matched spells: start offset median {:+.2f}s (|.| p50 {:.2f}s p90 {:.2f}s), "
          "end offset median {:+.2f}s (|.| p50 {:.2f}s p90 {:.2f}s)".format(
              np.median(so), np.median(abs(so)), np.percentile(abs(so), 90),
              np.median(eo), np.median(abs(eo)), np.percentile(abs(eo), 90)))
    md, fd = np.array(miss_dead_len) / 25, np.array(false_dead_len) / 25
    print("true stoppages completely missed: {} (median {:.1f}s);  invented stoppages: {} "
          "(median {:.1f}s)".format(len(md), np.median(md) if len(md) else 0, len(fd),
                                    np.median(fd) if len(fd) else 0))
    for k, (a, b) in team_acc.items():
        print("team correct when both say alive, {}: {:.1%} of {} frames".format(k, a / b, b))
    return cnt
