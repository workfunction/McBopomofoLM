"""v3.2b parity fixture: the T2 study's correction-entry rule (reports/typo.md §8, E+D w=0.25, neighbor prior
row : diagonal = 1 : 0.5 with other = diagonal, theta 1.766) on a sample of dev positions.

Read-only replay of typo/t2.py over results/typo/t2 (dev.pos / dev.e / dev.d / dev.pos.dirs): for every sampled
position the study's candidate set, neighbor direction of each candidate, E and D gains, and the decision
(top candidate and whether its gain reaches theta) computed by t2.gains() with the frozen config.
Sample: every 14th legal-but-wrong position (population L) and every 7th clean position (population C), in the
order of dev.pos.jsonl. Writes McBopomofoTests/SlothEFixtures/v32b_dev_t2.jsonl, one row per position:
  id, pop (L = a typo gave another legal syllable, C = no typo), rd (the sentence's readings with the typed
  syllable at p), p, intended (L only), cands, dir {cand: row|diag|other}, E {cand: gain}, D {cand: gain},
  top (best candidate), gain (its blended gain), offered (gain >= theta)
usage: PYTHONDONTWRITEBYTECODE=1 nice -n 15 $W/sloth/.venv/bin/python SlothE/make_v32b_fixture.py
"""
import json
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W / "typo"))
import t2  # noqa: E402

fz = json.loads((W / "typo" / "t2_frozen.json").read_text())
cfg, theta = tuple(fz["chosen"]), fz["theta"]
assert cfg == ("E+D w=0.25", 0.5, "diag"), cfg
pos, E, D = t2.load("dev")
rows, nL, nC = [], 0, 0
for x in pos:
    if x["pop"] == "L":
        take = nL % 14 == 0
        nL += 1
    else:
        take = nC % 7 == 0
        nC += 1
    if not take:
        continue
    r = t2.ranked(t2.gains(cfg, x, E, D))
    top, gain = (r[0] if r else ("", float("-inf")))
    rows.append({"id": x["id"], "pop": x["pop"], "rd": x["rd"], "p": x["p"], "intended": x["intended"], "cands": x["cands"],
                 "dir": x["dir"], "E": E[x["id"]], "D": D[x["id"]], "top": top, "gain": gain, "offered": bool(r) and gain >= theta})
fx = Path(__file__).resolve().parents[1] / "McBopomofoTests" / "SlothEFixtures" / "v32b_dev_t2.jsonl"
with open(fx, "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
Ls = [r for r in rows if r["pop"] == "L"]
Cs = [r for r in rows if r["pop"] == "C"]
print(f"rows {len(rows)} (L {len(Ls)}, C {len(Cs)}); theta {theta:.6f}; "
      f"C offered {sum(r['offered'] for r in Cs)} ({100 * sum(r['offered'] for r in Cs) / len(Cs):.1f} %); "
      f"L top-1 = intended and offered {sum(r['offered'] and r['top'] == r['intended'] for r in Ls)} ({100 * sum(r['offered'] and r['top'] == r['intended'] for r in Ls) / len(Ls):.1f} %) -> {fx}")
