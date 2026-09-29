"""v3.1b parity fixture: the frozen illegal-window rule on the dev illegal trials (reports/typo.md §8.4-8.5).

Reads (read-only) the study's own inputs through its own code: typo/t2_ill.load("dev") (T1 trials + slot1_nb
candidates, encoder E = results/typo/dev.enc.tsv lsm_p, left texts, decoder D = results/typo/t2/dev.ill.dec.jsonl)
and the frozen rule (typo/t2_ill_frozen.json D+E w=1 [1:1, other=w_diag]) + add-final bonus (typo/t2_fin_frozen.json
"ill" = 3, via t2_fin.bonus_for). Writes McBopomofoTests/SlothEFixtures/v31b_typo_dev.jsonl:
  {"ctx": id, "text": left text, "lp": {char: D}}                     one row per left text (D is per left text)
  {"typed", "ctx": id|null, "cands": [[reading, cmv, tch, nb, lm, E]], "want": [[reading, char]] (top 9),
   "gold": the intended char}                                         one row per illegal trial
"want" = t2_ill.rank's (syllable, char) pairs in its own order, deduplicated by char (first occurrence); its chars
are asserted equal to t2_ill.rank's char list. D is rounded to 6 decimals (fixture size); "want" is computed from
the rounded D, so the app and the study rank the same numbers.
usage: PYTHONDONTWRITEBYTECODE=1 nice -n 15 $W/sloth/.venv/bin/python SlothE/v31b/make_typo_fixture.py
"""
import json
import math
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(W / "typo"))
import t2_fin  # noqa: E402
import t2_ill  # noqa: E402

CFG = tuple(json.loads(t2_ill.FROZEN.read_text())["chosen"])
BONUS = json.loads(t2_fin.FROZEN.read_text())["chosen_bonus"]["ill"]
assert CFG == ("D+E", 1, 1, "diag") and BONUS == 3, (CFG, BONUS)
MAXN = 9   # the first page (the parity metric is top-9 identity)


def pairs(it):
    """t2_ill.rank with the syllable kept per char."""
    bon = it.get("bonus", {})
    t1, chars = {r: v + bon.get(r, 0.0) for r, v in it["t1"].items()}, it["chars"]
    if it["lp"] is None:
        syl = sorted(t1, key=lambda r: (-t1[r], r))
        ps = [(r, c) for r in syl for c in chars[r]]
    else:
        wd, o = CFG[-2], CFG[-1]
        lw = {"row": 0.0, "diag": math.log(wd), "other": math.log(wd) if o == "diag" else 0.0}
        keyed = []
        for r in t1:
            base = lw[it["dir"][r]] + CFG[1] * it["t1"][r] + bon.get(r, 0.0)
            vs = [it["lp"].get(c, math.log(1e-12)) for c in chars[r]] or [math.log(1e-12)]
            for c, v in zip(chars[r], vs):
                keyed.append((-(v + base), r, c))
        ps = [(r, c) for _, r, c in sorted(keyed)]
    out, seen = [], set()
    for r, c in ps:
        if c not in seen:
            seen.add(c)
            out.append([r, c])
    return out


prep = json.load(open(t2_ill.OUT / "dev.ill.prep.json"))
items = t2_ill.load("dev")
enc = t2_ill.T1.load_enc("dev")
rounded = {}
for it in items:
    if it["lp"] is not None:
        key = id(it["lp"])
        rounded.setdefault(key, {c: round(v, 6) for c, v in it["lp"].items()})
        it["lp"] = rounded[key]
ctx_id, rows, lps = {}, [], {}
for it in items:
    t = it["trial"]
    it["bonus"] = t2_fin.bonus_for(t["events"][0][0], it["t1"], BONUS)
    want = pairs(it)
    _, chars = t2_ill.rank(it, CFG)
    assert [c for _, c in want] == chars, "pairs() drifted from t2_ill.rank"
    cid = None
    if it["lp"] is not None:
        key = prep["left"][f"{t['s']}_{t['i']}"]
        cid = ctx_id.setdefault(key, len(ctx_id))
    cands = []
    for r, cmv, tch, nb, lm in t["cands"]:
        if cmv + tch <= 1 or nb:
            cands.append([r, cmv, tch, nb, lm, enc[(t["s"], t["i"], r)]["lsm_p"]])
    rows.append(dict(typed=t["events"][0][0], ctx=cid, cands=cands, want=want[:MAXN], gold=it["ch"]))
    lps[cid] = it["lp"]
fx = W / "McBopomofoLM" / "McBopomofoTests" / "SlothEFixtures" / "v31b_typo_dev.jsonl"
with open(fx, "w", encoding="utf-8") as f:
    for text, k in sorted(ctx_id.items(), key=lambda x: x[1]):
        f.write(json.dumps(dict(ctx=k, text=text, lp=lps[k]), ensure_ascii=False) + "\n")
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"trials {len(rows)} (with left text {sum(r['ctx'] is not None for r in rows)}), left texts {len(ctx_id)} -> {fx}")
