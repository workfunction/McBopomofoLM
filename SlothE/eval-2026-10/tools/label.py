"""Label units: g2pW bopomofo -> readings snapped to McBopomofo-legal single-char readings.
Phrase override: where the gold text max-matches a multi-char data.txt phrase, use the phrase's reading
(closest to g2pW when it has several). 一->ㄧ, 不->ㄅㄨˋ (what users type).
Outcomes: label_illegal (dropped) | kept (reachable by construction: every char has a legal single-char unigram).
usage: python label.py in.jsonl out.jsonl"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import evalcfg  # noqa: E402
import json, sys
from g2pw import G2PWConverter
MODEL = evalcfg.G2PW_MODEL
TONE = {"1": "", "2": "ˊ", "3": "ˇ", "4": "ˋ", "5": "˙"}
TM = "ˊˇˋ˙"
base = lambda r: "".join(c for c in r if c not in TM)
inp, outp = sys.argv[1], sys.argv[2]
uni, phr = {}, {}
for l in open(evalcfg.DATA_TXT, encoding="utf-8"):
    f = l.split()
    if len(f) != 3 or f[0].startswith(("_", "#")): continue
    rs = f[0].split("-")
    if len(rs) != len(f[1]): continue
    if len(f[1]) == 1: uni.setdefault(f[1], set()).add(f[0])
    elif len(f[1]) <= 6: phr.setdefault(f[1], set()).add(tuple(rs))
conv = G2PWConverter(model_dir=MODEL, style="bopomofo", enable_non_tradional_chinese=True)
rows = [json.loads(l) for l in open(inp, encoding="utf-8")]
st = {"in": len(rows), "g2pw_none": 0, "label_illegal": 0, "kept": 0, "snapped": 0, "sandhi": 0, "phrase_override": 0}
def snap(c, rd):
    L = uni.get(c, set())
    if rd in L: return rd, False
    same = [x for x in L if base(x) == base(rd)]
    if len(same) == 1: return same[0], True
    return None, False
with open(outp, "w", encoding="utf-8") as fo:
    for k in range(0, len(rows), 64):
        batch = rows[k:k + 64]
        for r, py in zip(batch, conv([x["han"] for x in batch])):
            H = r["han"]
            if len(py) != len(H) or any(p is None for p in py): st["g2pw_none"] += 1; continue
            R = []
            for c, p in zip(H, py):
                rd = p[:-1] + TONE[p[-1]] if p[-1] in TONE else p
                if c == "一" and rd != "ㄧ": rd = "ㄧ"; st["sandhi"] += 1
                if c == "不" and rd == "ㄅㄨˊ": rd = "ㄅㄨˋ"; st["sandhi"] += 1
                R.append(rd)
            # phrase override by greedy longest match
            i = 0
            while i < len(H):
                for n in range(min(6, len(H) - i), 1, -1):
                    cands = phr.get(H[i:i + n])
                    if cands:
                        best = max(cands, key=lambda t: sum(a == b for a, b in zip(t, R[i:i + n])))
                        if list(best) != R[i:i + n]:
                            # keep 一/不 as typed
                            best = [("ㄧ" if H[i + j] == "一" else "ㄅㄨˋ" if H[i + j] == "不" and best[j] == "ㄅㄨˊ" else best[j]) for j in range(n)]
                            if best != R[i:i + n]: st["phrase_override"] += 1
                            R[i:i + n] = best
                        i += n; break
                else:
                    i += 1
            ok = True
            for j, (c, rd) in enumerate(zip(H, R)):
                s, did = snap(c, rd)
                if s is None: ok = False; break
                R[j] = s; st["snapped"] += did
            if not ok: st["label_illegal"] += 1; continue
            r["readings"] = R; st["kept"] += 1
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps(st))
