"""Analyze a frozen fresh run. usage: analyze_fresh.py labeled.jsonl out.json [exclude_ids.txt]"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import evalcfg  # noqa: E402
import sys, json, random, math, collections
sys.path.insert(0, evalcfg.WORK)
import breakdown
rows = {r["id"]: r for r in map(json.loads, open(sys.argv[1], encoding="utf-8"))}
out = json.load(open(sys.argv[2], encoding="utf-8"))
excl = set(open(sys.argv[3]).read().split()) if len(sys.argv) > 3 else set()
CF = ["A", "inwalk", "B", "C", "D", "E"]
gen = [i for i in out["ids"] if "general" in rows[i]["pools"] and i not in excl]
def cs(i, c): g = rows[i]["han"]; t = out[c][i]; return sum(a == b for a, b in zip(g, t)), len(g), g == t
print(f"general pool n={len(gen)} units, {sum(len(rows[i]['han']) for i in gen)} chars")
for c in CF:
    ok = sum(cs(i, c)[0] for i in gen); ch = sum(cs(i, c)[1] for i in gen); s = sum(cs(i, c)[2] for i in gen)
    print(f"  {c:7s} char={ok/ch*100:.2f}%  sent={s/len(gen)*100:.1f}%")
def boot(a, b, units, n=2000):
    rnd = random.Random(1); da = [cs(i, a) for i in units]; db = [cs(i, b) for i in units]
    diffs = []
    for _ in range(n):
        idx = [rnd.randrange(len(units)) for _ in units]
        ch = sum(da[k][1] for k in idx); diffs.append((sum(db[k][0] for k in idx) - sum(da[k][0] for k in idx)) / ch * 100)
    diffs.sort(); pt = (sum(x[0] for x in db) - sum(x[0] for x in da)) / sum(x[1] for x in da) * 100
    fx = sum(1 for k in range(len(units)) for p, q, g in zip(out[a][units[k]], out[b][units[k]], rows[units[k]]["han"]) if p != g and q == g)
    br = sum(1 for k in range(len(units)) for p, q, g in zip(out[a][units[k]], out[b][units[k]], rows[units[k]]["han"]) if p == g and q != g)
    return pt, diffs[int(.025 * n)], diffs[int(.975 * n)], fx, br
for a, b in [("B", "C"), ("C", "D"), ("D", "E"), ("A", "B"), ("B", "E")]:
    pt, lo, hi, fx, br = boot(a, b, gen)
    print(f"  {b} vs {a}: {pt:+.2f}pt  95% CI [{lo:+.2f}, {hi:+.2f}]  fixed {fx} / broken {br} chars")
print("error breakdown (general pool; in-span = gold is a unigram on the chosen span):")
for c in ["inwalk"]:
    a = b = 0
    for i in gen:
        x, y = breakdown.classify(out["nodes_inwalk"][i], rows[i]["readings"], rows[i]["han"]); a += x; b += y
    print(f"  in-walk wrong chars: in-span {a}, needs re-segmentation {b}")
req = [out["nreq"][i] for i in gen]; print(f"decoder requests per unit: mean {sum(req)/len(req):.2f}, max {max(req)}")
# ㄗㄞˋ pool
pos = [(i, k) for i in out["ids"] if "zai" in rows[i]["pools"] and i not in excl for k, (c, r) in enumerate(zip(rows[i]["han"], rows[i]["readings"])) if r == "ㄗㄞˋ" and c in "在再"]
print(f"ㄗㄞˋ positions: {len(pos)} (在 {sum(rows[i]['han'][k]=='在' for i,k in pos)}, 再 {sum(rows[i]['han'][k]=='再' for i,k in pos)})")
for c in CF:
    m = collections.Counter((rows[i]["han"][k], out[c][i][k] if k < len(out[c][i]) else "?") for i, k in pos)
    acc = sum(v for (g, p), v in m.items() if g == p) / len(pos)
    print(f"  {c:7s} acc={acc*100:.1f}%  在→在 {m[('在','在')]} 在→再 {m[('在','再')]} 再→再 {m[('再','再')]} 再→在 {m[('再','在')]} other {sum(v for (g,p),v in m.items() if p not in '在再')}")
def mcnemar(a, b):
    n01 = sum(1 for i, k in pos if out[a][i][k] != rows[i]["han"][k] and out[b][i][k] == rows[i]["han"][k])
    n10 = sum(1 for i, k in pos if out[a][i][k] == rows[i]["han"][k] and out[b][i][k] != rows[i]["han"][k])
    n = n01 + n10; p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, j) for j in range(0, min(n01, n10) + 1)) / 2 ** n)
    return n01, n10, p
for a, b in [("C", "D"), ("B", "D"), ("B", "C"), ("D", "E")]:
    n01, n10, p = mcnemar(a, b); print(f"  McNemar {b} vs {a}: {b} fixes {n01}, breaks {n10}, p={p:.3g}")
