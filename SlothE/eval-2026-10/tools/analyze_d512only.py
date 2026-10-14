"""Compare D512 standalone beam decoding with the frozen configs. usage: analyze_d512only.py labeled.jsonl out.json d512only.json"""
import json, random, sys, collections
rows = {r["id"]: r for r in map(json.loads, open(sys.argv[1], encoding="utf-8"))}
out = json.load(open(sys.argv[2], encoding="utf-8")); d = json.load(open(sys.argv[3], encoding="utf-8"))
out["D512only"] = {i: (d.get(i) or "") for i in out["ids"]}
none = sum(1 for i in out["ids"] if not d.get(i))
gen = [i for i in out["ids"] if "general" in rows[i]["pools"]]
def cs(i, c): g = rows[i]["han"]; t = out[c][i]; return sum(a == b for a, b in zip(g, t)), len(g), g == t
print(f"general n={len(gen)}; D512only undecodable units (counted wrong): {none}")
for c in ["A", "B", "C", "D", "E", "D512only"]:
    ok = sum(cs(i, c)[0] for i in gen); ch = sum(cs(i, c)[1] for i in gen); s = sum(cs(i, c)[2] for i in gen)
    print(f"  {c:9s} char={ok/ch*100:.2f}%  sent={s/len(gen)*100:.1f}%")
def boot(a, b, n=2000):
    rnd = random.Random(1); da = [cs(i, a) for i in gen]; db = [cs(i, b) for i in gen]; ds = []
    for _ in range(n):
        idx = [rnd.randrange(len(gen)) for _ in gen]; ch = sum(da[k][1] for k in idx)
        ds.append((sum(db[k][0] for k in idx) - sum(da[k][0] for k in idx)) / ch * 100)
    ds.sort(); pt = (sum(x[0] for x in db) - sum(x[0] for x in da)) / sum(x[1] for x in da) * 100
    return pt, ds[50], ds[1949]
for a in ["B", "D"]:
    pt, lo, hi = boot(a, "D512only"); print(f"  D512only vs {a}: {pt:+.2f}pt [{lo:+.2f}, {hi:+.2f}]")
pos = [(i, k) for i in out["ids"] if "zai" in rows[i]["pools"] for k, (c, r) in enumerate(zip(rows[i]["han"], rows[i]["readings"])) if r == "ㄗㄞˋ" and c in "在再"]
for c in ["B", "D", "D512only"]:
    m = collections.Counter((rows[i]["han"][k], out[c][i][k] if k < len(out[c][i]) else "?") for i, k in pos)
    nz = sum(1 for i, k in pos if rows[i]["han"][k] == "再")
    print(f"  ㄗㄞˋ {c:9s} acc={sum(v for (g,p),v in m.items() if g==p)/len(pos)*100:.1f}%  再 recall={m[('再','再')]}/{nz} ({m[('再','再')]/nz*100:.0f}%)  在→再 {m[('在','再')]}")
