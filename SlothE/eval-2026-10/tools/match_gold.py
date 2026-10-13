"""Recover gold text for the 500 dev sentences by matching their readings against Common Voice zh-TW text."""
import json, re, sys, collections
repo, cvdir, out = sys.argv[1:4]
TONES = "ˊˇˋ˙"
def base(s): return s.rstrip(TONES)
legal = collections.defaultdict(set)
for line in open(f"{repo}/Source/Data/BPMFBase.txt", encoding="utf-8"):
    f = line.split()
    if len(f) >= 2: legal[f[0]].add(f[1])
for line in open(f"{repo}/Source/Data/BPMFMappings.txt", encoding="utf-8"):
    f = line.split()
    if len(f) >= 2 and len(f[0]) == len(f) - 1:
        for c, r in zip(f[0], f[1:]): legal[c].add(r)
import glob
han = re.compile(r"[㐀-鿿豈-﫿]")
pool = collections.defaultdict(list)
for p in glob.glob(f"{cvdir}/*.txt"):
    for line in open(p, encoding="utf-8"):
        s = line.strip()
        t = "".join(han.findall(s))
        if t: pool[len(t)].append((t, s))
dev = [json.loads(l) for l in open(f"{repo}/McBopomofoTests/SlothEFixtures/inwalk_parity.jsonl", encoding="utf-8")]
ok = 0
with open(out, "w", encoding="utf-8") as fo:
    for d in dev:
        R = d["readings"]; best = None
        for t, raw in pool.get(len(R), []):
            exact = sum(r in legal[c] for c, r in zip(t, R))
            loose = sum(any(base(x) == base(r) for x in legal[c]) for c, r in zip(t, R))
            key = (loose, exact)
            if best is None or key > best[0]: best = (key, t, raw)
        good = best is not None and best[0][0] == len(R)
        ok += good
        fo.write(json.dumps({"sid": d["sid"], "readings": R, "gold": best[1] if good else None,
                             "raw": best[2] if good else None, "exact": best[0][1] if best else 0,
                             "stock": d["stock_text"]}, ensure_ascii=False) + "\n")
print("matched", ok, "/", len(dev))
