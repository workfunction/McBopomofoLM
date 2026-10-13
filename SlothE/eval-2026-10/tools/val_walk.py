import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import evalcfg  # noqa: E402
import json, sys, numpy as np, subprocess
S = sys.argv[1]; _os.makedirs(S + "/val", exist_ok=True); FX = evalcfg.FX
cid = {}
for l in open(_os.path.join(evalcfg.BUNDLE, "char2id.tsv"), encoding="utf-8"):
    c, i = l.rstrip("\n").split("\t"); cid[c] = int(i)
C = 8342; lines = []; want = {}; stock = {}
for d in map(json.loads, open(FX + "/inwalk_parity.jsonl", encoding="utf-8")): stock[d["sid"]] = d["stock_text"]
for d in map(json.loads, open(FX + "/v302_dev_walks.jsonl", encoding="utf-8")):
    f = d["cache"].split("\t"); R = f[2].split(" ")
    x = np.full((len(R), C), -1e30, dtype=np.float32)
    for t in range(len(R)):
        tok = f[3 + t].split(" ")
        if tok[0] != "x": x[t, 2] = float(tok[0])
        for k in range(1, len(tok) - 1, 2):
            if tok[k] in cid: x[t, cid[tok[k]]] = float(tok[k + 1])
    p = f"{S}/val/{d['sid']}.f32"; x.tofile(p)
    pins = ";".join(f"{s},{l},{v}" for s, l, v in d["pins"]) or "-"
    lines.append(f"{d['sid']}\t{' '.join(R)}\t{p}\t{pins}")
    want[d["sid"]] = (d["walk"], bool(d["pins"]))
out = subprocess.run([evalcfg.WALK_CLI, evalcfg.DATA_TXT, evalcfg.BUNDLE],
                     input="\n".join(lines) + "\n", capture_output=True, text=True).stdout
ok = sk = 0; n = 0
for r in map(json.loads, out.splitlines()):
    w, has = want[r["id"]]; got = r["pinned"] if has else r["inwalk"]
    ok += got == w; n += 1
    sk += "".join(v for _, _, v in r["stock"]) == stock[r["id"]]
print(f"final walk matches fixture: {ok}/{n}; stock walk matches fixture: {sk}/{n}")
