"""Offline decoder-stage harness: in-walk nodes + decoder requests -> pins -> final text (no re-walk)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import evalcfg  # noqa: E402
import json, math
REPO = evalcfg.REPO
FX = REPO + "/McBopomofoTests/SlothEFixtures"
UNK_LOG_SHARE = 9.4155  # vocab.unkLogShare() asserted in SlothEInWalkTests
PEN = -15.0

def load_vocab():
    cid = {}
    for l in open(REPO + "/SlothE/Bundle/SlothE/char2id.tsv", encoding="utf-8"):
        c, i = l.rstrip("\n").split("\t"); cid[c] = int(i)
    return cid

def load_variants():
    cls = {}
    for l in open(REPO + "/SlothE/Bundle/SlothE/variants.tsv", encoding="utf-8"):
        f = l.split()
        for c in f: cls[c] = f[0]
    return cls

def is_variant_pair(a, b, cls):
    if a == b or len(a) != len(b): return False
    for x, y in zip(a, b):
        if x == y: continue
        if x not in cls or cls.get(x) != cls.get(y): return False
    return True

def parse_cache(line):
    f = line.split("\t")
    readings = f[2].split(" ")
    pos = []
    for t in range(len(readings)):
        tok = f[3 + t].split(" ")
        lp = {}
        unk = None if tok[0] == "x" else float(tok[0])
        for k in range(1, len(tok) - 1, 2): lp[tok[k]] = float(tok[k + 1])
        vals = list(lp.values()) + ([unk] if unk is not None else [])
        mx = max(vals); z = mx + math.log(sum(math.exp(v - mx) for v in vals))
        pos.append((lp, unk, z))
    return readings, pos

def char_lp(pos, t, ch, cid):
    lp, unk, z = pos[t]
    if ch in lp: return lp[ch] - z
    if cid.get(ch) == 2 and unk is not None: return unk - z - UNK_LOG_SHARE
    return float("-inf")

def enc_score(pos, start, value, cid):
    s = 0.0
    for k, ch in enumerate(value):
        v = char_lp(pos, start + k, ch, cid)
        s += v if math.isfinite(v) else PEN
    return s

def decide(enc, dec, lam, tau, values, shown, cls):
    n = len(values)
    if n < 2 or any((not math.isfinite(d)) for d in dec): return None
    b = [e + lam * d for e, d in zip(enc, dec)]
    best = max(range(n), key=lambda k: (b[k], -k))
    p = 1.0 / sum(math.exp(x - b[best]) for x in b)
    v = values[best]
    if v == shown or p < tau or is_variant_pair(v, shown, cls): return None
    return v

def load_requests():
    """Per sentence: in-walk nodes and one request per decoder call (context, values, start, len, shown, q35 scores)."""
    caches = {d["sid"]: d["cache"] for d in map(json.loads, open(FX + "/v302_dev_walks.jsonl", encoding="utf-8"))}
    out = []
    for d in map(json.loads, open(FX + "/v2_parity.jsonl", encoding="utf-8")):
        nodes = d["inwalk"]; calls = d["calls"]
        reqs = []; ctx = ""; ci = 0
        for (s, e, v) in nodes:
            if ci < len(calls) and calls[ci][0] == ctx and len(calls[ci][1][0]) == e - s:
                c = calls[ci]; reqs.append({"ctx": ctx, "values": c[1], "q35": c[2], "start": s, "len": e - s, "shown": v}); ci += 1
            ctx += v
        assert ci == len(calls), (d["sid"], ci, len(calls))
        out.append({"sid": d["sid"], "readings": d["readings"], "nodes": nodes, "final": d["final"], "pins": d["pins"],
                    "reqs": reqs, "cache": caches[d["sid"]]})
    return out
