"""Offline McBopomofoLM v3 pipeline (whole buffer, no carried encoder context):
stock walk -> 25M encoder -> in-walk walk -> q35 decoder pins (lambda 2, tau 0.5, top-3) -> pinned walk.
Optional extra scorer (D512) can replace or join the decoder.  Validated: walk 500/500, decoder decisions 1103/1104."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import evalcfg  # noqa: E402
import hashlib, json, os, subprocess, sys, tempfile
import numpy as np, torch
W = os.path.dirname(os.path.abspath(__file__))
S = os.path.dirname(W)
sys.path.insert(0, W)
from sloth_enc import model as ENC, SV  # noqa
from harness import load_variants, decide  # noqa
CLS = load_variants()
CLI = [evalcfg.WALK_CLI, evalcfg.DATA_TXT, evalcfg.BUNDLE]
_tmp = tempfile.mkdtemp(dir=evalcfg.WORK)

def sid_of(r):
    if r in SV: return SV[r]
    return SV.get("".join(c for c in r if c not in "ˊˇˋ˙"), 1)  # Vocabulary::mapSyllable toneless fallback

def logits(readings):
    sid = torch.tensor([[sid_of(r) for r in readings]])
    with torch.no_grad():
        return ENC(sid, torch.ones_like(sid, dtype=torch.bool))[0].float().numpy()

def walks(items):
    """items: [(id, readings, pins list)] -> {id: cli json}"""
    lines = []
    for k, (i, R, pins) in enumerate(items):
        p = f"{_tmp}/{hashlib.sha1(' '.join(R).encode()).hexdigest()}.f32"
        if not os.path.exists(p): logits(R).astype(np.float32).tofile(p)
        ps = ";".join(f"{s},{l},{v}" for s, l, v in pins) or "-"
        lines.append(f"{i}\t{' '.join(R)}\t{p}\t{ps}")
    out = subprocess.run(CLI, input="\n".join(lines) + "\n", capture_output=True, text=True, check=True).stdout
    return {d["id"]: d for d in map(json.loads, out.splitlines())}

text = lambda nodes: "".join(v for _, _, v in nodes)

def run(items, scorer, lam=2.0, tau=0.5):
    """items: [(id, readings, decoder_context)]; scorer(ctx, values, readings, start, len) -> scores or None."""
    first = walks([(i, R, []) for i, R, _ in items])
    pinsets = {}
    for i, R, dctx in items:
        pins = []
        for q in first[i]["requests"]:
            sc = scorer(dctx + q["ctx"], q["values"], R, q["start"], q["len"])
            if sc is None: continue
            v = decide(q["enc"], sc, lam, tau, q["values"], q["shown"], CLS)
            if v: pins.append((q["start"], q["len"], v))
        pinsets[i] = pins
    second = walks([(i, R, pinsets[i]) for i, R, _ in items])
    res = {}
    for i, R, _ in items:
        f = second[i]["pinned"] or first[i]["inwalk"]
        res[i] = {"stock": text(first[i]["stock"]), "inwalk": text(first[i]["inwalk"]), "final": text(f),
                  "requests": first[i]["requests"], "pins": pinsets[i]}
    return res
