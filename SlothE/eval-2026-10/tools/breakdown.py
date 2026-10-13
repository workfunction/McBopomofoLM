"""Classify wrong chars: in-span (gold span value is a unigram on the chosen node's span) vs needs re-segmentation."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import evalcfg  # noqa: E402
import json
DATA = evalcfg.DATA_TXT
_uni = None
def unigrams():
    global _uni
    if _uni is None:
        _uni = {}
        for l in open(DATA, encoding="utf-8"):
            f = l.split()
            if len(f) == 3 and not f[0].startswith(("_", "#")): _uni.setdefault(f[0], set()).add(f[1])
    return _uni
def classify(nodes, readings, gold):
    """nodes [[s,e,v]] -> (in_span_wrong, seg_wrong) char counts"""
    u = unigrams(); a = b = 0
    for s, e, v in nodes:
        g = gold[s:e]
        if len(v) != e - s: continue
        w = sum(x != y for x, y in zip(v, g))
        if not w: continue
        if g in u.get("-".join(readings[s:e]), ()): a += w
        else: b += w
    return a, b
