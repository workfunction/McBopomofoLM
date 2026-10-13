"""Evaluate decoder-stage variants with the offline pipeline.
items: [(id, readings, gold, decoder_ctx)]. Scores cached in score_cache.jsonl keyed by (scorer, ctx, span readings, values)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import evalcfg  # noqa: E402
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline
from harness import decide
CACHE_PATH = os.path.join(evalcfg.WORK, "score_cache.jsonl")
_cache = {}
if os.path.exists(CACHE_PATH):
    for l in open(CACHE_PATH, encoding="utf-8"):
        k, v = json.loads(l); _cache[k] = v
_out = open(CACHE_PATH, "a", encoding="utf-8")
_q35 = _d512 = None

def raw(name, ctx, span, vals):
    global _q35, _d512
    k = json.dumps([name, ctx[-64:], span, vals], ensure_ascii=False)
    if k not in _cache:
        if name == "q35":
            if _q35 is None: import q35 as _m; _q35 = _m
            v = _q35.score(ctx, vals)
        else:
            if _d512 is None: import d512 as _m; _d512 = _m
            v = [_d512.log_prob(ctx, span, x) for x in vals]
            v = None if None in v else v
        _cache[k] = v; _out.write(json.dumps([k, v], ensure_ascii=False) + "\n"); _out.flush()
    return _cache[k]

def scorer(spec):
    """spec: 'q35' | 'd512' | ('both', w) -> q35 + w*d512 (in decoder units, lambda applies on top)."""
    def f(ctx, vals, span):
        if spec == "q35": return raw("q35", ctx, span, vals)
        if spec == "d512": return raw("d512", ctx, span, vals)
        a, b = raw("q35", ctx, span, vals), raw("d512", ctx, span, vals)
        return None if a is None or b is None else [x + spec[1] * y for x, y in zip(a, b)]
    return f

_first = {}
def first_walks(items):
    need = [(i, R, []) for i, R, *_ in items if i not in _first]
    for k in range(0, len(need), 200):
        _first.update(pipeline.walks(need[k:k + 200]))
    return _first

def run(items, spec, lam, tau, seq, full=False):
    fw = first_walks(items); f = scorer(spec); pinsets = {}
    for i, R, gold, dctx in items:
        cur = list(pipeline.text(fw[i]["inwalk"])); pins = []
        for q in fw[i]["requests"]:
            ctx = dctx + ("".join(cur[:q["start"]]) if seq else q["ctx"])
            if full:  # whole-buffer scoring: candidate plus the text right of the node
                base = cur if seq else list(pipeline.text(fw[i]["inwalk"]))
                pre = dctx + "".join(base[:q["start"]]); post = "".join(base[q["start"] + q["len"]:])
                sc = f(pre, [v + post for v in q["values"]], R[q["start"]:])
            else:
                sc = f(ctx, q["values"], R[q["start"]:q["start"] + q["len"]])
            if sc is None: continue
            v = decide(q["enc"], sc, lam, tau, q["values"], q["shown"], pipeline.CLS)
            if v:
                pins.append((q["start"], q["len"], v))
                if seq and len(v) == q["len"] and len(cur) == len(R): cur[q["start"]:q["start"] + q["len"]] = list(v)
        pinsets[i] = pins
    out = {}
    need = [(i, R, pinsets[i]) for i, R, *_ in items if pinsets[i]]
    sec = {}
    for k in range(0, len(need), 200): sec.update(pipeline.walks(need[k:k + 200]))
    for i, R, gold, dctx in items:
        out[i] = pipeline.text(sec[i]["pinned"]) if i in sec else pipeline.text(fw[i]["inwalk"])
    return out

def acc(items, outs, subset=None):
    ch = ok = sent = n = 0
    for i, R, gold, _ in items:
        if subset is not None and i not in subset: continue
        t = outs[i]; n += 1; ch += len(gold); ok += sum(a == b for a, b in zip(gold, t)); sent += gold == t
    return ok, ch, sent, n
