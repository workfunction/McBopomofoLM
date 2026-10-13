import sys, json, math, hashlib, itertools; sys.path.insert(0, '.')
from harness import load_variants, decide
cls = load_variants()
gold = {r['sid']: r['gold'] for r in map(json.loads, open('dev_gold.jsonl', encoding='utf-8'))}
data = [json.loads(l) for l in open('d512_scores.jsonl', encoding='utf-8')]
fold = {d['sid']: int(hashlib.sha256(d['sid'].encode()).hexdigest(), 16) % 2 for d in data}

def run(score_fn, lam, tau, subset=None):
    ch = ok = sent = 0; outs = {}
    for d in data:
        if subset is not None and fold[d['sid']] != subset: continue
        text = [v for _, _, v in d['nodes']]; idx = {s: i for i, (s, _, _) in enumerate(d['nodes'])}
        for r in d['reqs']:
            dec = score_fn(r)
            if dec is None or any(x is None for x in dec): continue
            v = decide(r['enc'], dec, lam, tau, r['values'], r['shown'], cls)
            if v: text[idx[r['start']]] = v
        t = ''.join(text); g = gold[d['sid']]; outs[d['sid']] = t
        ch += len(g); ok += sum(a == b for a, b in zip(g, t)); sent += g == t
    return ok, ch, sent, outs

FNS = {
    'q35 (current)': lambda r: r['q35'],
    'D512 span': lambda r: r['d_span'],
    'D512 full-sentence': lambda r: r['d_full'],
    'q35 + D512 span': None, 'q35 + D512 full': None,
}
LAMS = [0.25, 0.5, 1, 2, 3, 5]; TAUS = [0.34, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
def combo(key):  # enc + 2*q35 + lam*d512  ->  pass as dec with lam folded: dec' = q35 + (lam2/2)*d
    return lambda lam2: (lambda r: None if any(x is None for x in r[key]) else [q + lam2 / 2 * x for q, x in zip(r['q35'], r[key])])

base = run(FNS['q35 (current)'], 2.0, 0.5)
def fmt(ok, ch, sent, n): return f'char={ok/ch*100:.2f}% sent={sent/n*100:.1f}%'
print('stock walk / in-walk only are in the earlier table')
print(f"{'q35 (current, λ=2 τ=.5)':34s} {fmt(base[0], base[1], base[2], 500)}")
for name in ['D512 span', 'D512 full-sentence']:
    r = run(FNS[name], 2.0, 0.5); print(f"{name+' @λ=2 τ=.5':34s} {fmt(r[0], r[1], r[2], 500)}")

def cv(name, make):  # make(lam) -> (score_fn, lam_eff)
    tot = [0, 0, 0]; picks = []; outs = {}
    for test in (0, 1):
        train = 1 - test
        best = max(((lam, tau) for lam in LAMS for tau in TAUS),
                   key=lambda p: run(make(p[0])[0], make(p[0])[1], p[1], train)[0])
        picks.append(best)
        ok, ch, sent, o = run(make(best[0])[0], make(best[0])[1], best[1], test)
        tot[0] += ok; tot[1] += ch; tot[2] += sent; outs.update(o)
    # fixes / harms vs current
    fx = hm = 0
    for sid, t in outs.items():
        g = gold[sid]; b = base[3][sid]
        fx += sum(x == y != z for x, y, z in zip(g, t, b)); hm += sum(x == z != y for x, y, z in zip(g, t, b))
    print(f"{name+' [2-fold CV]':34s} {fmt(tot[0], tot[1], tot[2], 500)}  picks(λ,τ)={picks}  vs current: +{fx} fixed / -{hm} broken chars")
print('--- tuned with 2-fold CV (tune on one half, report on the other) ---')
cv('q35 (current)', lambda lam: (FNS['q35 (current)'], lam))
cv('D512 span', lambda lam: (FNS['D512 span'], lam))
cv('D512 full-sentence', lambda lam: (FNS['D512 full-sentence'], lam))
cv('q35 + D512 span', lambda lam: (combo('d_span')(lam), 2.0))
cv('q35 + D512 full', lambda lam: (combo('d_full')(lam), 2.0))
