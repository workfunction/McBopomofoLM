import sys, json, hashlib; sys.path.insert(0, '.')
import evalpipe, pipeline
gold = {r['sid']: r for r in map(json.loads, open('dev_gold.jsonl', encoding='utf-8'))}
items = [(sid, r['readings'], r['gold'], '') for sid, r in gold.items()]
fold = {i: int(hashlib.sha256(i.encode()).hexdigest(), 16) % 2 for i, *_ in items}
pipeline.CLI[1] = 'data_twins.txt'
res = {}
for spec in ['q35', ('both', .25), ('both', .5)]:
    for lam in [0.5, 1, 1.5]:
        for tau in [0.4, 0.5, 0.6]:
            o = evalpipe.run(items, spec, lam, tau, True, True)
            res[(str(spec), lam, tau)] = o
            ok, ch, s, n = evalpipe.acc(items, o)
            print(f'{str(spec):14s} λ={lam} τ={tau} char={ok/ch*100:.2f}% sent={s/n*100:.1f}%', flush=True)
for spec in ['q35', str(('both', .25)), str(('both', .5))]:
    tot = [0, 0, 0]; picks = []
    for test in (0, 1):
        tr = {i for i in fold if fold[i] != test}; te = {i for i in fold if fold[i] == test}
        best = max((k for k in res if k[0] == spec), key=lambda k: evalpipe.acc(items, res[k], tr)[0])
        ok, ch, s, n = evalpipe.acc(items, res[best], te); tot[0] += ok; tot[1] += ch; tot[2] += s; picks.append(best[1:])
    print(f'CV {spec:14s} char={tot[0]/tot[1]*100:.2f}% sent={tot[2]/500*100:.1f}% picks={picks}')
