import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import evalcfg  # noqa: E402
import sys, json; sys.path.insert(0, '.')
import evalpipe, pipeline
gold = {r['sid']: r for r in map(json.loads, open('dev_gold.jsonl', encoding='utf-8'))}
items = [(sid, r['readings'], r['gold'], '') for sid, r in gold.items()]
for name, data in [('data.txt', evalcfg.DATA_TXT), ('twins', 'data_twins.txt')]:
    pipeline.CLI[1] = data; evalpipe._first.clear()
    for v, spec, lam, tau, seq, full in [('q35 span (current)', 'q35', 2, .5, False, False), ('q35 full', 'q35', 2, .5, True, True),
                                         ('q35 full λ1', 'q35', 1, .5, True, True), ('D512 full', 'd512', 2, .5, True, True),
                                         ('D512 full λ1', 'd512', 1, .5, True, True), ('q35+.25D512 full', ('both', .25), 2, .6, True, True)]:
        o = evalpipe.run(items, spec, lam, tau, seq, full); ok, ch, s, n = evalpipe.acc(items, o)
        print(f'{name:8s} {v:20s} char={ok/ch*100:.2f}% sent={s/n*100:.1f}%', flush=True)
