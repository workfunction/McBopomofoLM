import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import evalcfg  # noqa: E402
import sys, json; sys.path.insert(0, '.')
import evalpipe, pipeline
gold = {r['sid']: r for r in map(json.loads, open('dev_gold.jsonl', encoding='utf-8'))}
items = [(sid, r['readings'], r['gold'], '') for sid, r in gold.items()]
base = {}
for name, data in [('data.txt', evalcfg.DATA_TXT), ('twins', sys.argv[1])]:
    pipeline.CLI[1] = data; evalpipe._first.clear()
    for v, seq in [('q35', False), ('q35+seq', True)]:
        o = evalpipe.run(items, 'q35', 2, .5, seq); ok, ch, s, n = evalpipe.acc(items, o)
        print(f'{name:8s} {v:8s} char={ok/ch*100:.2f}% sent={s/n*100:.1f}%', flush=True)
        if name == 'data.txt': base[v] = o
        else:
            for i, R, g, _ in items:
                if o[i] != base[v][i]: print('   changed', i, base[v][i], '->', o[i], '| gold', g)
