import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import evalcfg  # noqa: E402
import sys, json; sys.path.insert(0, '.')
import evalpipe, pipeline
FX = evalcfg.FX
gold = {r['sid']: r for r in map(json.loads, open('dev_gold.jsonl', encoding='utf-8'))}
items = [(sid, r['readings'], r['gold'], '') for sid, r in gold.items()]
fw = evalpipe.first_walks(items)
def show(name, outs):
    ok, ch, s, n = evalpipe.acc(items, outs); print(f'{name:44s} char={ok/ch*100:.2f}% sent={s/n*100:.1f}%', flush=True)
show('stock McBopomofo', {i: pipeline.text(fw[i]['stock']) for i, *_ in items})
show('in-walk only (no decoder)', {i: pipeline.text(fw[i]['inwalk']) for i, *_ in items})
for name, spec, lam, tau, seq in [('current: q35 λ2 τ.5', 'q35', 2, .5, False), ('q35 + sequential context', 'q35', 2, .5, True),
                                  ('D512 λ2 τ.5', 'd512', 2, .5, False), ('D512 λ2 τ.5 + sequential', 'd512', 2, .5, True),
                                  ('q35+0.25·D512 λ2 τ.6', ('both', .25), 2, .6, False), ('q35+0.25·D512 λ2 τ.6 + sequential', ('both', .25), 2, .6, True)]:
    show(name, evalpipe.run(items, spec, lam, tau, seq))
