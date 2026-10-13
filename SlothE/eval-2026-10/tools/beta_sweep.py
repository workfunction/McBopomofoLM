import sys, json; sys.path.insert(0, '.')
import evalpipe, pipeline, breakdown
gold = {r['sid']: r for r in map(json.loads, open('dev_gold.jsonl', encoding='utf-8'))}
items = [(sid, r['readings'], r['gold'], '') for sid, r in gold.items()]
base = list(pipeline.CLI)
for beta in [0.3, 0.5, 0.7, 1.0]:
    pipeline.CLI = base + [str(beta)]; evalpipe._first.clear()
    fw = evalpipe.first_walks(items)
    iw = {i: pipeline.text(fw[i]['inwalk']) for i, *_ in items}
    a = b = 0
    for i, R, g, _ in items:
        x, y = breakdown.classify(fw[i]['inwalk'], R, g); a += x; b += y
    o1 = evalpipe.acc(items, iw)
    fin = evalpipe.run(items, 'q35', 2, .5, False); o2 = evalpipe.acc(items, fin)
    print(f'beta={beta}: in-walk char={o1[0]/o1[1]*100:.2f}% sent={o1[2]/500*100:.1f}% (wrong in-span {a}, needs-reseg {b}) | +q35 char={o2[0]/o2[1]*100:.2f}% sent={o2[2]/500*100:.1f}%', flush=True)
