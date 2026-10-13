import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import evalcfg  # noqa: E402
import sys, json, collections; sys.path.insert(0, '.')
import pipeline
uni = {}; maxlen = 1
for l in open(evalcfg.DATA_TXT, encoding='utf-8'):
    f = l.split()
    if len(f) == 3 and not f[0].startswith(('_', '#')): uni.setdefault(f[0], set()).add(f[1])
def word_cover(H, R, k):
    """is there a dictionary word (len>=2, matching reading) in gold covering position k?"""
    for n in range(2, 7):
        for s in range(max(0, k - n + 1), min(k, len(H) - n) + 1):
            if H[s:s + n] in uni.get('-'.join(R[s:s + n]), ()): return True
    return False
for dom in ['chat', 'news']:
    rows = {r['id']: r for r in map(json.loads, open(f'../fresh/{dom}_lab.jsonl', encoding='utf-8'))}
    out = json.load(open(f'../fresh/{dom}_out.json', encoding='utf-8'))
    gen = [i for i in out['ids'] if 'general' in rows[i]['pools']]
    fw = pipeline.walks([(i, rows[i]['readings'], []) for i in gen])
    c = collections.Counter(); ex = collections.defaultdict(list)
    for i in gen:
        H, R = rows[i]['han'], rows[i]['readings']; T = out['C'][i]
        assert pipeline.text(fw[i]['inwalk']) == out['inwalk'][i]
        reqs = {q['start']: q for q in fw[i]['requests']}
        for s, e, v in fw[i]['inwalk']:
            for k in range(s, e):
                if k >= len(T) or T[k] == H[k]: continue
                g = H[s:e]
                if g in uni.get('-'.join(R[s:e]), ()):
                    q = reqs.get(s)
                    kind = 'in-span: no decoder request' if q is None else ('in-span: gold in top-3, decoder kept wrong' if g in q['values'] else 'in-span: gold outside encoder top-3')
                else:
                    kind = 'reseg: gold word in dictionary (frequency/segmentation)' if word_cover(H, R, k) else 'reseg: gold has no dictionary word here (coverage)'
                c[kind] += 1
                if len(ex[kind]) < 5: ex[kind].append(f'{H} -> {T}')
    print(f'== {dom}: config C remaining wrong chars on general pool = {sum(c.values())}')
    for k, v in c.most_common(): print(f'  {v:5d}  {k}   e.g. {ex[k][:2]}')
