import sys, json, ast, collections; sys.path.insert(0, '.')
from harness import FX
from d512 import log_prob
gold = {r['sid']: r for r in map(json.loads, open('dev_gold.jsonl', encoding='utf-8'))}
st = collections.Counter(); n = collections.Counter(); bad = []
for d in map(json.loads, open(FX + '/v31a_dev_chunks.jsonl', encoding='utf-8')):
    G = gold[d['sid']]['gold']; R = d['readings']
    committed = ''
    for (s, e), ce in zip(d['chunks'], d['Ce1']):
        fin = ''.join(v for _, _, v in ce['final'])
        for i in range(s, e):
            if R[i] != 'ㄗㄞˋ' or G[i] not in '在再': continue
            k = i - s; where = 'chunk end' if i == e - 1 else 'mid chunk'
            left = committed + fin[:k]
            sc = {c: log_prob(left, R[i:e], c + fin[k + 1:]) for c in '在再'}
            dp = max(sc, key=sc.get)
            n[where] += 1; st[(where, 'ours')] += fin[k] == G[i]; st[(where, 'd512')] += dp == G[i]
            if fin[k] != G[i] or dp != G[i]:
                bad.append(f"{where:9s} {G}  chunk={fin!r} gold={G[i]} ours={fin[k]} d512={dp}")
        committed += fin
for w in n:
    print(f"{w:10s} n={n[w]:2d}  ours(v3.1 carried ctx)={st[(w,'ours')]}  D512(same info)={st[(w,'d512')]}")
print('\n'.join(bad))
