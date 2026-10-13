import sys, json, collections; sys.path.insert(0, '.')
from harness import *
from d512 import log_prob
cid = load_vocab(); cls = load_variants()
gold = {r['sid']: r for r in map(json.loads, open('dev_gold.jsonl', encoding='utf-8'))}
caches = {d['sid']: d['cache'] for d in map(json.loads, open(FX + '/v302_dev_walks.jsonl', encoding='utf-8'))}
ip = {d['sid']: d for d in map(json.loads, open(FX + '/inwalk_parity.jsonl', encoding='utf-8'))}
fin = {d['sent_id']: ''.join(n[2] for n in d['walk_nodes']) for d in map(json.loads, open(FX + '/walk2_dev_final2_Aprime.walks.jsonl', encoding='utf-8'))}
stats = collections.Counter(); rows = []
for sid, g in gold.items():
    R = g['readings']; G = g['gold']
    _, pos = parse_cache(caches[sid])
    for i, r in enumerate(R):
        if r != 'ㄗㄞˋ' or G[i] not in '在再': continue
        enc = {c: char_lp(pos, i, c, cid) for c in '在再'}
        enc_pick = max(enc, key=enc.get)
        # D512: left gold context, this syllable only (what a 1-char node gets)
        d1 = {c: log_prob(G[:i], [r], c) for c in '在再'}
        # D512: left context + this syllable + the next word (2 syllables), next char from gold
        d2 = {c: log_prob(G[:i], R[i:i+2], c + G[i+1]) for c in '在再'} if i + 1 < len(R) else d1
        row = dict(sid=sid, i=i, gold=G[i], sent=G, stock=ip[sid]['stock_text'][i], final=fin[sid][i], enc=enc_pick,
                   d1=max(d1, key=d1.get), d2=max(d2, key=d2.get), end=i == len(R) - 1)
        rows.append(row)
        for k in ('stock', 'final', 'enc', 'd1', 'd2'): stats[k] += row[k] == row['gold']
n = len(rows)
print('ㄗㄞˋ cases in dev:', n, collections.Counter(r['gold'] for r in rows))
for k, name in [('stock', 'stock McBopomofo'), ('final', 'current McBopomofoLM (whole sentence)'), ('enc', 'our encoder alone'),
                ('d1', 'D512: left context + ㄗㄞˋ only'), ('d2', 'D512: left context + ㄗㄞˋ + next syllable')]:
    print(f'{name:45s} {stats[k]}/{n}')
for r in rows:
    if r['final'] != r['gold'] or r['d1'] != r['gold']:
        print(r['sid'], r['sent'], 'gold', r['gold'], 'final', r['final'], 'enc', r['enc'], 'd1', r['d1'], 'd2', r['d2'])
