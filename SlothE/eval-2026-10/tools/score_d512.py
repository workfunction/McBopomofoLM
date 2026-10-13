import sys, json, time; sys.path.insert(0, '.')
from harness import *
from d512 import log_prob
cid = load_vocab(); data = load_requests()
t0 = time.time(); n = miss = 0; lat = []
with open('d512_scores.jsonl', 'w', encoding='utf-8') as fo:
    for d in data:
        rd, pos = parse_cache(d['cache'])
        nodes = d['nodes']; R = d['readings']
        for r in d['reqs']:
            r['enc'] = [enc_score(pos, r['start'], v, cid) for v in r['values']]
            span = R[r['start']:r['start'] + r['len']]
            t1 = time.time()
            r['d_span'] = [log_prob(r['ctx'], span, v) for v in r['values']]
            lat.append((time.time() - t1) * 1000)
            # whole-sentence variant: no context, full reading, in-walk text with this node swapped
            pre = ''.join(x[2] for x in nodes if x[1] <= r['start']); post = ''.join(x[2] for x in nodes if x[0] >= r['start'] + r['len'])
            r['d_full'] = [log_prob('', R, pre + v + post) if len(R) <= 32 else None for v in r['values']]
            n += 1; miss += any(x is None for x in r['d_span'])
        fo.write(json.dumps({k: d[k] for k in ('sid', 'readings', 'nodes', 'reqs')}, ensure_ascii=False) + '\n')
lat.sort()
print(f'requests={n} span_fallback={miss} time={time.time()-t0:.0f}s span-request p50={lat[len(lat)//2]:.1f}ms p95={lat[int(len(lat)*.95)]:.1f}ms')
print('sentences >32 syllables:', sum(len(d['readings']) > 32 for d in data))
