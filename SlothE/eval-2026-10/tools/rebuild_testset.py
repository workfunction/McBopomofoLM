"""Rebuild the evaluated test units from the public index (text is not redistributed).
For each unit the page at `url` is re-fetched, reduced to its Han characters, and the window of length `n`
whose SHA-1 equals `sha1` becomes the unit text. Units whose page changed or vanished are reported as missing.
usage: python rebuild_testset.py ../testset/news_index.jsonl news_lab.jsonl [--limit N]"""
import hashlib, json, re, sys, time, urllib.request
from collections import defaultdict
src, dst = sys.argv[1], sys.argv[2]
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
HAN = re.compile(r"[一-鿿]")
rows = [json.loads(l) for l in open(src, encoding="utf-8")]
if limit: rows = rows[:limit]
by_url = defaultdict(list)
for r in rows: by_url[r["url"]].append(r)
found = missing = 0; last = 0.0
with open(dst, "w", encoding="utf-8") as out:
    for url, units in by_url.items():
        time.sleep(max(0.0, 1.2 - (time.time() - last))); last = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Cookie": "over18=1"})
            html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            html = ""; print("fetch failed", url, e, file=sys.stderr)
        html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
        han = "".join(HAN.findall(re.sub(r"<[^>]+>", " ", html)))
        for r in units:
            n, want, hit = r["n"], r["sha1"], None
            for i in range(0, len(han) - n + 1):
                if hashlib.sha1(han[i:i + n].encode()).hexdigest() == want: hit = han[i:i + n]; break
            if hit is None: missing += 1; continue
            found += 1
            r = dict(r, han=hit, readings=r["readings"].split(" "))
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps({"units": len(rows), "found": found, "missing": missing}))
