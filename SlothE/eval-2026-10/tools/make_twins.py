"""data.txt variant: add 再-twins for '<prefix>在' phrases where 再 is grammatical and the twin is missing.
Twin score = 在-phrase score - 0.1 (log10), so the encoder decides when it has a preference."""
import sys
src, dst, listing = sys.argv[1:4]
PREFIXES = """會 就 要 將 也 都 先 能 才 別 請 你 他 她 我們 你們 他們 咱們 大家 可以 應該 必須 需要 想 得 還會 不會 也會 都會 才會 就會 只會 將會
不要 不能 不想 不用 不必 不再 無法 一定要 打算 準備 希望 等 等下 等一下 之後 然後 以後 明天 下次 改天 稍後 晚點 回頭 重新 沒有 不得不 只好 只能 還要 還得 就要
又 也要 都要 還能 也能 不如 最好 何不 何必 願意 可能 或許 也許 能夠 要不要 會不會 能不能 可不可以""".split()
lines = open(src, encoding="utf-8").read().split("\n")
header, body = lines[0], [l for l in lines[1:] if l]
lm = {}; order = []
for l in body:
    r, v, s = l.split(" ")
    if r not in lm: lm[r] = []; order.append(r)
    lm[r].append((v, s))
have = {(r, v) for r in lm for v, _ in lm[r]}
added = []
for r in list(lm):
    for v, s in list(lm[r]):
        if len(v) >= 2 and v.endswith("在") and r.endswith("ㄗㄞˋ") and v[:-1] in PREFIXES:
            t = v[:-1] + "再"
            if (r, t) not in have:
                lm[r].append((t, f"{float(s) - 0.1:.8f}")); have.add((r, t)); added.append(f"{v}->{t} {s}")
with open(dst, "w", encoding="utf-8") as f:
    f.write(header + "\n")
    for r in sorted(lm, key=lambda x: x.encode()):
        for v, s in lm[r]: f.write(f"{r} {v} {s}\n")
open(listing, "w", encoding="utf-8").write("\n".join(added) + "\n")
print(len(added), "twins added")
