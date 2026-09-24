"""v3.0 parity fixture: walk2/window3b.py C delta=3 order on the 143 dev units (reports/window3b.md).

Reads (read-only): results/window3/dev.units.jsonl (window3_cli output: node, displayed text, v2 order b2,
window with A scores and pinned texts), walk2/cache/sloth_25m.tsv (the encoder log-probs window3_cli used),
walk2/frozen_window3b.json (C_noharm delta). Writes McBopomofoTests/SlothEFixtures/window3b_dev_c3.jsonl:
one row per unit = the unit + "cache" (its sentence's cache line) + "want" (window3b.order_c(u, 3.0)).
usage: PYTHONDONTWRITEBYTECODE=1 $W/sloth/.venv/bin/python SlothE/make_window3b_fixture.py [--w $W]
"""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--w", default=str(Path(__file__).resolve().parents[2]))
W = Path(ap.parse_args().w)
sys.path.insert(0, str(W / "walk2"))
import window3b  # noqa: E402

delta = json.loads((W / "walk2" / "frozen_window3b.json").read_text())["C_noharm"]["delta"]
assert delta == 3, delta
cache = {}
for line in open(W / "walk2" / "cache" / "sloth_25m.tsv", encoding="utf-8"):
    cache[line.split("\t", 1)[0]] = line.rstrip("\n")
units = window3b.load_units("dev")
out = Path(__file__).resolve().parents[1] / "McBopomofoTests" / "SlothEFixtures" / "window3b_dev_c3.jsonl"
jumps = 0
with open(out, "w", encoding="utf-8") as f:
    for u in units:
        want = window3b.order_c(u, float(delta))
        jumps += want[0] != u["b2"][0]
        f.write(json.dumps({**u, "cache": cache[u["sid"]], "want": want}, ensure_ascii=False) + "\n")
print(f"units={len(units)} jumps={jumps} delta={delta} -> {out}")
