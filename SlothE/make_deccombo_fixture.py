"""Reference for the decoder-combo parity test (McBopomofoTests/SlothEDecoderTests.mm): walk2's
in-walk + decoder-gated override walks (12M, beta 0.3, top-3), each row with its own (lambda, tau):
  * final config A (PLAN.md §4b v2, walk2/frozen_final.json: lambda 3, tau 0.9) -- the app's shipped
    constants -- on dev (500): results/walk2/dev_final_A.walks.jsonl (walk2 final_run.py dev, fixed v2
    decoder cache: <= 3 candidates per call, every score finite and < 0); copy in SlothE/walk2-ref/
  * v1 (lambda 1.5, tau 0.9), kept as a second decision-logic regression:
      test_fork (217): results/walk2@test_fork/walks_deccombo-12m.jsonl
      dev (500): SlothE/make_dev_deccombo_ref.py --scores live (walk2's own per-node live decoder calls)
Readings: dev from SlothEFixtures/inwalk_parity.jsonl, test_fork from walk2/cache/sloth_12m@test_fork.tsv.
Output: McBopomofoTests/SlothEFixtures/deccombo_parity.jsonl  {"set","sid","lambda","tau","readings","nodes","text"}
usage: python3 make_deccombo_fixture.py --final-a D --dev-ref D --fork-ref F --fork-cache C
"""
import argparse
import json
from pathlib import Path

FIX = Path(__file__).resolve().parent.parent / "McBopomofoTests" / "SlothEFixtures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-a", required=True)
    ap.add_argument("--dev-ref", required=True)
    ap.add_argument("--fork-ref", required=True)
    ap.add_argument("--fork-cache", required=True)
    a = ap.parse_args()
    dev_read = {d["sid"]: d["readings"] for d in map(json.loads, open(FIX / "inwalk_parity.jsonl", encoding="utf-8"))}
    fork_read = {}
    for line in open(a.fork_cache, encoding="utf-8"):
        f = line.split("\t", 3)
        fork_read[f[0]] = f[2].split()
    n = {}
    with open(FIX / "deccombo_parity.jsonl", "w", encoding="utf-8") as out:
        def emit(set_name, sid, lam, tau, readings, nodes):
            out.write(json.dumps({"set": set_name, "sid": sid, "lambda": lam, "tau": tau, "readings": readings,
                                  "nodes": nodes, "text": "".join(v for _, _, v in nodes)}, ensure_ascii=False) + "\n")
            n[set_name] = n.get(set_name, 0) + 1
        for d in map(json.loads, open(a.final_a, encoding="utf-8")):
            assert d["config"] == "A", d
            emit("dev_final_A", d["sent_id"], 3.0, 0.9, dev_read[d["sent_id"]], d["walk_nodes"])
        for set_name, ref, readings in (("test_fork_v1", a.fork_ref, fork_read), ("dev_v1", a.dev_ref, dev_read)):
            for d in map(json.loads, open(ref, encoding="utf-8")):
                assert d["cfg"] == ["12m", 0.3, 1.5, 0.9], d["cfg"]
                emit(set_name, d["sent_id"], 1.5, 0.9, readings[d["sent_id"]], d["walk_nodes"])
    print(n)


if __name__ == "__main__":
    main()
