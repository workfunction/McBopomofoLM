// Offline McBopomofoLM walk: stock walk, in-walk walk (+ decoder requests), and the pinned walk.
// stdin, one sentence per line:  id \t readings(space) \t logits.f32 path \t pins ("s,l,v;..." or "-")
// stdout: one JSON per line.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include "McBopomofoLM.h"
#include "SlothEWalk.h"
using namespace McBopomofoSlothE;

static std::vector<std::string> Split(const std::string& s, char d) {
  std::vector<std::string> o; std::string c; std::istringstream is(s);
  while (std::getline(is, c, d)) o.push_back(c);
  return o;
}
static std::string Q(const std::string& s) {
  std::string o = "\"";
  for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
  return o + "\"";
}
static std::string Nodes(const Formosa::Gramambular2::ReadingGrid::WalkResult& w) {
  std::string o = "["; size_t p = 0; bool first = true;
  for (const auto& n : w.nodes) {
    size_t l = n->spanningLength();
    o += (first ? "" : ",") + std::string("[") + std::to_string(p) + "," + std::to_string(p + l) + "," + Q(n->value()) + "]";
    p += l; first = false;
  }
  return o + "]";
}
int main(int argc, char** argv) {
  if (argc < 3) { std::cerr << "usage: walk_cli data.txt bundleDir [beta]\n"; return 2; }
  InWalkParams params;
  if (argc > 3) params.beta = std::stod(argv[3]);
  McBopomofo::McBopomofoLM lmObj;
  lmObj.loadLanguageModel(argv[1]);
  std::shared_ptr<Formosa::Gramambular2::LanguageModel> lm(std::shared_ptr<void>(), &lmObj);
  Vocabulary vocab; VariantTable variants; std::string err;
  if (!vocab.load(argv[2], &err)) { std::cerr << "vocab: " << err << "\n"; return 1; }
  if (!variants.load(std::string(argv[2]) + "/variants.tsv", &err)) { std::cerr << "variants: " << err << "\n"; return 1; }
  std::string line;
  while (std::getline(std::cin, line)) {
    auto f = Split(line, '\t');
    if (f.size() < 4) continue;
    auto readings = Split(f[1], ' ');
    SlothEGrid grid(lm);
    grid.setReadingSeparator("-");
    for (const auto& r : readings) grid.insertReading(r);
    auto stock = grid.walk();
    auto r = std::make_shared<ForwardResult>();
    r->charCount = vocab.charCount();
    const size_t T = readings.size(), C = static_cast<size_t>(r->charCount);
    r->logits.resize(T * C);
    std::ifstream in(f[2], std::ios::binary);
    in.read(reinterpret_cast<char*>(r->logits.data()), static_cast<std::streamsize>(T * C * sizeof(float)));
    for (const auto& rd : readings) r->ids.push_back(vocab.mapSyllable(rd).id);
    ComputeLogZ(r.get(), vocab);
    std::vector<WalkPin> pins;
    if (f[3] != "-") for (const auto& p : Split(f[3], ';')) {
      auto q = Split(p, ',');
      pins.push_back({static_cast<size_t>(std::stoul(q[0])), static_cast<size_t>(std::stoul(q[1])), q[2]});
    }
    std::vector<InWalkNode> detail; InWalkStats stats;
    auto walk = grid.rescoredWalk(*r, vocab, variants, params, &stats, nullptr, &detail);
    auto reqs = BuildDecoderRequests(detail, 3);
    std::string rj = "[";
    for (size_t i = 0; i < reqs.size(); ++i) {
      const auto& q = reqs[i];
      rj += (i ? "," : "") + std::string("{\"start\":") + std::to_string(q.start) + ",\"len\":" + std::to_string(q.len) +
            ",\"ctx\":" + Q(q.context) + ",\"shown\":" + Q(q.shown) + ",\"raw\":" + Q(q.raw) + ",\"values\":[";
      for (size_t k = 0; k < q.values.size(); ++k) rj += (k ? "," : "") + Q(q.values[k]);
      rj += "],\"enc\":[";
      for (size_t k = 0; k < q.enc.size(); ++k) rj += (k ? "," : "") + std::to_string(q.enc[k]);
      rj += "]}";
    }
    rj += "]";
    std::string pinned = "null";
    if (!pins.empty()) {
      InWalkStats ps;
      pinned = Nodes(grid.rescoredWalk(*r, vocab, variants, params, &ps, &pins));
    }
    std::cout << "{\"id\":" << Q(f[0]) << ",\"stock\":" << Nodes(stock) << ",\"inwalk\":" << Nodes(walk)
              << ",\"requests\":" << rj << ",\"pinned\":" << pinned << "}\n";
  }
}
