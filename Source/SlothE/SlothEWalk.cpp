// In-walk rescoring for the McBopomofoLM side-by-side test build (phase 2).
// See SlothEWalk.h; line references are to scratch/ime-lm-poc/walk2/walk2_walk.h.

#include "SlothEWalk.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>

namespace McBopomofoSlothE {

namespace {

using Formosa::Gramambular2::ReadingGrid;

// A copy of a grid node with a chosen unigram made current, without marking
// it overridden. Only ever placed in a WalkResult, never in the grid.
class SelectedNode : public ReadingGrid::Node {
 public:
  SelectedNode(const ReadingGrid::Node& node, size_t unigramIndex)
      : ReadingGrid::Node(node) {
    unigramIter_ =
        unigrams_.begin() + static_cast<std::ptrdiff_t>(unigramIndex);
  }
};

struct Edge {
  size_t unigram = 0;
  double lm = 0;
  double neural = 0;
};

struct NodeEdges {
  size_t start = 0;
  size_t len = 0;
  ReadingGrid::NodePtr node;
  std::vector<Edge> edges;
  bool pinned = false;
};

struct Pick {
  size_t nodeIndex = 0;
  size_t unigram = 0;
};

// walk2_walk.h walkEdges() (:174): Viterbi over (node, unigram) edges.
// pins (indexed like all, -1 = none): a pinned node offers only that unigram
// at the engine override score (walk2 --repin).
std::vector<Pick> WalkEdges(const std::vector<NodeEdges>& all, size_t n,
                            double beta, double gamma,
                            const std::vector<long>* pins) {
  struct State {
    size_t fromIndex = 0;
    bool reached = false;
    Pick pick;
    double maxScore = -std::numeric_limits<double>::infinity();
  };
  std::vector<Pick> path;
  if (n == 0) {
    return path;
  }
  std::vector<State> v(n + 1);
  v[0].maxScore = 0.0;
  for (size_t ni = 0; ni < all.size(); ++ni) {
    const NodeEdges& ne = all[ni];
    double best = -std::numeric_limits<double>::infinity();
    size_t bestUnigram = 0;
    if (pins != nullptr && (*pins)[ni] >= 0) {
      best = ReadingGrid::Node::kOverridingScore;
      bestUnigram = static_cast<size_t>((*pins)[ni]);
    } else {
      for (const Edge& e : ne.edges) {
        double w = ne.pinned ? e.lm : e.lm + beta * e.neural + gamma;
        if (w > best) {
          best = w;
          bestUnigram = e.unigram;
        }
      }
    }
    double score = v[ne.start].maxScore + best;
    State& t = v[ne.start + ne.len];
    if (score > t.maxScore) {
      t.maxScore = score;
      t.fromIndex = ne.start;
      t.reached = true;
      t.pick = Pick{ni, bestUnigram};
    }
  }
  for (size_t cur = n; cur > 0; cur = v[cur].fromIndex) {
    if (!v[cur].reached) {
      return {};
    }
    path.push_back(v[cur].pick);
  }
  std::reverse(path.begin(), path.end());
  return path;
}

std::string PickValue(const std::vector<NodeEdges>& all, const Pick& p) {
  return all[p.nodeIndex].node->unigrams()[p.unigram].value();
}

}  // namespace

double InWalkNeuralScore(const ForwardResult& r, const Vocabulary& vocab,
                         size_t start, size_t len, const std::string& value,
                         double penalty) {
  if (Utf8Length(value) != len) {
    return penalty * static_cast<double>(len);
  }
  double s = 0;
  size_t i = 0;
  for (size_t k = 0; k < len; ++k) {
    uint32_t cp = NextCodePoint(value, &i);
    if (!r.known(start + k)) {
      continue;  // provisional view: no score yet for this position
    }
    int32_t cid = vocab.charIdForCodePoint(cp);
    double lp = cid >= 0 ? LogProb(r, vocab, start + k, cid)
                         : -std::numeric_limits<double>::infinity();
    s += std::isfinite(lp) ? lp : penalty;
  }
  return s;
}

std::string WalkText(const ReadingGrid::WalkResult& walk) {
  std::string text;
  for (const auto& node : walk.nodes) {
    text += node->value();
  }
  return text;
}

ReadingGrid::WalkResult SlothEGrid::rescoredWalk(
    const ForwardResult& r, const Vocabulary& vocab,
    const VariantTable& variants, const InWalkParams& params,
    InWalkStats* stats, const std::vector<WalkPin>* pins,
    std::vector<InWalkNode>* detail) {
  InWalkStats local;
  InWalkStats& st = stats != nullptr ? *stats : local;
  st = InWalkStats();
  const size_t n = readings_.size();
  if (detail != nullptr) {
    detail->clear();
  }
  if (n == 0 || r.length() != n) {
    return walk();
  }

  // walk2_walk.h collect() (:155): nodes in the stock walk's visiting order.
  std::vector<NodeEdges> all;
  for (size_t i = 0; i < spans_.size(); ++i) {
    const Span& span = spans_[i];
    for (size_t len = 1; len <= span.maxLength(); ++len) {
      const NodePtr& node = span.nodeOf(len);
      if (node == nullptr || node->unigrams().empty()) {
        continue;
      }
      NodeEdges ne;
      ne.start = i;
      ne.len = len;
      ne.node = node;
      const auto& unigrams = node->unigrams();
      if (node->isOverridden()) {
        // Stock semantics for user / user-override-model picks.
        ne.pinned = true;
        size_t current = 0;
        for (size_t k = 0; k < unigrams.size(); ++k) {
          if (unigrams[k].value() == node->value()) {
            current = k;
            break;
          }
        }
        ne.edges.push_back(Edge{current, node->score(), 0});
      } else {
        for (size_t k = 0; k < unigrams.size(); ++k) {
          ne.edges.push_back(Edge{k, unigrams[k].score(),
                                  InWalkNeuralScore(r, vocab, i, len,
                                                    unigrams[k].value(),
                                                    params.penalty)});
        }
      }
      all.push_back(std::move(ne));
    }
  }

  std::vector<long> pinIndex;
  if (pins != nullptr && !pins->empty()) {
    pinIndex.assign(all.size(), -1);
    for (const WalkPin& pin : *pins) {
      for (size_t ni = 0; ni < all.size(); ++ni) {
        const NodeEdges& ne = all[ni];
        if (ne.pinned || ne.start != pin.start || ne.len != pin.len) {
          continue;
        }
        const auto& us = ne.node->unigrams();
        for (size_t u = 0; u < us.size(); ++u) {
          if (us[u].value() == pin.value) {
            pinIndex[ni] = static_cast<long>(u);
            ++st.pinsApplied;
            break;
          }
        }
        break;
      }
    }
  }
  std::vector<Pick> picks = WalkEdges(all, n, params.beta, params.gamma,
                                      pinIndex.empty() ? nullptr : &pinIndex);
  std::vector<Pick> base = WalkEdges(all, n, 0.0, 0.0, nullptr);
  if (picks.empty() || base.empty()) {
    if (detail != nullptr) {
      detail->clear();
    }
    return walk();
  }

  // Baseline chars per position (walk2 charsOf(): "" where a node is not one
  // char per syllable), and where each baseline node starts / ends.
  std::vector<std::string> baseChars(n);
  std::vector<size_t> baseNodeAt(n, 0);
  for (size_t b = 0; b < base.size(); ++b) {
    const NodeEdges& ne = all[base[b].nodeIndex];
    std::vector<std::string> cs = Utf8Chars(PickValue(all, base[b]));
    for (size_t k = 0; k < ne.len; ++k) {
      baseNodeAt[ne.start + k] = b;
      if (cs.size() == ne.len) {
        baseChars[ne.start + k] = cs[k];
      }
    }
  }

  WalkResult out;
  for (const Pick& p : picks) {
    const NodeEdges& ne = all[p.nodeIndex];
    std::string value = PickValue(all, p);
    size_t unigram = p.unigram;
    bool useBaseNodes = false;

    // walk2_walk.h VariantGuard::apply() (:263)
    std::vector<std::string> cs = Utf8Chars(value);
    if (params.variantGuard && !ne.pinned && cs.size() == ne.len) {
      bool differs = false;
      bool onlyVariants = true;
      std::string baseValue;
      for (size_t k = 0; k < cs.size(); ++k) {
        const std::string& b = baseChars[ne.start + k];
        baseValue += b;
        if (cs[k] == b) {
          continue;
        }
        differs = true;
        if (b.empty() || !variants.isVariantPair(cs[k], b)) {
          onlyVariants = false;
        }
      }
      if (differs && onlyVariants) {
        ++st.reverted;
        const auto& unigrams = ne.node->unigrams();
        auto it = std::find_if(unigrams.begin(), unigrams.end(),
                               [&baseValue](const auto& u) {
                                 return u.value() == baseValue;
                               });
        if (it != unigrams.end()) {
          unigram = static_cast<size_t>(it - unigrams.begin());
        } else {
          // Use the baseline nodes when they tile exactly this span.
          size_t first = baseNodeAt[ne.start];
          size_t last = baseNodeAt[ne.start + ne.len - 1];
          const NodeEdges& fb = all[base[first].nodeIndex];
          const NodeEdges& lb = all[base[last].nodeIndex];
          if (fb.start == ne.start && lb.start + lb.len == ne.start + ne.len) {
            useBaseNodes = true;
            for (size_t b = first; b <= last; ++b) {
              const NodeEdges& bn = all[base[b].nodeIndex];
              const auto& bu = bn.node->unigrams()[base[b].unigram];
              out.nodes.push_back(
                  bn.pinned || bu.value() == bn.node->value()
                      ? bn.node
                      : std::make_shared<SelectedNode>(*bn.node,
                                                       base[b].unigram));
            }
          } else {
            ++st.unrepresentable;
          }
        }
      }
    }
    if (detail != nullptr) {
      InWalkNode dn;
      dn.start = ne.start;
      dn.len = ne.len;
      dn.raw = value;
      dn.shown = useBaseNodes ? std::string() : ne.node->unigrams()[unigram].value();
      if (useBaseNodes) {
        for (size_t k = 0; k < ne.len; ++k) {
          dn.shown += baseChars[ne.start + k];
        }
      }
      dn.userPinned = ne.pinned;
      std::vector<std::string> seen;
      for (const Edge& e : ne.edges) {
        const std::string& v = ne.node->unigrams()[e.unigram].value();
        if (Utf8Chars(v).size() != ne.len ||
            std::find(seen.begin(), seen.end(), v) != seen.end()) {
          continue;
        }
        seen.push_back(v);
        dn.aligned.emplace_back(v, e.neural);
      }
      detail->push_back(std::move(dn));
    }
    if (useBaseNodes) {
      continue;
    }
    const auto& chosen = ne.node->unigrams()[unigram];
    if (ne.pinned || chosen.value() == ne.node->value()) {
      out.nodes.push_back(ne.node);
    } else {
      out.nodes.push_back(std::make_shared<SelectedNode>(*ne.node, unigram));
    }
  }
  out.totalReadings = n;
  out.vertices = n;
  out.edges = all.size();
  st.nodes = out.nodes.size();
  return out;
}

namespace {
// The snapshot only walks; it never looks readings up.
class NoLookupLanguageModel : public Formosa::Gramambular2::LanguageModel {
 public:
  std::vector<Unigram> getUnigrams(const std::string&) override { return {}; }
  bool hasUnigrams(const std::string&) override { return false; }
};
}  // namespace

namespace {
std::atomic<int64_t> gLiveSnapshots{0};
}  // namespace

int64_t SlothEGrid::LiveSnapshots() { return gLiveSnapshots.load(); }

SlothEGrid::~SlothEGrid() {
  if (isSnapshot_) {
    gLiveSnapshots.fetch_sub(1);
  }
}

std::unique_ptr<SlothEGrid> SlothEGrid::snapshot() const {
  auto copy = std::make_unique<SlothEGrid>(std::make_shared<NoLookupLanguageModel>());
  copy->isSnapshot_ = true;
  gLiveSnapshots.fetch_add(1);
  copy->separator_ = separator_;
  copy->cursor_ = cursor_;
  copy->readings_ = readings_;
  copy->spans_.resize(spans_.size());
  for (size_t i = 0; i < spans_.size(); ++i) {
    for (size_t len = 1; len <= spans_[i].maxLength(); ++len) {
      const NodePtr& node = spans_[i].nodeOf(len);
      if (node != nullptr) {
        copy->spans_[i].add(std::make_shared<Node>(*node));
      }
    }
  }
  return copy;
}

std::vector<DecoderRequest> BuildDecoderRequests(
    const std::vector<InWalkNode>& nodes, size_t topK) {
  std::vector<DecoderRequest> out;
  std::string context;
  for (size_t k = 0; k < nodes.size(); ++k) {
    const InWalkNode& n = nodes[k];
    if (!n.userPinned && n.aligned.size() >= 2) {
      std::vector<size_t> order(n.aligned.size());
      for (size_t j = 0; j < order.size(); ++j) {
        order[j] = j;
      }
      std::stable_sort(order.begin(), order.end(), [&n](size_t a, size_t b) {
        return n.aligned[a].second > n.aligned[b].second;
      });
      DecoderRequest req;
      req.node = k;
      req.start = n.start;
      req.len = n.len;
      req.context = context;
      req.shown = n.shown;
      req.raw = n.raw;
      for (size_t j = 0; j < order.size() && j < topK; ++j) {
        req.values.push_back(n.aligned[order[j]].first);
        req.enc.push_back(n.aligned[order[j]].second);
      }
      out.push_back(std::move(req));
    }
    context += n.shown;
  }
  return out;
}

bool DecideDecoderPin(const DecoderRequest& request,
                      const std::vector<double>& decoderScores,
                      const DecoderParams& params, const VariantTable& variants,
                      WalkPin* pin) {
  size_t n = request.values.size();
  if (n < 2 || decoderScores.size() != n) {
    return false;
  }
  for (double d : decoderScores) {
    if (!std::isfinite(d) || d >= 0.0) {  // walk2 Dec: finite and strictly negative
      return false;
    }
  }
  std::vector<double> b(n);
  size_t best = 0;
  for (size_t k = 0; k < n; ++k) {
    b[k] = request.enc[k] + params.lambda * decoderScores[k];
    if (b[k] > b[best]) {
      best = k;
    }
  }
  double z = 0;
  for (size_t k = 0; k < n; ++k) {
    z += std::exp(b[k] - b[best]);
  }
  double p = 1.0 / z;
  const std::string& v = request.values[best];
  if (v == request.shown || v == request.raw || p < params.tau ||
      variants.isVariantPair(v, request.shown)) {
    return false;
  }
  pin->start = request.start;
  pin->len = request.len;
  pin->value = v;
  return true;
}

std::vector<size_t> DemoteWalkValue(const std::vector<size_t>& order,
                                    const std::vector<RerankItem>& items,
                                    const std::string& walkValue,
                                    const VariantTable* variants) {
  size_t blockEnd = 0;
  while (blockEnd < order.size() && items[order[blockEnd]].aligned) {
    ++blockEnd;
  }
  std::vector<size_t> kept;
  std::vector<size_t> shown;
  std::vector<size_t> shownVariants;
  for (size_t k = 0; k < blockEnd; ++k) {
    const std::string& v = items[order[k]].value;
    if (v == walkValue) {
      shown.push_back(order[k]);
    } else if (variants != nullptr && variants->isVariantPair(v, walkValue)) {
      shownVariants.push_back(order[k]);
    } else {
      kept.push_back(order[k]);
    }
  }
  if (shown.empty()) {
    return order;
  }
  std::vector<size_t> out = kept;
  out.insert(out.end(), shown.begin(), shown.end());
  out.insert(out.end(), shownVariants.begin(), shownVariants.end());
  out.insert(out.end(), order.begin() + static_cast<std::ptrdiff_t>(blockEnd),
             order.end());
  return out;
}

}  // namespace McBopomofoSlothE
