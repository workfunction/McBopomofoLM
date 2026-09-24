// In-walk rescoring for the McBopomofoLM side-by-side test build (phase 2).
//
// Port of scratch/ime-lm-poc/walk2/walk2_walk.h (Walk2Grid::collect / walkEdges,
// VariantGuard::apply) with the frozen config of walk2/frozen.json: an edge is
// a (node, unigram) pair weighted
//     unigram LM score + beta * sum_i log p_sloth(char_i | pos_i) + gamma
// with beta = 0.3, gamma = 0, an illegal char scoring `penalty` (-15) and a
// value that is not one char per syllable scoring penalty * span. Positions
// are visited in index order and span lengths ascending, with strict '>' at
// both the unigram pick and the relaxation, so beta = 0 reproduces the stock
// ReadingGrid::walk(). One addition for the app: a node the user (or the user
// override model) has overridden offers only its current value at the
// engine's own override score, with no neural term, exactly like the stock
// walk, so candidate picks behave as in stock McBopomofo.

#ifndef SOURCE_SLOTHE_SLOTHEWALK_H_
#define SOURCE_SLOTHE_SLOTHEWALK_H_

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "SlothEEngine.h"
#include "reading_grid.h"

namespace McBopomofoSlothE {

struct InWalkParams {
  double beta = 0.3;
  double gamma = 0.0;
  double penalty = -15.0;
  bool variantGuard = true;
};

struct InWalkStats {
  size_t nodes = 0;
  size_t reverted = 0;         // nodes the variant guard restored
  size_t unrepresentable = 0;  // guarded text not expressible as a unigram
  size_t pinsApplied = 0;      // decoder pins found in the grid
};

// A decoder decision (walk2 --repin): node [start, start+len) must take value,
// at the engine's override score, in one re-walk of the same in-walk.
struct WalkPin {
  size_t start = 0;
  size_t len = 0;
  std::string value;
};

// One node of the in-walk walk, with what the decoder stage needs.
struct InWalkNode {
  size_t start = 0;
  size_t len = 0;
  std::string raw;     // chosen unigram
  std::string shown;   // after the variant guard
  bool userPinned = false;
  // the node's unigrams whose value is one char per syllable, deduplicated,
  // in unigram order, with their neural term (walk2_cli --nodes)
  std::vector<std::pair<std::string, double>> aligned;
};

// walk2 dec_combo: decoder-gated override (config A', PLAN.md §9,
// walk2/frozen_final2.json: 25M, beta 0.3, lambda 2, tau 0.5, top-3; v1 was
// 12M lambda 3 tau 0.9): blend b = enc + lambda * dec over the encoder's top-3
// span-aligned candidates, pin the top if p = softmax(b)[top] >= tau, it is
// neither the shown nor the raw value, and not a variant-only swap. Decoder
// scores must all be finite and < 0 (walk2 Dec check); otherwise no decision.
struct DecoderParams {
  double lambda = 2.0;
  double tau = 0.5;
  size_t topK = 3;
};

struct DecoderRequest {
  size_t node = 0;
  size_t start = 0;
  size_t len = 0;
  std::string context;  // shown in-walk text left of the node
  std::vector<std::string> values;
  std::vector<double> enc;
  std::string shown;
  std::string raw;
};

std::vector<DecoderRequest> BuildDecoderRequests(
    const std::vector<InWalkNode>& nodes, size_t topK);
bool DecideDecoderPin(const DecoderRequest& request,
                      const std::vector<double>& decoderScores,
                      const DecoderParams& params, const VariantTable& variants,
                      WalkPin* pin);

// walk2 neural term for `value` placed at positions start..start+len-1.
double InWalkNeuralScore(const ForwardResult& r, const Vocabulary& vocab,
                         size_t start, size_t len, const std::string& value,
                         double penalty);

// Concatenated node values of a walk.
std::string WalkText(
    const Formosa::Gramambular2::ReadingGrid::WalkResult& walk);

class SlothEGrid : public Formosa::Gramambular2::ReadingGrid {
 public:
  using ReadingGrid::ReadingGrid;

  // The rescored, variant-guarded walk as a WalkResult the key handler can use
  // in place of walk(): each node's value() is the chosen unigram. Nodes whose
  // choice differs from the grid node's current value are private copies, so
  // the grid itself is never modified. r must come from this grid's readings;
  // otherwise the stock walk is returned.
  // pins: decoder decisions for a re-walk (walk2 --repin); the variant guard
  // still uses the unpinned stock walk as its baseline. detail: per node of
  // the returned walk (before pins are considered for the detail's own walk).
  WalkResult rescoredWalk(const ForwardResult& r, const Vocabulary& vocab,
                          const VariantTable& variants,
                          const InWalkParams& params, InWalkStats* stats,
                          const std::vector<WalkPin>* pins = nullptr,
                          std::vector<InWalkNode>* detail = nullptr);

  // Deep copy (nodes included) for use on another thread; never shares node
  // state with this grid.
  std::unique_ptr<SlothEGrid> snapshot() const;
  // Snapshots alive right now (all threads); tests check the backlog.
  static int64_t LiveSnapshots();

  ~SlothEGrid();  // ReadingGrid has no virtual destructor: always delete as SlothEGrid

 private:
  bool isSnapshot_ = false;
};

// Candidate-window option (walk2 report 8.1 b'): within the aligned block at
// the front of `order` (as produced by RerankOrder), move the entry equal to
// walkValue, followed by its orthographic variants, to the end of the block.
// Everything else keeps its position order.
std::vector<size_t> DemoteWalkValue(const std::vector<size_t>& order,
                                    const std::vector<RerankItem>& items,
                                    const std::string& walkValue,
                                    const VariantTable* variants);

}  // namespace McBopomofoSlothE

#endif  // SOURCE_SLOTHE_SLOTHEWALK_H_
