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
#include <limits>
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
  // v3.2a (PLAN.md §18 D7): a pin kept from an earlier pause pass. It is
  // applied only while its node is still on the un-pinned in-walk path of the
  // current buffer (a node that a later syllable turned into a longer word is
  // not forced back).
  bool sticky = false;
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

// walk2 dec_combo: decoder-gated override. Blend b = enc + lambda * dec over the
// encoder's top-3 span-aligned candidates, pin the top if p = softmax(b)[top] >=
// tau, it is neither the shown nor the raw value, and not a variant-only swap.
// Decoder scores must all be finite and < 0 (walk2 Dec check); otherwise no
// decision.
//
// v3.4 (SlothE/eval-2026-10/): the decoder scores the whole buffer around the
// candidate instead of the candidate span alone (fullBuffer), with lambda 1:
// each request is scored as
//   context   = carried text + the in-walk text left of the node, with the pins
//               already decided in this pass (requests go left to right)
//   scored    = candidate + the in-walk text right of the node (at most
//               suffixCapChars chars)
// RunPipeline does the sequencing; DecideDecoderPin only sees the scores.
// fullBuffer = false is the v3.0-v3.3 per-key gate (config A', walk2/
// frozen_final2.json: lambda 2, tau 0.5, top-3; candidate scored with the
// pre-pin text on its left only), which the v2 parity tests pin.
struct DecoderParams {
  double lambda = 1.0;
  double tau = 0.5;
  size_t topK = 3;
  bool fullBuffer = true;
  // chars of the buffer text right of the node that join the scored string
  // (the nearest ones); 0 = the candidate alone
  size_t suffixCapChars = 24;

  // The v3.0-v3.3 gate: config A' (lambda 2, tau 0.5, top-3), span-only.
  // Also the v3.2a pause gate's rule, which always scores the whole buffer.
  static DecoderParams V2() {
    DecoderParams p;
    p.lambda = 2.0;
    p.fullBuffer = false;
    return p;
  }
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

// v3.0 candidate window "C" (PLAN.md §12 W2, reports/window3b.md): the
// jump-ahead threshold delta, in the in-walk's total path score (nats).
constexpr double kJumpAheadDelta = 3.0;
// v3.0.1: wall-time budget for the pinned walks at window open (r14 review);
// over it, C is skipped and the window keeps the v2 order.
constexpr double kJumpAheadBudgetMs = 16.0;

// One candidate-window entry with the walk a user pick of it gives
// (walk2/window3_cli "A"): its node span, and the in-walk re-run on a grid copy
// with overrideCandidate(loc, entry), no decoder pins. score = sum over that
// walk's picks of LM + beta * neural, the picked node at its real LM score
// (not the override score); -inf and "" when the pick fails.
struct PinnedEntry {
  size_t start = 0;
  size_t len = 0;
  std::string value;
  double score = 0;
  std::string text;
};

// Displayed char per reading position (walk2 cv.chars_of): "" where a node's
// value is not one char per syllable.
std::vector<std::string> DisplayedChars(
    const Formosa::Gramambular2::ReadingGrid::WalkResult& walk, size_t n);

// walk2/window3b.py order_c: starting from `order` (the v2 window order), move
// X to rank 1 when A(X) - A(S*) >= delta, or when there is no S*.
//   S* = best-A entry on the node's span [nodeStart, nodeStart+nodeLen) other
//        than nodeValue and its variant-only swaps;
//   X  = best-A entry on any other span whose value differs from the displayed
//        text over its own span and whose pinned text differs from the
//        displayed sentence (when every position has a displayed char).
// Ties go to the smaller window index. Everything else keeps `order`.
std::vector<size_t> JumpAheadOrder(const std::vector<size_t>& order,
                                   const std::vector<PinnedEntry>& entries,
                                   size_t nodeStart, size_t nodeLen,
                                   const std::string& nodeValue,
                                   const std::vector<std::string>& shownChars,
                                   const VariantTable* variants,
                                   double delta = kJumpAheadDelta);

class SlothEGrid : public Formosa::Gramambular2::ReadingGrid {
 public:
  using ReadingGrid::ReadingGrid;

  // LM score of `value` at node [start, start+len): its first unigram with
  // that value (walk2 lmOf); -inf when there is none.
  double unigramScore(size_t start, size_t len, const std::string& value) const;

  // The v2 window items for `candidates` against the walk node at
  // [spanStart, spanStart+spanLength) with reading nodeReading: aligned = same
  // reading and one char per syllable. Aligned score = ScoreCandidate, except
  // in a 1-reading grid, where it is the McBopomofo unigram score (v3.0.2,
  // reports/w5.md key h).
  std::vector<RerankItem> windowItems(const std::vector<Candidate>& candidates,
                                      const std::string& nodeReading,
                                      size_t spanStart, size_t spanLength,
                                      const ForwardResult& r,
                                      const Vocabulary& vocab) const;

  // PinnedEntry for each of `candidates` (= candidatesAt(loc)); empty if they
  // are not this grid's window at loc, or if the walks would take longer than
  // budgetMs (checked after each entry: time so far, and time so far projected
  // over all entries).
  std::vector<PinnedEntry> pinnedEntries(
      size_t loc, const std::vector<Candidate>& candidates,
      const ForwardResult& r, const Vocabulary& vocab,
      const VariantTable& variants, const InWalkParams& params,
      double budgetMs = std::numeric_limits<double>::infinity()) const;

  // The rescored, variant-guarded walk as a WalkResult the key handler can use
  // in place of walk(): each node's value() is the chosen unigram. Nodes whose
  // choice differs from the grid node's current value are private copies, so
  // the grid itself is never modified. r must come from this grid's readings;
  // otherwise the stock walk is returned.
  // A 1-reading grid gets the stock walk (v3.0.2b), with pins ignored and an
  // empty detail, unless r carries context (r.contextLength > 0, v3.1a Ce1).
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

  // v3.2b (legal-but-wrong correction entry, reports/typo.md §8): a copy of this
  // grid in which the reading at `pos` is another one, with every user override
  // cleared (the study scored sentences without picks). Only the spans that
  // cover `pos` change; their nodes need language-model lookups, which happen
  // on the calling (main) thread in patchFor() and are handed over as a
  // SpanPatch list, so the copy itself is made on any thread without the model.
  struct SpanPatch {
    size_t start = 0;
    size_t len = 0;
    std::string reading;  // the joined key looked up
    std::vector<Formosa::Gramambular2::LanguageModel::Unigram> unigrams;  // score-ranked, as the grid's own nodes
  };
  // Main thread (looks the language model up). Empty when pos is out of range.
  std::vector<SpanPatch> patchFor(size_t pos, const std::string& reading);
  std::unique_ptr<SlothEGrid> snapshotWithReading(
      size_t pos, const std::string& reading,
      const std::vector<SpanPatch>& patch) const;
  // Snapshots alive right now (all threads); tests check the backlog.
  static int64_t LiveSnapshots();

  ~SlothEGrid();  // ReadingGrid has no virtual destructor: always delete as SlothEGrid

 private:
  bool isSnapshot_ = false;
};

// v3.2b: the in-walk's total path score of `grid`'s walk over the encoder result
// r: the sum over the walked nodes of LM + beta * neural (typo/t2_e.cpp `total`).
// False when the walk has no model detail (a 1-reading buffer without carried
// context shows the stock walk).
bool InWalkTotal(SlothEGrid* grid, const ForwardResult& r, const Vocabulary& vocab,
                 const VariantTable& variants, const InWalkParams& params,
                 double* total);

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
