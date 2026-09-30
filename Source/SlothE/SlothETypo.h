// v3.1b illegal-syllable correction window (reports/typo.md §2-3, §8.4-8.5,
// §9; frozen rules typo/frozen.json, typo/t2_ill_frozen.json,
// typo/t2_fin_frozen.json).
//
// Candidates: T1 "slot1_nb" = every legal syllable one C/M/V slot edit away
// (tone counted as a slot) or one neighbor-key substitution away from the
// canonical Standard (Dachen) key sequence of the typed illegal syllable
// (typo/typo_cli.cpp). Rules, over those candidates:
//   LM+prior (i')  lm - [2 edits] + [tone changed] + 0.5 nb + 0.5 dslots
//   T1 (iii)       E + 2 lm - [2 edits] + [tone changed] + nb + dslots
//                  (E = encoder, left context + the candidate: max full-vocab
//                  log-softmax over the candidate's chars)
//   D+E            per (syllable, char): D(char) + T1(syllable) + add-final
//                  bonus 3 (D = log P_dec(next text starts with char | left
//                  text), w = 1, direction prior w_diag = 1: no term)
// Entries are chars (deduplicated, first occurrence), each with its syllable.

#ifndef SOURCE_SLOTHE_SLOTHETYPO_H_
#define SOURCE_SLOTHE_SLOTHETYPO_H_

#include <atomic>
#include <functional>
#include <string>
#include <vector>

#include "SlothEDecoder.h"
#include "SlothEEngine.h"
#include "SlothEWalk.h"
#include "language_model.h"

namespace McBopomofoSlothE {

constexpr size_t kTypoMaxEntries = 27;       // 3 pages of 9
constexpr double kTypoAddFinalBonusT1 = 1;   // t2_fin_frozen "T1"
constexpr double kTypoAddFinalBonusWindow = 3;  // t2_fin_frozen "ill" (D+E, and its T1 fallback)
constexpr double kTypoNoEncoder = -1e4;      // a syllable with no scoreable char (typo/score.py)

struct TypoCandidate {
  std::string reading;
  int cmv = 0;               // C/M/V slots that differ from the typed syllable
  bool toneChanged = false;
  bool neighbor = false;     // produced by a neighbor-key substitution
  bool addFinal = false;     // typed + one final (vowel) slot, same rest (t2_fin)
  double lm = 0;             // best unigram log-prob of the reading
  int dslots = 0;            // non-tone chars of the reading - those of the typed syllable
  std::vector<std::string> chars;  // unigram values, language-model order
  double encoder = kTypoNoEncoder;  // E, once computed
};

struct TypoEntry {
  std::string reading;
  std::string value;
};

// slot1_nb candidates of an illegal composed syllable, in reading order.
std::vector<TypoCandidate> TypoCandidates(const std::string& typed,
                                          Formosa::Gramambular2::LanguageModel& lm);

double TypoLMPriorScore(const TypoCandidate& c);
double TypoT1Score(const TypoCandidate& c);  // uses c.encoder

// E of a candidate from an encoder pass whose position `pos` is the candidate.
double TypoEncoderScore(const ForwardResult& r, size_t pos, const Vocabulary& vocab,
                        const std::vector<std::string>& chars);

// Syllables by score (ties: reading), then each syllable's chars in order.
std::vector<TypoEntry> TypoSyllableOrder(const std::vector<TypoCandidate>& cands,
                                         const std::vector<double>& scores);
// D+E: every (syllable, char) by D(char) + T1(syllable) + bonus(syllable).
std::vector<TypoEntry> TypoDecoderOrder(const std::vector<TypoCandidate>& cands,
                                        const std::function<double(const std::string&)>& D);

// The window order. enc = E filled in; D = nullptr when the decoder gave
// nothing; leftTextEmpty = no carried or buffer text left of the reading.
// source: "dn" (D+E) or "t1" (a T1 rule).
std::vector<TypoEntry> TypoWindowOrder(const std::vector<TypoCandidate>& cands, bool enc,
                                       bool leftTextEmpty,
                                       const std::function<double(const std::string&)>* D,
                                       std::string* source);

// The model part of a window (compute queue): E for every candidate (one
// uncached encoder pass each: context + the candidate), then D (one
// next-token call on leftText; decoder may be null), then the window order.
// `cancelled` is checked before every model call; once set, nothing more is
// called and false is returned.
bool ScoreTypoWindow(Engine* engine, Decoder* decoder, std::vector<TypoCandidate> cands,
                     const std::vector<std::string>& context, const std::string& leftText,
                     const std::atomic<bool>& cancelled, std::vector<TypoEntry>* entries,
                     std::string* source);

// ---------------------------------------------------------------------------
// v3.2b: the correction entry for a typo that gave a DIFFERENT LEGAL syllable
// (reports/typo.md §8; frozen typo/t2_frozen.json: E+D w = 0.25, neighbor prior
// row : diagonal = 1 : 0.5 with "other" = diagonal, theta 1.766).
//
// For the syllable at reading position p of the buffer, over the T1 slot1_nb
// candidates c of its reading:
//   E(c) = total(buffer with c at p) - total(buffer)   (A' in-walk path totals:
//          sum over the walked nodes of LM + beta * neural; no picks, no pins)
//   D(c) = score(c) - score(typed)  with score(x) = log sum, over x's chars, of
//          P_dec(next text starts with the char | text left of p)
//   gain(c) = E(c) + log w(c) + 0.25 * D(c)
// The best gain (ties: reading order) is offered when it is >= theta.

constexpr double kLegalTypoBlend = 0.25;
constexpr double kLegalTypoTheta = 1.765735609807406;

enum class NeighborDirection { kRow = 0, kDiag = 1, kOther = 2 };

// log w of a direction: 0 for a same-row neighbor, log 0.5 otherwise.
double LegalTypoLogWeight(NeighborDirection d);

// t2_e.cpp dirCands(): row = a same-row left/right neighbor-key substitution in
// the typed syllable's canonical keys gives the candidate; else diag = an
// up/down-row neighbor (Space <-> the bottom row counts as diag); else other.
std::vector<NeighborDirection> LegalTypoDirections(
    const std::string& typed, const std::vector<TypoCandidate>& cands,
    Formosa::Gramambular2::LanguageModel& lm);

struct LegalTypoDecision {
  bool offered = false;
  size_t index = 0;   // into cands
  double gain = 0;
};
// eGain / dGain per candidate. t2.py gains() + ranked(): in this operation order.
LegalTypoDecision DecideLegalTypo(const std::vector<TypoCandidate>& cands,
                                  const std::vector<NeighborDirection>& dirs,
                                  const std::vector<double>& eGain,
                                  const std::vector<double>& dGain);

// Everything the score needs from the language model, gathered on the main thread.
struct LegalTypoPlan {
  size_t pos = 0;
  std::string typed;
  std::vector<std::string> typedChars;                 // single-char unigram values of typed
  std::vector<TypoCandidate> cands;
  std::vector<NeighborDirection> dirs;
  std::vector<std::vector<std::string>> chars;         // per candidate, like typedChars
  std::vector<SlothEGrid::SpanPatch> typedPatch;
  std::vector<std::vector<SlothEGrid::SpanPatch>> patches;  // per candidate
};
// Main thread. The buffer's reading at pos must be a legal syllable.
LegalTypoPlan MakeLegalTypoPlan(SlothEGrid* grid, size_t pos,
                                Formosa::Gramambular2::LanguageModel& lm);

struct LegalTypoResult {
  bool finished = false;      // false: cancelled, or a model was not available
  bool retry = false;         // not finished only because the next-token function was not loaded yet
  bool offered = false;
  std::string reading;        // the offered syllable
  std::string value;          // the char the buffer shows there after the change
  double gain = 0;
  double eGain = 0;
  double dGain = 0;
  size_t candidates = 0;
};

// Compute queue. base = a snapshot of the buffer's grid (SlothEGrid::snapshot());
// readings = its readings; context = carried encoder syllables (v3.1a);
// leftText = carried text + shown text left of pos; a decoder without the
// next-token function (or not loaded yet) finishes nothing (retry). `cancelled`
// is checked before every model call; the return value is false only when it was
// set (the result says whether the job finished, and why not).
bool ScoreLegalTypo(Engine* engine, Decoder* decoder, const SlothEGrid& base,
                    const LegalTypoPlan& plan,
                    const std::vector<std::string>& readings,
                    const std::vector<std::string>& context,
                    const std::string& leftText, const InWalkParams& params,
                    const std::atomic<bool>& cancelled, LegalTypoResult* result);

}  // namespace McBopomofoSlothE

#endif  // SOURCE_SLOTHE_SLOTHETYPO_H_
