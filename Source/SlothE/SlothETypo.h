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

}  // namespace McBopomofoSlothE

#endif  // SOURCE_SLOTHE_SLOTHETYPO_H_
