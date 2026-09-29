// v3.1b correction window; see SlothETypo.h. Candidate generation is
// typo/typo_cli.cpp (candidates(), neighbors(), Sim) line for line; the
// scores follow typo/score.py scored() and typo/t2_ill.py rank() in the same
// arithmetic order, so ties break the same way.

#include "SlothETypo.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <tuple>

#include "Mandarin.h"

namespace McBopomofoSlothE {

namespace {

using Formosa::Mandarin::BopomofoKeyboardLayout;
using Formosa::Mandarin::BopomofoReadingBuffer;
using BPMF = Formosa::Mandarin::BopomofoSyllable;

// The study's layout (Standard / Dachen) for the neighbor keys.
// ponytail: other layouts get the same Standard-layout neighbors; a per-layout
// table needs its own study.
const BopomofoKeyboardLayout* Layout() { return BopomofoKeyboardLayout::StandardLayout(); }

// Physical neighbors on a US keyboard (row stagger), restricted to layout keys
// plus Space (typo_cli neighbors()).
const std::map<char, std::string>& Neighbors() {
  static const std::map<char, std::string> table = [] {
    const char* rows[4] = {"1234567890-=", "qwertyuiop[]", "asdfghjkl;'", "zxcvbnm,./"};
    std::map<char, std::string> nb;
    auto ok = [](char k) { return k == ' ' || !Layout()->keyToComponents(k).empty(); };
    for (int r = 0; r < 4; ++r) {
      int n = static_cast<int>(strlen(rows[r]));
      for (int c = 0; c < n; ++c) {
        char k = rows[r][c];
        if (!ok(k)) continue;
        std::string s;
        auto add = [&](int rr, int cc) {
          if (rr < 0 || rr > 3 || cc < 0 || cc >= static_cast<int>(strlen(rows[rr]))) return;
          if (ok(rows[rr][cc])) s += rows[rr][cc];
        };
        add(r, c - 1);
        add(r, c + 1);
        add(r - 1, c);
        add(r - 1, c + 1);
        add(r + 1, c - 1);
        add(r + 1, c);
        if (r == 3 && c >= 1 && c <= 7) s += ' ';
        nb[k] = s;
      }
    }
    nb[' '] = "xcvbnm,";
    return nb;
  }();
  return table;
}

std::string KeysOf(const std::string& reading) {
  BPMF syl = BPMF::FromComposedString(reading);
  std::string k = Layout()->keySequenceFromSyllable(syl);
  if (!syl.hasToneMarker()) k += ' ';  // tone 1 is completed by Space
  return k;
}

// KeyHandler's BPMF branch for one key sequence: the composed readings.
struct Sim {
  BopomofoReadingBuffer buf{Layout()};
  std::vector<std::string> composed;
  int stray = 0;
  void key(char k) {
    bool valid = buf.isValidKey(k);
    if (valid) {
      buf.combineKey(k);
      if (!buf.hasToneMarker()) return;
    }
    bool compose = valid && buf.hasToneMarker() && !buf.hasToneMarkerOnly();
    compose |= (!buf.isEmpty() && (k == ' ' || k == '\r'));
    if (compose) {
      composed.push_back(buf.syllable().composedString());
      buf.clear();
      return;
    }
    if (valid) return;
    if (k == ' ') ++stray;
  }
};

int CmvDist(BPMF a, BPMF b) {
  return (a.consonantComponent() != b.consonantComponent()) +
         (a.middleVowelComponent() != b.middleVowelComponent()) +
         (a.vowelComponent() != b.vowelComponent());
}

const std::vector<BPMF::Component>& Components(int which) {
  static const std::vector<std::vector<BPMF::Component>> all = [] {
    std::vector<BPMF::Component> cons {0}, vows {0};
    for (BPMF::Component c = BPMF::B; c <= BPMF::S; ++c) cons.push_back(c);
    for (BPMF::Component v = BPMF::A; v <= BPMF::ERR; v += 0x80) vows.push_back(v);
    return std::vector<std::vector<BPMF::Component>> {
        cons, {0, BPMF::I, BPMF::U, BPMF::UE}, vows,
        {BPMF::Tone1, BPMF::Tone2, BPMF::Tone3, BPMF::Tone4, BPMF::Tone5}};
  }();
  return all[static_cast<size_t>(which)];
}

bool IsTone(const std::string& ch) { return ch == "ˊ" || ch == "ˇ" || ch == "ˋ" || ch == "˙"; }

bool IsFinal(const std::string& ch) {
  static const char* finals[] = {"ㄚ", "ㄛ", "ㄜ", "ㄝ", "ㄞ", "ㄟ", "ㄠ", "ㄡ", "ㄢ", "ㄣ", "ㄤ", "ㄥ", "ㄦ"};
  return std::find(std::begin(finals), std::end(finals), ch) != std::end(finals);
}

int NonToneChars(const std::string& reading) {
  int n = 0;
  for (const std::string& ch : Utf8Chars(reading)) n += !IsTone(ch);
  return n;
}

// t2_fin add_final(): typed has no final; reading = typed + one final.
bool AddsFinal(const std::string& typed, const std::string& reading) {
  std::vector<std::string> t = Utf8Chars(typed);
  if (std::any_of(t.begin(), t.end(), IsFinal)) return false;
  std::vector<std::string> r = Utf8Chars(reading);
  std::string rest;
  size_t kept = 0;
  for (const std::string& ch : r) {
    if (!IsFinal(ch)) {
      rest += ch;
      ++kept;
    }
  }
  return kept + 1 == r.size() && rest == typed;
}

}  // namespace

std::vector<TypoCandidate> TypoCandidates(const std::string& typed,
                                          Formosa::Gramambular2::LanguageModel& lm) {
  BPMF s = BPMF::FromComposedString(typed);
  std::map<std::string, std::pair<BPMF, int>> out;  // reading -> (syllable, neighbor)
  auto consider = [&](BPMF c, int nb) {
    if (!(c.hasConsonant() || c.hasMiddleVowel() || c.hasVowel()) || c == s) return;
    std::string r = c.composedString();
    if (!lm.hasUnigrams(r)) return;
    auto it = out.find(r);
    if (it == out.end()) {
      out[r] = {c, nb};
    } else {
      it->second.second |= nb;
    }
  };
  auto cm = s.consonantComponent(), mm = s.middleVowelComponent(), vv = s.vowelComponent();
  for (auto c : Components(0))
    for (auto m : Components(1))
      for (auto v : Components(2)) {
        int d = (c != cm) + (m != mm) + (v != vv);
        if (d > 1) continue;
        for (auto t : Components(3)) consider(BPMF(static_cast<BPMF::Component>(c | m | v | t)), 0);
      }
  std::string k = KeysOf(typed);
  for (size_t p = 0; p < k.size(); ++p) {
    auto it = Neighbors().find(k[p]);
    if (it == Neighbors().end()) continue;
    for (char n : it->second) {
      std::string kk = k;
      kk[p] = n;
      Sim sim;
      for (char x : kk) sim.key(x);
      if (sim.composed.size() == 1 && sim.buf.isEmpty() && !sim.stray)
        consider(BPMF::FromComposedString(sim.composed[0]), 1);
    }
  }
  std::vector<TypoCandidate> cands;
  const int typedSlots = NonToneChars(typed);
  for (auto& [r, pr] : out) {
    TypoCandidate c;
    c.reading = r;
    c.cmv = CmvDist(s, pr.first);
    c.toneChanged = s.toneMarkerComponent() != pr.first.toneMarkerComponent();
    c.neighbor = pr.second != 0;
    if (!(c.cmv + c.toneChanged <= 1 || c.neighbor)) continue;  // slot1_nb
    c.lm = -1e9;
    for (const auto& u : lm.getUnigrams(r)) {
      c.lm = std::max(c.lm, u.score());
      c.chars.push_back(u.value());
    }
    c.dslots = NonToneChars(r) - typedSlots;
    c.addFinal = AddsFinal(typed, r);
    cands.push_back(std::move(c));
  }
  return cands;
}

// typo/score.py scored(), same operation order.
double TypoLMPriorScore(const TypoCandidate& c) {  // (i') a_lm 1, w2 1, wnb 0.5, wlen 0.5, wt -1
  double v = 1.0 * c.lm;
  v -= 1.0 * (c.cmv + c.toneChanged >= 2) + -1.0 * c.toneChanged;
  v += 0.5 * c.neighbor + 0.5 * c.dslots;
  return v;
}

double TypoT1Score(const TypoCandidate& c) {  // (iii) a_lm 2, w2 1, wnb 1, wlen 1, wt -1, E = lsm_p
  double v = 2.0 * c.lm;
  v += c.encoder;
  v -= 1.0 * (c.cmv + c.toneChanged >= 2) + -1.0 * c.toneChanged;
  v += 1.0 * c.neighbor + 1.0 * c.dslots;
  return v;
}

double TypoEncoderScore(const ForwardResult& r, size_t pos, const Vocabulary& vocab,
                        const std::vector<std::string>& chars) {
  if (r.base != nullptr || pos >= r.ids.size()) return kTypoNoEncoder;
  const size_t C = static_cast<size_t>(r.charCount);
  const float* row = r.logits.data() + pos * C;
  double mx = -std::numeric_limits<double>::infinity();
  for (size_t k = 0; k < C; ++k) mx = std::max(mx, static_cast<double>(row[k]));
  double sum = 0;
  for (size_t k = 0; k < C; ++k) sum += std::exp(static_cast<double>(row[k]) - mx);
  const double lse = mx + std::log(sum);  // full-vocab log-softmax (typo/score.py _lsm_row)
  double best = -std::numeric_limits<double>::infinity();
  bool any = false;
  for (const std::string& ch : chars) {
    int32_t id = vocab.charId(ch);
    if (id < 0 || id == kUnkCharId) continue;
    best = std::max(best, static_cast<double>(row[static_cast<size_t>(id)]) - lse);
    any = true;
  }
  return any ? best : kTypoNoEncoder;
}

namespace {

std::vector<TypoEntry> CharsInOrder(const std::vector<std::pair<std::string, std::string>>& pairs) {
  std::vector<TypoEntry> out;
  std::vector<std::string> seen;
  for (const auto& [r, ch] : pairs) {
    if (std::find(seen.begin(), seen.end(), ch) != seen.end()) continue;
    seen.push_back(ch);
    out.push_back({r, ch});
    if (out.size() == kTypoMaxEntries) break;
  }
  return out;
}

}  // namespace

std::vector<TypoEntry> TypoSyllableOrder(const std::vector<TypoCandidate>& cands,
                                         const std::vector<double>& scores) {
  std::vector<size_t> idx(cands.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
    return std::make_tuple(-scores[a], cands[a].reading) < std::make_tuple(-scores[b], cands[b].reading);
  });
  std::vector<std::pair<std::string, std::string>> pairs;
  for (size_t i : idx)
    for (const std::string& ch : cands[i].chars) pairs.emplace_back(cands[i].reading, ch);
  return CharsInOrder(pairs);
}

std::vector<TypoEntry> TypoDecoderOrder(const std::vector<TypoCandidate>& cands,
                                        const std::function<double(const std::string&)>& D) {
  std::vector<std::tuple<double, std::string, std::string>> keyed;  // t2_ill.rank pairs
  for (const TypoCandidate& c : cands) {
    double base = 0.0 + 1.0 * TypoT1Score(c) + (c.addFinal ? kTypoAddFinalBonusWindow : 0.0);
    for (const std::string& ch : c.chars) keyed.emplace_back(-(D(ch) + base), c.reading, ch);
  }
  std::sort(keyed.begin(), keyed.end());
  std::vector<std::pair<std::string, std::string>> pairs;
  for (const auto& [k, r, ch] : keyed) pairs.emplace_back(r, ch);
  return CharsInOrder(pairs);
}

std::vector<TypoEntry> TypoWindowOrder(const std::vector<TypoCandidate>& cands, bool enc,
                                       bool leftTextEmpty,
                                       const std::function<double(const std::string&)>* D,
                                       std::string* source) {
  *source = "t1";
  std::vector<double> scores;
  if (!enc) {  // no encoder pass: the synchronous LM+prior rule
    for (const TypoCandidate& c : cands) scores.push_back(TypoLMPriorScore(c));
    return TypoSyllableOrder(cands, scores);
  }
  if (leftTextEmpty || D == nullptr) {
    // empty left text: the study's fallback (T1 + the window's bonus 3);
    // no next-token scores: the T1 rule with its own frozen bonus 1
    double bonus = leftTextEmpty ? kTypoAddFinalBonusWindow : kTypoAddFinalBonusT1;
    for (const TypoCandidate& c : cands) scores.push_back(TypoT1Score(c) + (c.addFinal ? bonus : 0.0));
    return TypoSyllableOrder(cands, scores);
  }
  *source = "dn";
  return TypoDecoderOrder(cands, *D);
}

bool ScoreTypoWindow(Engine* engine, Decoder* decoder, std::vector<TypoCandidate> cands,
                     const std::vector<std::string>& context, const std::string& leftText,
                     const std::atomic<bool>& cancelled, std::vector<TypoEntry>* entries,
                     std::string* source) {
  bool enc = !cands.empty();
  for (TypoCandidate& c : cands) {
    if (cancelled.load()) return false;
    double ms = 0;
    bool hit = false;
    std::vector<std::string> one {c.reading};
    auto r = engine->forward(one, &ms, &hit, &context, false);
    if (r == nullptr) {
      enc = false;
      break;
    }
    c.encoder = TypoEncoderScore(*r, 0, engine->vocabulary(), c.chars);
  }
  std::vector<float> lp;
  bool haveD = false;
  if (enc && !leftText.empty() && decoder != nullptr && decoder->hasFirstChars()) {
    if (cancelled.load()) return false;
    DecoderCallStats stats;
    haveD = decoder->nextTokenLogProbs(leftText, &lp, &stats);
  }
  if (cancelled.load()) return false;
  std::function<double(const std::string&)> D = [decoder, &lp](const std::string& ch) {
    return decoder->firstCharLogProb(lp, ch);
  };
  *entries = TypoWindowOrder(cands, enc, leftText.empty(), haveD ? &D : nullptr, source);
  return true;
}

}  // namespace McBopomofoSlothE
