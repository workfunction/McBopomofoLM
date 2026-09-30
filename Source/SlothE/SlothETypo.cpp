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
#include <set>
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

// ---------------------------------------------------------------------------
// v3.2b legal-but-wrong correction entry

namespace {

// typo/t2_e.cpp splitNeighbors(): the neighbor keys split by direction.
struct DirectionTables {
  std::map<char, std::string> row, diag;
};

const DirectionTables& Directions() {
  static const DirectionTables tables = [] {
    DirectionTables t;
    const char* rows[4] = {"1234567890-=", "qwertyuiop[]", "asdfghjkl;'", "zxcvbnm,./"};
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < static_cast<int>(strlen(rows[r])); ++c) {
        char k = rows[r][c];
        auto nb = Neighbors().find(k);
        if (nb == Neighbors().end()) continue;
        for (char n : nb->second) {
          bool row = false;
          for (int d : {-1, 1}) {
            if (c + d >= 0 && c + d < static_cast<int>(strlen(rows[r])) && rows[r][c + d] == n) row = true;
          }
          (row ? t.row : t.diag)[k] += n;
        }
      }
    }
    t.diag[' '] = Neighbors().at(' ');
    return t;
  }();
  return tables;
}

std::vector<std::string> SingleChars(const std::vector<std::string>& values) {
  std::vector<std::string> out;
  for (const std::string& v : values) {
    if (Utf8Length(v) == 1) out.push_back(v);
  }
  return out;
}

}  // namespace

double LegalTypoLogWeight(NeighborDirection d) {
  return d == NeighborDirection::kRow ? 0.0 : std::log(0.5);
}

std::vector<NeighborDirection> LegalTypoDirections(
    const std::string& typed, const std::vector<TypoCandidate>& cands,
    Formosa::Gramambular2::LanguageModel& lm) {
  std::set<std::string> row, diag;
  const std::string k = KeysOf(typed);
  for (int pass = 0; pass < 2; ++pass) {
    const std::map<char, std::string>& table = pass ? Directions().diag : Directions().row;
    for (size_t p = 0; p < k.size(); ++p) {
      auto it = table.find(k[p]);
      if (it == table.end()) continue;
      for (char n : it->second) {
        std::string kk = k;
        kk[p] = n;
        Sim sim;
        for (char x : kk) sim.key(x);
        if (sim.composed.size() == 1 && sim.buf.isEmpty() && !sim.stray && sim.composed[0] != typed &&
            lm.hasUnigrams(sim.composed[0])) {
          (pass ? diag : row).insert(sim.composed[0]);
        }
      }
    }
  }
  std::vector<NeighborDirection> out;
  for (const TypoCandidate& c : cands) {
    out.push_back(row.count(c.reading) ? NeighborDirection::kRow
                                       : diag.count(c.reading) ? NeighborDirection::kDiag
                                                               : NeighborDirection::kOther);
  }
  return out;
}

LegalTypoDecision DecideLegalTypo(const std::vector<TypoCandidate>& cands,
                                  const std::vector<NeighborDirection>& dirs,
                                  const std::vector<double>& eGain,
                                  const std::vector<double>& dGain) {
  LegalTypoDecision best;
  bool any = false;
  for (size_t i = 0; i < cands.size(); ++i) {
    // t2.py: e = E + pr; d = D + pr; gain = e + w * (d - pr)
    const double pr = LegalTypoLogWeight(dirs[i]);
    const double e = eGain[i] + pr;
    const double d = dGain[i] + pr;
    const double gain = e + kLegalTypoBlend * (d - pr);
    if (!std::isfinite(gain)) continue;
    if (!any || gain > best.gain || (gain == best.gain && cands[i].reading < cands[best.index].reading)) {
      best.index = i;
      best.gain = gain;
      any = true;
    }
  }
  best.offered = any && best.gain >= kLegalTypoTheta;
  return best;
}

LegalTypoPlan MakeLegalTypoPlan(SlothEGrid* grid, size_t pos,
                                Formosa::Gramambular2::LanguageModel& lm) {
  LegalTypoPlan plan;
  if (grid == nullptr || pos >= grid->readings().size()) {
    return plan;
  }
  plan.pos = pos;
  plan.typed = grid->readings()[pos];
  std::vector<std::string> typedValues;
  for (const auto& u : lm.getUnigrams(plan.typed)) typedValues.push_back(u.value());
  plan.typedChars = SingleChars(typedValues);
  plan.cands = TypoCandidates(plan.typed, lm);
  plan.dirs = LegalTypoDirections(plan.typed, plan.cands, lm);
  plan.typedPatch = grid->patchFor(pos, plan.typed);
  for (const TypoCandidate& c : plan.cands) {
    plan.chars.push_back(SingleChars(c.chars));
    plan.patches.push_back(grid->patchFor(pos, c.reading));
  }
  return plan;
}

bool ScoreLegalTypo(Engine* engine, Decoder* decoder, const SlothEGrid& base,
                    const LegalTypoPlan& plan,
                    const std::vector<std::string>& readings,
                    const std::vector<std::string>& context,
                    const std::string& leftText, const InWalkParams& params,
                    const std::atomic<bool>& cancelled, LegalTypoResult* result) {
  *result = LegalTypoResult();
  if (engine == nullptr || decoder == nullptr || !decoder->hasFirstChars() ||
      plan.pos >= readings.size() || plan.cands.size() != plan.patches.size()) {
    return false;
  }
  result->candidates = plan.cands.size();
  auto total = [&](const std::string& reading, const std::vector<SlothEGrid::SpanPatch>& patch,
                   double* t, std::shared_ptr<const ForwardResult>* keep) {
    std::vector<std::string> rs = readings;
    rs[plan.pos] = reading;
    double ms = 0;
    bool hit = false;
    std::shared_ptr<const ForwardResult> r = engine->forward(rs, &ms, &hit, &context, false);
    if (r == nullptr) return false;
    std::unique_ptr<SlothEGrid> grid = base.snapshotWithReading(plan.pos, reading, patch);
    if (!InWalkTotal(grid.get(), *r, engine->vocabulary(), engine->variants(), params, t)) return false;
    if (keep != nullptr) *keep = r;
    return true;
  };
  if (cancelled.load()) return false;
  double baseTotal = 0;
  if (!total(plan.typed, plan.typedPatch, &baseTotal, nullptr)) return true;  // finished, nothing to offer
  std::vector<double> eGain(plan.cands.size(), -std::numeric_limits<double>::infinity());
  for (size_t i = 0; i < plan.cands.size(); ++i) {
    if (cancelled.load()) return false;
    double t = 0;
    if (total(plan.cands[i].reading, plan.patches[i], &t, nullptr)) eGain[i] = t - baseTotal;
  }
  if (cancelled.load()) return false;
  std::vector<float> lp;
  DecoderCallStats stats;
  if (!decoder->nextTokenLogProbs(leftText, &lp, &stats)) {
    result->retry = stats.unavailable;  // the n* function is loading: try again later
    return true;
  }
  const double typedScore = decoder->charsLogProb(lp, plan.typedChars);
  std::vector<double> dGain(plan.cands.size());
  for (size_t i = 0; i < plan.cands.size(); ++i) {
    dGain[i] = decoder->charsLogProb(lp, plan.chars[i]) - typedScore;
  }
  LegalTypoDecision d = DecideLegalTypo(plan.cands, plan.dirs, eGain, dGain);
  result->finished = true;
  result->gain = d.gain;
  if (!d.offered) {
    return true;
  }
  result->offered = true;
  result->reading = plan.cands[d.index].reading;
  result->eGain = eGain[d.index];
  result->dGain = dGain[d.index];
  // the char the buffer shows at pos once the syllable is changed
  std::vector<std::string> rs = readings;
  rs[plan.pos] = result->reading;
  double ms = 0;
  bool hit = false;
  std::shared_ptr<const ForwardResult> r = engine->forward(rs, &ms, &hit, &context, false);
  std::string shown;
  if (r != nullptr) {
    std::unique_ptr<SlothEGrid> grid = base.snapshotWithReading(plan.pos, result->reading, plan.patches[d.index]);
    auto walk = grid->rescoredWalk(*r, engine->vocabulary(), engine->variants(), params, nullptr);
    std::vector<std::string> chars = DisplayedChars(walk, rs.size());
    if (plan.pos < chars.size()) shown = chars[plan.pos];
  }
  if (shown.empty() && !plan.chars[d.index].empty()) shown = plan.chars[d.index][0];
  if (shown.empty()) {
    result->offered = false;  // nothing to label the entry with
    return true;
  }
  result->value = shown;
  return true;
}

}  // namespace McBopomofoSlothE
