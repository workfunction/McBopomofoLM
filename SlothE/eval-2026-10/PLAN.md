# McBopomofoLM × D512 evaluation — context, findings, plan (2026-10-13)

## User's goal
User maintains McBopomofoLM (repo /home/user/McBopomofoLM, branch lm): a fork of 小麥注音 (macOS Zhuyin IME) that adds two neural models on Apple Neural Engine:
- SlothE-T 25M ternary bidirectional encoder (per-position legal-masked char distribution over the whole buffer's readings) — its log-prob is added into McBopomofo's Viterbi walk (gramambular2 grid over data.txt dictionary) with beta=0.3 ("in-walk").
- pred_q35_60m decoder (Qwen3.5 60M, text-only causal LM, chat-register fine-tuned): for each node of the in-walk walk with ≥2 same-length "aligned" unigrams, take encoder top-3, b = enc + 2·dec, pin top if softmax p ≥ 0.5 and ≠ shown; re-walk with pins (config A').
- v3.1 adds carried context (encoder sees committed readings, decoder sees last 64 committed chars) and a typo-correction window.
User preferences: "60-point" solutions by default (cheap, low complexity), but full answers; recommend one option.

The user asked: would 0c's public D512 reranker (Funmula/0c-sequence-d512, MIT, 32.6M causal Transformer, ONNX CPU, scores log P(candidate | ≤64 chars confirmed context, span readings); trained on 1.245B chars news+financial forum before 2026-09-01) be better in our IME, and what does 0c have that we lack. Their real complaints: 在/再 confusion (both ㄗㄞˋ), and sentences coming out "fragmented" (example: 那週是馬丁路德金連假 → default 那州市馬丁路德金蓮價).

## Findings so far
1. 0c features we lack: commit-time punctuation model (public, MIT, F1 0.82), Tab completion, fuzzy-sound pairs, mixed zh-en, emoji menu, Win/Linux.
2. Offline dev test (500 Common Voice dev sentences, 4,352 chars; gold recovered by matching readings to CV text, 500/500): decoder-slot swap, 2-fold CV over (λ,τ):
   current q35 96.39%/77.0% (λ,τ tuned on this dev earlier, so in-sample); D512 alone ≤96.25%; q35+D512 96.55%/78.4% (+10/−3 chars, n.s.). Model-alone top-1 among requests with gold in top-3: q35 963/1068, D512 944/1068. Decoder stage can fix at most 100 chars (gold in top-3 but not shown); in-walk leaves 171 wrong chars.
3. 在/再 (36 ㄗㄞˋ cases in dev, only 4 再): whole sentence ours 35/36; chunked (v3.1) ours 31/36 vs D512 32/36. Found mechanism: data.txt has 808 phrases with 在 lacking a 再 twin (會在, 還在, 就在, 他在, 你在, 但在…) vs 211 the other way; e.g. 會再加以討論 → node 會在 locked (encoder prefers 再 by 2.65 nats; no decoder request because no other aligned unigram). Generic "encoder overrides inside multi-char words" is harmful (+7/−51 at 2 nats).
4. Built a validated offline full pipeline on Linux (scratchpad/work/pipeline.py, walkcli/walk_cli built from repo C++ sources with CoreFoundation/CommonCrypto shims; PyTorch 25M encoder from HF fp32 master ≈ app within ~0.02 nats; transformers q35 decoder): stock & final walks match fixtures 500/500; decoder decisions match 1103/1104. Missing: carried context/chunking, user learning, typo window.
5. User sentence (typed in one buffer, no learning): stock 那洲是馬丁路德今廉價; ours 那州市馬丁路德金蓮價. Causes: encoder 州(−0.36) > 周(−2.34), 是≈市; walk segments ㄐㄧㄣ-ㄌㄧㄢˊ as dictionary word 金蓮 → 連假 unreachable for any decoder (decoder only swaps within fixed node spans); q35 flips 是→市 using stale context "那州". Pipeline design issue: decoder request contexts use the pre-pin in-walk text, not earlier pins. D512 with sequential context gives 那周是馬丁路德金蓮駕. D512 whole-sentence ranking puts the gold sentence first, but our pipeline never proposes it.

## In-flight
- Two Sonnet agents collecting fresh text dated ≥2026-10-01 (unseen by both models): news (≥1,000 units, 6–32 Han chars) and PTT/Dcard chat (≥1,000 units, ≥200 containing 再), dedup vs Common Voice.
- dev500 run of variants: current, sequential-context q35, D512 (±sequential), q35+D512 (±sequential).

## Plan (to review)
1. Label fresh units with g2pW (bopomofo). Normalize to what users type: undo 一/不 tone sandhi (g2pW outputs 一 as ㄧˊ/ㄧˋ in 一次/一個, users type ㄧ; 不 → ㄅㄨˋ). Drop units whose syllables McBopomofo can't take or whose gold chars are not reachable (no dictionary unigram per char with that reading). Keep a 在/再 subset (all ㄗㄞˋ positions) and user_reported cases.
2. Freeze configs on dev500 (CV-tuned), then evaluate once on fresh set: stock, current, sequential-context q35, D512 variants, q35+D512; plus targeted 在/再 fix candidate (for 在/再 inside a dictionary phrase, inject the swapped twin as an aligned candidate so encoder+decoder decide). Report char acc / sentence acc / 在再 accuracy, paired bootstrap CIs, fixes vs breaks, per domain (news vs chat).
3. Recommend to the user what (if anything) to change in the app; app changes would need Mac/ANE validation by the user.

## Open questions for the advisor
- Is this the right goal framing and evaluation design? Anything simpler that answers "should we adopt D512 / how to fix 在再 and fragmentation"?
- Sandhi normalization and reachability filtering: right call?
- Tuning on dev500 (Common Voice, in-sample for current λ,τ) then single-shot on fresh set: OK?
- Is the 在/再 twin-injection fix the simplest effective fix, vs. e.g. data.txt edits (add 再 twins / delete pseudo-phrases like 會在), or lowering beta, or letting the decoder see segment alternatives (proposing different segmentations, e.g. whole-sentence D512 rescoring of n-best walks)?
- Any bigger architectural change that would address "fragmentation" (segmentation locked by the dictionary walk) more simply?

## Pre-registration (frozen 2026-10-13, before the fresh set is opened)
Frozen configs (offline pipeline, beta 0.3, top-3, decoder ctx '' since units are single buffers):
- A stock McBopomofo (data.txt)
- B current app: q35 span-only decoder, pre-pin context, λ2 τ.5, data.txt
- C q35 full-buffer decoder (candidate + rest of buffer), sequential context, λ1 τ.5, data.txt
- D = C with data_twins.txt (18 再-twins, score = 在-score − 0.1)
- E q35 + 0.25·D512 full-buffer, sequential, λ1 τ.5, data_twins.txt
Context-trim rule (also the app rule): keep candidate+buffer intact, trim left context to fit 96 tokens (q35); D512 context capped at 64 chars.
Test units: news (PTS, SETN, ETtoday, Storm; 2026-10-01..13) and chat (PTT/Dcard, same window), labeled by g2pW snapped to dictionary-legal readings (一→ㄧ, 不→ㄅㄨˋ, phrase-reading override by greedy longest data.txt match); units with an illegal reading are dropped and counted. user_reported cases are reported separately; user-001 is dev cv08868 and excluded.
Pools: "general" = units without 再 (unbiased by the 再 oversampling), random sample ≤1,500 per domain (seed 20261013); "ㄗㄞˋ" = every ㄗㄞˋ position in all 再 units + all 在 units (chat 再 units manually adjudicated; suspect/misused ones excluded).
Metrics: char acc / sentence acc on general pool per domain; 在/再 confusion matrix on the ㄗㄞˋ pool; fix/break lists; in-span vs reseg error breakdown; decoder request counts/lengths.
Decision rules:
1. Ship C over B if C−B char acc on chat general pool has a paired-bootstrap (units, 2,000 draws) 95% CI excluding 0 (news reported but not decisive; if chat is unavailable, news decides).
2. Ship twins (D over C) if McNemar over ㄗㄞˋ positions favors D (p<0.05) and 在→再 breaks ≤ 1/3 of 再 fixes.
3. Adopt D512 (E over D) only if chat char-acc gain ≥ 0.2pt with CI excluding 0; otherwise "no D512" regardless of news.
4. One run, no re-tuning after opening. B→C bundles seq ctx + full-buffer + λ2→1; attribution only from dev.
