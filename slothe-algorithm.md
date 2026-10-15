# SlothE 神經重排序演算法說明

本文件是 [algorithm.md](algorithm.md) 的姊妹篇，說明 McBopomofoLM 在小麥注音的選字引擎（Gramambular2）之上加的 SlothE 神經重排序：兩個跑在 Apple Neural Engine（ANE）上的模型怎麼參與斷詞、選字、選字窗排序和打錯救回，以及模型怎麼載入。給想看懂或修改這部分程式的人。

- 適用範圍：`lm` 分支 v3.4p（commit `9bf72996`）。
- 每一節都標了程式碼位置（`檔案:行號`）。所有公式和常數都取自程式碼；從程式碼推得、但沒有直接寫在程式碼裡的說法，標為〔推論〕。
- 使用者看的介紹在 [README](README.markdown)；效能、安裝、診斷記錄在 [DEVELOPMENT.md](DEVELOPMENT.md)。

## 建置與測試

建置方式見 `DEVELOPMENT.md`「自行編譯」和 `AGENTS.md`。完整測試約 22–28 分鐘，其中 10 個載入模型的測試佔約 18 分鐘（每次重新 build，ANE 快取就失效）；同一時間只能有一個測試程式載入模型。

---

## 目錄

- [整體架構](#整體架構)
  - [在 Gramambular2 之上](#在-gramambular2-之上)
  - [每個按鍵的時間線](#每個按鍵的時間線)
  - [世代與過期](#世代與過期)
- [模型](#模型)
  - [Encoder](#encoder)
  - [Decoder](#decoder)
- [斷詞時的重新打分](#斷詞時的重新打分)
  - [邊的分數](#邊的分數)
  - [多字節點的 neural 分數](#多字節點的-neural-分數)
  - [異體字 guard 與特例](#異體字-guard-與特例)
- [Decoder 閘門](#decoder-閘門)
  - [產生 request 的節點](#產生-request-的節點)
  - [整句打分的字串](#整句打分的字串)
  - [混合與決策](#混合與決策)
  - [超過長度時](#超過長度時)
  - [分數快取](#分數快取)
  - [舊的逐鍵路徑與停頓複查](#舊的逐鍵路徑與停頓複查)
- [選字窗排序](#選字窗排序)
- [帶前文](#帶前文)
- [打錯救回](#打錯救回)
  - [不合法讀音的修正選字窗](#不合法讀音的修正選字窗)
  - [合法但打錯的修正項](#合法但打錯的修正項)
- [「再」雙胞胎詞](#再雙胞胎詞)
- [載入與 Neural Engine](#載入與-neural-engine)
  - [檔案核對](#檔案核對)
  - [模型副本](#模型副本)
  - [放置檢查](#放置檢查)
  - [lazy 函式與 eager pass](#lazy-函式與-eager-pass)
  - [清快取的標記與自我修復](#清快取的標記與自我修復)
  - [退回的層次](#退回的層次)
- [實際範例](#實際範例)
- [程式碼與測試位置](#程式碼與測試位置)

---

## 整體架構

### 在 Gramambular2 之上

SlothE 不改 Gramambular2 的資料結構，也不改網格裡的節點。它做三件事：

1. **重新打分的 walk（in-walk）**：`SlothEGrid` 繼承 `ReadingGrid`，多一個 `rescoredWalk()`。它跟原版 `walk()` 走同一張 DAG，只是每條邊的分數多了 encoder 的項。結果是一個 `WalkResult`，key handler 拿它取代 `_latestWalk`；選到的 unigram 跟節點目前的值不同時，放的是節點的私有複本（`SelectedNode`），網格本身不動（`Source/SlothE/SlothEWalk.cpp:21`、`:137`，`Source/SlothE/SlothEWalk.h:203`）。
2. **Decoder 閘門**：對有兩個以上候選字的節點，用 decoder 給候選字打分，夠有把握時產生一個 `WalkPin`，在同一個 in-walk 裡把那個節點固定成某個值（`Source/SlothE/SlothEWalk.h:45`）。
3. **選字窗排序、帶前文、打錯救回**：都建立在前兩者的分數上。

開關（`Source/KeyHandler.mm:2827`、`:3245`）：

| 條件 | 行為 |
|---|---|
| 不是注音模式、「AI 選字」（`slothERerankEnabled`）關掉、或 encoder 沒載入 | 跟原版小麥注音完全一樣 |
| encoder 載入，decoder 沒載入或「decoder」開關（`slothEDecoderEnabled`）關掉 | 只有 in-walk（encoder-only） |
| 兩個都在 | in-walk ＋ decoder 閘門 |

兩個開關預設都開（`Source/Preferences.swift:487`、`:530`）。

### 每個按鍵的時間線

主執行緒先立刻顯示一個「暫定」結果，真正的模型計算放到背景的序列佇列，算完再回主執行緒套用。

| 步驟 | 執行緒 | 做什麼 | 位置 |
|---|---|---|---|
| 1 | 主 | `beginKeystroke`；中止進行中的停頓複查和修正項工作 | `Source/KeyHandler.mm:500`、`:3470` |
| 2 | 主 | **settle**：組字中、按的是會送出或開選字窗的鍵（Enter，以及沒有未完成注音時的 Space、Tab、↓、選字鍵、Control 組合）時，最多等 `commitWaitMilliseconds`（30 ms）拿這一世代的最終結果；其他非注音鍵只拿已經到的結果，不等；注音鍵直接跳過 | `Source/KeyHandler.mm:515`、`:4029`、`:4044`，`Source/SlothE/SlothERuntime.mm:368` |
| 3 | 主 | 原版處理按鍵。組出一個音節時 `insertReading` → `_walk`：先跑原版 `walk()`，世代加一 | `Source/KeyHandler.mm:2814` |
| 4 | 主 | **暫定重選**（provisional re-pick）：拿上一輪的 encoder 結果，只保留跟這次讀音相同的前綴位置，後面的位置當成未知；前綴裡的 decoder pin 一併帶入；在主執行緒跑一次 `rescoredWalk` 並顯示。按鍵處理中只記下「要送出工作」 | `Source/KeyHandler.mm:3315`，`Source/SlothE/SlothEPipeline.cpp:328` |
| 5 | 主 | 按鍵處理結束後送出一個 pipeline 工作（用 decoder 時附上網格的深複本）；新 buffer 的第一個鍵，模型閒置超過 1 s 時另送一次 prewarm；排停頓計時器 | `Source/KeyHandler.mm:536`–`:555`、`:3357`，`Source/SlothE/SlothERuntime.mm:1837` |
| 6 | 背景 | `RunPipeline`：① encoder（有快取就不跑模型）→ 發布 encoder 階段（decoder 沒開時到這裡就結束）→ ② 在網格複本上跑 `rescoredWalk` 取得節點細節 → ③ 建 decoder request → ④ 由左到右逐一打分、決定 pin → 發布最終結果 | `Source/SlothE/SlothEPipeline.cpp:89` |
| 7 | 主 | `_slothEDidFinishPipeline`：結果仍是這一世代的話，用 encoder 結果加 pin 在真正的網格上再跑一次 `rescoredWalk`；文字有變才更新組字區 | `Source/KeyHandler.mm:3415`、`:3296` |

佇列（`Source/SlothE/SlothERuntime.mm:364`–`:367`）：

- **compute queue**：序列，QoS user-interactive，所有模型呼叫都在這裡。`submitComputeJob` 最多讓一個工作排隊，新的工作取代還沒開始的舊工作（舊工作連同它持有的網格複本一起釋放），正在跑的不受影響（`Source/SlothE/SlothERuntime.mm:536`）。修正選字窗、停頓複查、修正項用 `submitSideComputeJob`，排在後面、不被取代。
- **load queue**：序列，QoS utility，載入模型、lazy 函式、eager pass。
- **log queue**：診斷記錄。

### 世代與過期

- `_slothGeneration` 在每次 `_walk` 和 `clear` 時加一（`Source/KeyHandler.mm:2819`、`:441`）。送出工作時把 `PipelineSlot::latestRequested` 設成目前世代（`:3366`）。
- 背景工作在這些時間點檢查 `latestRequested != generation`，過期就丟掉，不發布：開始前（`skipped`）、每個 request 之前、每一步裁切之間、每次 decoder 呼叫之後、發布前（`Source/SlothE/SlothEPipeline.cpp:97`–`:198`）。過期時 slot 裡留著 encoder 階段的結果。
- 主執行緒收到結果時，再比一次世代和讀音序列。結果記為 `stale`、`consumed`（commit 或 settle 已經套用過）、`deferred`（現在不是單純組字狀態，例如開著選字窗）、`applied` 或 `unchanged`（`Source/KeyHandler.mm:3424`–`:3451`）。

---

## 模型

### Encoder

SlothE-T 25M。「三元權重、雙向」來自 `DEVELOPMENT.md`；程式碼只把它當成「整串音節進、每個位置的 logits 出」的函式。

**輸入**（`Source/SlothE/SlothEEngine.cpp:312`、`:644`，`Source/SlothE/SlothECoreML.mm:102`–`:173`）：

1. 每個讀音先正規化（`NormalizeSyllable`，`SlothEEngine.cpp:204`）：去掉前後空白、NFC、刪掉 U+02C9（一聲符號）、把中點類符號換成 U+02D9（輕聲），開頭的輕聲移到最後。
2. 查 `syl_vocab.tsv`（1,539 個音節）：完全相符 → 去掉聲調再查 → 都沒有就是 `<unk>`。
3. 音節 id 在 CPU 上查 fp16 嵌入表 `enc25m_embed_f16.bin`（1,539 × 352），填成 `[1, L, 352]` 的輸入和 `[1, L]` 的 mask；多出的位置補 0、mask 0。
4. 前面可以接上**帶前文**的音節（見[帶前文](#帶前文)）。

**長度函式**：`enc25m.mlmodelc` 是 multifunction 模型，有 L8、L16、L32、L64、L256 五個函式（沒有 L128）。每次用「長度 ≥ T 的最小函式」，T 包含帶前文的音節（`SlothECoreML.mm:105`、`:125`）。整串超過 256 個音節時 `forward()` 回傳 null，該 buffer 不用模型（`SlothEEngine.cpp:649`）。

**輸出**：每個位置對 8,342 個字 id 的 logits（fp16，轉成 float）。合法字由 `syl2legal.bin` 的位元遮罩決定；log-prob 是在合法字上做 log-softmax（`SlothEEngine.cpp:433`、`:456`）：

$$
\log p(c \mid t) = \text{logit}_t(c) - \log\sum_{c' \in \text{legal}(s_t)} e^{\text{logit}_t(c')}
$$

不合法的字是 $-\infty$。

**UNK 分攤**（`SlothEEngine.h:30`–`:35`，`SlothEEngine.cpp:303`、`:470`）：`char2id.tsv` 有 18,756 個字，其中 12,276 個罕用字共用 id 2（UNK）。給這種字打分時，用 UNK 的 log-prob 再減掉 $\log 12276 \approx 9.4155$：

$$
\text{CharLogProb}(c) = \log p(\text{UNK} \mid t) - \log N_{\text{UNK}}
$$

$N_{\text{UNK}}$ 從載入的 `char2id` 數；跟預期的 12,276 不同時只寫一行 log（`SlothERuntime.mm:941`）。

**快取**：`Engine` 保留最近 8 次結果（LRU），key 是帶前文音節（以 `\x1e` 分隔）加上讀音（以 `\x1f` 分隔）。帶前文只用來算，結果只保留 buffer 本身的位置，並記下 `contextLength`（`SlothEEngine.cpp:572`、`:689`）。修正選字窗的單音節試算用 `store = false`，不擠掉目前 buffer 的快取。

**暫定檢視**：`ForwardResult` 可以是另一個結果的 view：位置 < `knownPrefix` 讀 base 的分數，之後的位置未知（`LogProb` 回傳 0，in-walk 直接略過）（`SlothEEngine.h:107`、`SlothEEngine.cpp:458`）。

### Decoder

`pred_q35_60m`（Qwen3.5 架構，60M，見 `SlothEDecoder.h:1`），`dec60m.mlmodelc`。

**Tokenizer**（`Source/SlothE/SlothEDecoder.cpp:118`–`:309`）：模型自己的 byte-level BPE（`dec_tokenizer.json`）。先切出 special token（最長優先），其餘用 GPT-2 的 pre-tokenize 規則切段，每個 byte 換成 GPT-2 的 byte 符號，再依 merge 的 rank 合併。找不到的符號用 `<unk>`。

**打分函式 t16 / t32 / t64 / t96**（`SlothEDecoder.cpp:329`、`:342`，`SlothECoreML.mm:178`、`:485`）：

- 一個 request 是（左文, 最多 3 個字串）。每個字串變成 `<bos>` ＋ tokenize(左文最後 64 字 ＋ 字串)。
- 三條序列一次送進模型（batch 3，用 `<pad>` 補齊），用「長度 ≥ 最長序列的最小函式」。
- 分數是整條序列逐 token 的 log-prob 總和：

$$
\text{dec}(x) = \sum_{k=1}^{n-1} \log P(\text{tok}_k \mid \text{tok}_0 \ldots \text{tok}_{k-1}),\quad \text{tok}_0 = \langle\text{bos}\rangle
$$

  這個總和也包含左文的 token。候選字共用同一段左文，所以這部分在候選之間大致相同〔推論：BPE 可能跨過左文和候選字的交界合併，那時就不完全相同〕。
- 最長序列超過 96 個 token、少於 2 個 token，或候選超過 3 個，就不打分（`score()` 回傳 false）。那個函式還沒載入時也不打分，並排一個背景載入（`SlothEDecoder.cpp:356`–`:369`）。
- 程式碼裡沒有分塊（chunking）：batch 固定是 3，閘門最多取 3 個候選（`topK = 3`）；太長的情況靠下面的[裁切](#超過長度時)處理。

**下一個 token 函式 n16 / n32 / n64 / n96**（`SlothEDecoder.cpp:391`，`SlothECoreML.mm:303`–`:350`）：batch 1；輸入是 `<bos>` ＋ tokenize(左文最後 64 字)，右邊補 0，另外傳最後一個 id 的位置 `last`；輸出是下一個 token 在整個詞表上的 log-softmax。只在打錯救回時用，一律 lazy 載入。

**字首表**：`dec_next_first.tsv`（3,766 行）列出「以某字開頭的 token」，用來把下一個 token 的機率換成「下一段文字以字 c 開頭」的機率（`SlothEDecoder.cpp:458`、`:471`）：

$$
D(c) = \log \max\Big(\sum_{\text{tok} \text{ 以 } c \text{ 開頭}} e^{\,lp(\text{tok})},\ 10^{-12}\Big)
$$

---

## 斷詞時的重新打分

### 邊的分數

參數是 config A'（`Source/SlothE/SlothEWalk.h:31`–`:36`，`SlothERuntime.mm:524`）：β = 0.3、γ = 0、懲罰 = −15、異體字 guard 開。

一條邊是（節點, unigram）。沒被使用者改過的節點，每個 unigram $u$ 的權重是（`SlothEWalk.cpp:76`、`:182`）：

$$
w(u) = \text{LM}(u) + \beta \cdot \text{neural}(u) + \gamma
$$

- $\text{LM}(u)$ 是語言模型的 unigram 分數（data.txt 的 log10 值，見 algorithm.md），$\text{neural}(u)$ 是 encoder 的自然對數 log-prob 總和（下一節）。兩者直接相加；β 0.3 是在這個混合尺度上掃出來的（`SlothE/eval-2026-10/README.md` 的 β 掃描）。
- 使用者（或使用者選字記憶）改過的節點只有一條邊：目前的值，分數用引擎的 override 分數（`kOverridingScore` = 42），沒有 neural 項，跟原版一樣（`SlothEWalk.cpp:170`）。
- decoder pin 指定的節點也只留被指定的 unigram，分數同樣是 `kOverridingScore`（`SlothEWalk.cpp:71`）。

`WalkEdges` 是 DAG 上的 Viterbi（`SlothEWalk.cpp:52`）：每個節點先取權重最高的 unigram，再做鬆弛。位置依序走、跨度由短到長、兩處都用嚴格的 `>`，所以 β = 0 時跟原版 `walk()` 的結果相同（`SlothEWalk.h:4`–`:14`）。

### 多字節點的 neural 分數

`InWalkNeuralScore`（`SlothEWalk.cpp:108`）：值 $v$ 放在位置 $s \ldots s+\ell-1$：

$$
\text{neural}(v) =
\begin{cases}
-15 \cdot \ell & \text{若 } v \text{ 的字數} \ne \ell \text{（不是一音一字）} \\
\sum_{k=0}^{\ell-1} f\big(\text{CharLogProb}(v_k \mid s+k)\big) & \text{否則}
\end{cases}
$$

其中 $f(x) = x$（$x$ 有限時），否則 $f(x) = -15$：字不在 `char2id`，或不是該音節的合法字。UNK 字套用上面的分攤。暫定檢視裡未知的位置不計分（加 0）。

選字窗用的 `ScoreCandidate` 是同樣的總和，但不能打分的字算 −30，不是 −15（`SlothEEngine.h:29`、`SlothEEngine.cpp:476`）。

### 異體字 guard 與特例

**異體字 guard**（`SlothEWalk.cpp:257`–`:305`）：另外跑一次 β = 0、沒有 pin 的 walk 當基準。in-walk 選的值如果是一音一字，而且跟基準只差在異體字（`variants.tsv` 的 80 組，例如 台／臺），就改回基準的寫法：節點有那個 unigram 就用它；沒有的話，基準的節點剛好鋪滿同一段時改用基準節點；都不行就保留（記為 unrepresentable）。

**特例**：

- 只有 1 個讀音、而且沒有帶前文時，`rescoredWalk` 直接回傳原版 walk，不產生 decoder request（`SlothEWalk.cpp:152`）。
- encoder 結果的長度跟讀音數不同時，也回傳原版 walk。
- 停頓複查留下的 sticky pin，只在它的節點仍在「不套 pin 的 in-walk 路徑」上時才套用；為此多跑一次 Viterbi（`SlothEWalk.cpp:196`–`:212`）。

---

## Decoder 閘門

預設參數（`SlothEWalk.h:87`–`:104`，`SlothERuntime.mm:497`）：λ = 1、τ = 0.5、topK = 3、`fullBuffer` = true、右文上限 `suffixCapChars` = 24。隱藏設定 `SlothEDecoderFullBuffer` = NO 會改用 `DecoderParams::V2()`（λ 2、只打分候選字本身）；`SlothEDecoderSuffixCapChars` 設右文上限。

### 產生 request 的節點

`BuildDecoderRequests`（`SlothEWalk.cpp:464`）走 in-walk 的節點：

- 節點的 **aligned 候選** = 它的 unigram 裡一音一字的值，去重，照 unigram 順序，各帶 neural 分數（`SlothEWalk.cpp:318`–`:327`）。
- 使用者沒改過、aligned 候選 ≥ 2 的節點才產生 request。
- 候選依 neural 分數由高到低排（穩定排序），取前 3 個；`enc` 就是這些 neural 分數。
- 也記下節點的 `shown`（guard 之後顯示的值）和 `raw`（in-walk 選的 unigram）。

### 整句打分的字串

`BuildFullBufferInput`（`Source/SlothE/SlothEPipeline.cpp:206`）。request 由左到右處理，`pinned[j]` 是這一輪已經替節點 j 決定的值：

- **左文** = 帶前文的文字 ＋ 節點左邊每個節點的文字（已決定 pin 的用 pin 的值，否則用 shown），取最後 64 字。
- **打分字串** = 候選字 ＋ 右文；右文 = 節點右邊每個節點的文字（同樣套用已決定的 pin），取前 24 字。
- `request.values` 保持原本的候選字：`DecideDecoderPin` 要拿它比對 shown、raw 和異體字，pin 的值也必須是網格裡的 unigram（`SlothEPipeline.h:93`–`:110`）。

### 混合與決策

`DecideDecoderPin`（`SlothEWalk.cpp:496`）：

1. 每個 decoder 分數都必須有限而且 < 0，否則不決定。
2. $b_k = \text{enc}_k + \lambda \cdot \text{dec}_k$，取最大的 $k^*$（平手時取前面的，也就是 encoder 分數較高的）。
3. $p = 1 / \sum_k e^{\,b_k - b_{k^*}}$，也就是 softmax 之後 $k^*$ 的機率。
4. 以下情況不 pin：$v_{k^*}$ 就是 shown 或 raw；$p < \tau$；$v_{k^*}$ 跟 shown 只差在異體字。
5. 其餘情況產生 pin（start, len, $v_{k^*}$）。整輪結束後，主執行緒用所有 pin 重跑 in-walk。

### 超過長度時

`fits` 檢查每條序列都 ≤ 96 個 token（`Decoder::largestLength()`）（`SlothEPipeline.cpp:137`、`:228`–`:259`）：

1. 太長時從左邊逐字刪左文，候選字和右文保持完整。每刪一個字就要重新 tokenize，所以每一步都會檢查有沒有新的按鍵，有就立刻停。
2. 左文刪光還是太長（而且有右文），就**退回只打分候選字本身**（左文照同樣方式裁切），記入 `decoderFallbacks`，不會跳過這個節點。
3. 連候選字本身都放不下，`Decoder::score` 回傳 false，這個節點沒有決定（`decoderNoDecision`）。需要的函式還沒載入時也一樣，但會排一個背景載入。

### 分數快取

`DecoderScoreCache`（`SlothEPipeline.h:30`，`SlothEPipeline.cpp:23`）：thread-safe 的 LRU，1,024 筆，key = 左文 ＋ 每個打分字串（以 `\x1f` 分隔）。整句打分時，key 裡是打分字串（候選字＋右文），不是候選字本身。

### 舊的逐鍵路徑與停頓複查

| | v3.4 逐鍵閘門（預設） | 舊逐鍵閘門 `fullBuffer = false` | 停頓複查（v3.2a） |
|---|---|---|---|
| 什麼時候跑 | 每個 pipeline 工作 | 每個 pipeline 工作 | 最後一個鍵之後 150 ms |
| λ / τ / topK | 1 / 0.5 / 3 | 2 / 0.5 / 3 | 2 / 0.5 / 3（`V2()`） |
| 左文 | 帶前文＋左邊文字，套用這一輪的 pin | 帶前文＋改字前的 shown 文字 | 同 v3.4 |
| 打分字串 | 候選字＋右文（≤ 24 字） | 候選字本身 | 候選字＋右邊**全部**的 shown 文字 |
| 太長時 | 裁左文 → 退回只打分候選字 | 不決定 | 不決定 |
| 位置 | `SlothEPipeline.cpp:164` | `SlothEPipeline.cpp:132`、`:169` | `SlothEPipeline.cpp:262`，`KeyHandler.mm:3574` |

停頓複查的細節（`Source/KeyHandler.mm:3454`–`:3667`）：

- 預設**關**：沒有設定值時，預設值是「整句打分關掉時才開」（`Source/Preferences.swift:538`–`:566`）。
- 它的 pin 標為 sticky，涵蓋整個 buffer。之後逐鍵結果套用時：跟停頓時讀音相同的前綴用停頓的 pin，之後的部分用逐鍵的 pin（`_slothEMergeStickyPins`，`:3524`）。
- 在某個詞上選字、或換掉某個讀音時，從那個位置起的 sticky pin 作廢（`:3497`、`:3508`）。
- 任何按鍵都會中止進行中的停頓複查（`:3470`）。

---

## 選字窗排序

`_slothECandidateOrder`（`Source/KeyHandler.mm:4080`）。開選字窗之前會先 settle（最多等 30 ms）；選字窗只讀快取的 encoder 結果（`Engine::lookup`），**不在主執行緒跑模型**。沒有快取結果，或節點的讀音有對不到 `syl_vocab` 的（標點、符號），就用原版順序。

1. **aligned 區塊排序**（`SlothEWalk.cpp:580`，`SlothEEngine.cpp:492`）：跟游標所在節點同讀音、同字數的候選是 aligned，分數用 `ScoreCandidate`；只有 1 個讀音的 buffer 改用 LM 分數（`unigramScore`）。aligned 的依分數由高到低穩定排序，其他候選照原版順序接在後面。異體字 guard：比 walk 的值排得高的異體字，移到 walk 的值後面。
2. **目前的值往後放**（`DemoteWalkValue`，`SlothEWalk.cpp:533`）：偏好 `slothEDemoteShownCandidate` 預設開（`Preferences.swift:519`）。aligned 區塊裡，跟畫面上的值相同的候選，連同它的異體字，移到區塊尾端。
3. **插隊**（`JumpAheadOrder`，`SlothEWalk.cpp:683`；`pinnedEntries`，`:602`）：
   - 對每個候選，在網格複本上模擬「使用者選了它」，重跑 in-walk（不套 decoder pin）。總分 $A$ = 所有節點的 LM ＋ β·neural，被選的節點用它真正的 LM 分數，不用 override 分數。
   - $S^*$ = 同一跨度上 $A$ 最高、而且不是目前的值或它的異體字的候選。
   - $X$ = 其他跨度上 $A$ 最高、會改變該跨度的顯示文字、而且整句結果跟目前畫面不同的候選。
   - $X$ 存在，而且 $S^*$ 不存在或 $A(X) - A(S^*) \ge \delta$（δ = 3.0 nats）時，$X$ 移到第一名（`SlothEWalk.h:135`）。
   - 模擬超過 16 ms（`kJumpAheadBudgetMs`，用目前耗時和推估的總耗時判斷）就放棄，維持步驟 1–2 的順序（`SlothEWalk.cpp:653`–`:661`）。
4. **修正項**（v3.2b，預設關）：見[合法但打錯的修正項](#合法但打錯的修正項)。

**只打一個音節時**：步驟 1 用 LM 分數，所以 aligned 區塊是詞頻順序（有帶前文時也一樣）；所有候選都在同一個跨度上，沒有 $X$，不會插隊；但步驟 2 仍然會把畫面上的字移到區塊尾端（`McBopomofoTests/SlothEInWalkTests.mm:360`–`:388` 驗證的就是這個順序）。

---

## 帶前文

v3.1a，`CarriedContext`（`Source/SlothE/SlothEPipeline.h:186`–`:228`，`SlothEPipeline.cpp:351`–`:450`）。只存在記憶體，不寫檔、不寫進記錄。

**存了什麼**：最近送出的字，每字一筆（讀音, 字），最多 64 字（`kMaxChars`），以及最後一次送出的時間。

- buffer 送出時，`clear` 先把顯示中的 walk 轉成（讀音, 字）配對：一音一字、而且每個讀音都是注音的節點，每字配一個讀音；其他節點的字沒有讀音（`CommittedChars`）。實際送出的文字跟 walk 文字不同時，全部當成沒有讀音的字（`Source/KeyHandler.mm:431`、`:3075`）。
- 空 buffer 時直接送給 app 的可列印鍵（例如 Space），只在之後能用游標位置驗證時才記入；否則不記入，並標記前文「有缺口」（`KeyHandler.mm:3214`）。

**模型看到什麼**：

- **encoder**：從最後一個沒有讀音的字之後算起的音節，最多最新的 24 個（`kMaxSyllables`）；再依 `Engine::usableContext` 只取放得進 `min(256, 最長函式) − buffer 長度` 的最新那些，接在 buffer 前面（`SlothEEngine.cpp:599`）。
- **decoder**：全部的前文文字（≤ 64 字），接在每個 request 的左文前面，最後再取左文最後 64 字。
- 只有 1 個讀音的 buffer，帶前文時也走 in-walk（`SlothEWalk.cpp:152`）。

**清除的時機**：

| 原因（記錄裡的字） | 條件 | 位置 |
|---|---|---|
| `idle` | 新 buffer 的第一個鍵，距上次送出超過 120 s | `SlothEPipeline.cpp:385` |
| `moved` | app 回報的游標位置跟預期不同 | 同上；`KeyHandler.mm:3138` |
| `no_selection` | app 不提供游標位置，而且距上次送出超過 15 s | 同上 |
| `key` | 空 buffer 時按 Esc、Tab、方向鍵、Home／End／PageUp／PageDown，或任何 Command／Control／Option 組合 | `KeyHandler.mm:3176` |
| `unverified_backspace` | 空 buffer 時按 Backspace，而前文有缺口 | 同上 |
| `activate` / `deactivate` / `commit_composition` | 輸入法啟用、停用、系統要求送出組字 | `Source/InputMethodController.swift:311`、`:326`、`:357` |

- 空 buffer 時按 Backspace（前文沒有缺口）只刪掉最後一字；按 Return 保留前文，但標記有缺口。
- 讀 app 游標位置超過 20 ms、回傳 NSNotFound 或不合理的值時，這個 app 在這個 process 裡不再問，改用 15 s 規則（`KeyHandler.mm:3105`、`:3146`–`:3156`）。
- 每次清除和每次送出，也會丟掉暫定重選用的上一輪結果（`_slothEDropProvisionalBase`）。

---

## 打錯救回

### 不合法讀音的修正選字窗

v3.1b（`Source/SlothE/SlothETypo.h:1`–`:16`，`Source/SlothE/SlothETypo.cpp`）。

**觸發**：偏好 `keepReadingUponCompositionError` 預設開（`Preferences.swift:349`）。組出一個語言模型沒有的讀音時，讀音留在組字區，同時在背景開始打分（`KeyHandler.mm:784`）。之後繼續打字會覆蓋同一格（這是原版 `BopomofoReadingBuffer` 的行為）。在這個讀音上按 Space 或 ↓，就開修正選字窗（`KeyHandler.mm:722`、`:2963`）。

**候選**（`TypoCandidates`，`SlothETypo.cpp:143`），以標準（大千）鍵盤為準：

1. **槽位編輯**：聲母、介音、韻母三個槽位（C/M/V）最多改一個，聲調任意；語言模型有這個讀音才算。
2. **鄰鍵替換**：把打出來的讀音換回大千的按鍵序列（一聲補一個 Space），每個鍵換成美式鍵盤上的實體鄰鍵（左右、上排兩個、下排兩個；最下排 x 到「,」和 Space 互為鄰鍵），再模擬按鍵；剛好組出一個合法讀音才算（`SlothETypo.cpp:33`–`:92`、`:166`–`:178`）。
3. 保留「C/M/V 差 ＋ 聲調是否改變 ≤ 1」或來自鄰鍵的候選（slot1_nb）。每個候選記下：`lm` = 最高 unigram 分數；`chars` = unigram 值；`dslots` = 非聲調符號數的差；`addFinal` = 打的讀音沒有韻母、候選剛好多一個韻母。

**分數**（`SlothETypo.cpp:201`–`:298`）。$[\cdot]$ 是 0/1，「兩處編輯」= C/M/V 差 ＋ 聲調改變 ≥ 2：

| 規則 | 公式 | 用在 |
|---|---|---|
| LM＋prior (i') | $\text{lm} - [\text{兩處編輯}] + [\text{聲調改變}] + 0.5\,[\text{鄰鍵}] + 0.5\,\text{dslots}$ | 沒有 encoder 結果（50 ms 內沒算完、或 encoder 失敗） |
| T1 (iii) | $2\,\text{lm} + E - [\text{兩處編輯}] + [\text{聲調改變}] + [\text{鄰鍵}] + \text{dslots}$ | 左文是空的（再加 $3\,[\text{addFinal}]$），或沒有 decoder 分數（再加 $1\,[\text{addFinal}]$） |
| D＋E | 每個（音節, 字）：$D(\text{字}) + \text{T1}(\text{音節}) + 3\,[\text{addFinal}]$ | 其他情況 |

- $E$（`TypoEncoderScore`，`:216`）：encoder 跑一次「帶前文 ＋ buffer 裡游標左邊的音節 ＋ 候選音節」（每個候選一次，不進快取），取候選位置在**整個字表**上的 log-softmax（不套合法遮罩），候選各字裡最大的那個；UNK 和查不到的字不算；一個都沒有時是 −10⁴。
- $D$：一次 n 函式呼叫，左文 = 帶前文 ＋ 游標左邊顯示的文字；用字首表換成 $D(c)$。
- 規則 i'、T1 先排音節（同分依讀音），再依序列出每個音節的字；D＋E 直接排（音節, 字）。字去重，最多 27 個（三頁）（`SlothETypo.h:33`）。

**時序**：工作在讀音一出現就開始；開窗時最多等 50 ms（`KeyHandler.mm:262`、`:2988`），沒好就用 LM＋prior 立刻列出。開著的窗不會更新；同一個讀音、同樣的前文再開一次，就用算好的分數。讀音改變或 buffer 清掉時取消工作；取消在下一次模型呼叫前生效。選了某一項 = 插入那個讀音，並當成使用者選了那個字（`KeyHandler.mm:3024`）。

### 合法但打錯的修正項

v3.2b，偏好 `slothETypoEntryEnabled` **預設關**（`Preferences.swift:574`–`:583`）。打成另一個合法讀音時，選字窗第一頁的最後一格提供修正（`SlothETypo.h:91`–`:106`，`SlothETypo.cpp:376`–`:531`）。

對位置 $p$ 的讀音，候選 $c$ 跟上一節一樣是 slot1_nb：

$$
E(c) = \text{total}(\text{buffer，第 } p \text{ 格換成 } c) - \text{total}(\text{buffer})
$$

total 是 in-walk 路徑上所有節點的 LM ＋ β·neural（`InWalkTotal`，`SlothEWalk.cpp:447`）。每個候選都在網格複本上算（只重建涵蓋 $p$ 的跨度，清掉使用者選字），encoder 也各跑一次整個 buffer。

$$
D(c) = \text{score}(c) - \text{score}(\text{打的讀音}),\quad \text{score}(x) = \log\max\Big(\sum_{\text{x 的單字}} \sum_{\text{tok}} e^{\,lp(\text{tok})},\ 10^{-12}\Big)
$$

左文是帶前文 ＋ 位置 $p$ 左邊顯示的文字。

$$
\text{gain}(c) = E(c) + \log w(c) + 0.25 \cdot D(c)
$$

- $\log w$：同一排左右鄰鍵 0，其他（上下排、或不是鄰鍵）$\log 0.5$（`SlothETypo.cpp:376`、`:380`）。
- gain 最高的候選（同分取讀音較前的）在 gain ≥ θ = 1.765735609807406 時提供（`SlothETypo.h:105`–`:106`）。
- 標籤是「字 注音」，字是換讀音之後 in-walk 在那一格顯示的字。
- **觸發**：停頓 150 ms 後、方向鍵把游標移到別的音節時、或開選字窗時（開窗不等：分數沒好就這次不放，只先開始算）。
- 放在第一頁最後一格（index = 頁大小 − 1），候選不足一頁時接在最後（`KeyHandler.mm:3834`）。選了它就換掉那個讀音（`KeyHandler.mm:3883`）。

---

## 「再」雙胞胎詞

v3.4，在建置詞庫時做，不是執行時（`Source/Data/curation/compilers/postprocess.py:132`–`:189`，`Source/Data/Postprocess.txt:94` 的 `add-twins zai-twins.txt`）。

- **問題**：data.txt 有「會在」「就在」「你在」這類詞，卻沒有對應的「會再」。walk 把「在」鎖在詞裡，那個節點只有一個 aligned 候選，不會產生 decoder request，encoder 再怎麼偏好「再」也贏不了（`SlothE/eval-2026-10/PLAN.md:16`）。
- **規則**：對 `Source/Data/zai-twins.txt` 的每個前綴，data.txt 裡有「前綴＋在」（讀音以 ㄗㄞˋ 結尾、一音一字）而沒有「前綴＋再」時，在同一個讀音下補一條「前綴＋再」，分數 = 「在」詞的分數 − 0.1（log10），插在分數順序的位置。所以預設仍然是「在」，由模型決定。
- **18 個前綴**：不會、他、她、你、還能、還會、會、就、將、先、只能、只好、只會、最好、才會、也會、也許、應該。只加「再」當副詞合文法的前綴；不加「在」本身是詞的一部分的（實在、現在、存在…）。
- 清單裡的前綴找不到「前綴＋在」時，建置只出警告。

補了詞之後，這些節點有兩個 aligned 候選（「X在」「X再」），in-walk 和 decoder 閘門都有機會選「再」。

---

## 載入與 Neural Engine

兩個模型**只跑在 ANE 上**（compute units = CPU_AND_NE），沒有 CPU 版本（`Source/SlothE/SlothECoreML.h:1`–`:5`）。需要 macOS 15（multifunction 模型）。

載入順序（`Source/SlothE/SlothERuntime.mm:852`、`:974`）：準備模型副本 → 處理清快取標記 → 核對 encoder 檔案 → 載入 encoder 全部 5 個函式並檢查放置 → `Engine::load`（詞表、遮罩、異體字表；檢查模型和詞表大小相符；拿 `{1, 1}` 試跑一次，logits 必須都有限）→ 清理舊副本 → 核對 decoder 檔案 → 載入 tokenizer 和 t16 並檢查放置 → 載入字首表（失敗只寫 log，修正選字窗改用 T1 規則）→ 試打分 `"今天"`、`"天氣"`（必須有限且 < 0）→ 開始 eager pass。

### 檔案核對

`runtime-manifest.txt` 每行是 `<sha256> <bytes> <路徑>`，包含編譯好的 `.mlmodelc` 目錄裡的每個檔案（`SlothEEngine.h:238`–`:283`，`SlothEEngine.cpp:779`–`:899`）。

- encoder 需要：`enc25m.mlmodelc`、`enc25m_embed_f16.bin`、`syl_vocab.tsv`、`char2id.tsv`、`syl2legal.bin`、`variants.tsv`；decoder 需要：`dec60m.mlmodelc`、`dec_tokenizer.json`、`dec_next_first.tsv`（`SlothEEngine.cpp:726`）。
- 每個檔案的大小和 sha256 都要相符；缺檔、清單沒列、大小或雜湊不符都算載入失敗（原因 `integrity`）。
- manifest 在啟動時讀一次，存成快照；之後每次核對（包括每次 lazy 載入之前）都用這個快照，不再讀檔，因為更新時 bundle 裡的 manifest 可能已經被換掉（`SlothEEngine.h:257`–`:260`，`SlothERuntime.mm:693`）。
- 副本一律拿 **bundle 的** manifest 核對（app 的簽章保護它），不用副本自己的。

### 模型副本

v3.4p（`SlothEEngine.h:285`–`:367`，`SlothEEngine.cpp:1042`–`:1318`，`SlothERuntime.mm:1561`–`:1794`）。

**為什麼要有副本**：Core ML／ANE 的編譯快取跟著**模型檔案**（路徑加檔案身分），不跟著執行檔或 app。實驗中換掉執行檔或整個 app，快取仍然命中；同樣的位元組重新複製到同一個路徑，就沒命中（`reports/ane-cache-keying.md`，在工作區）。所以把模型放在 app 外面一個更新不會碰的位置，模型沒變的更新就不必重新編譯。

- **位置**：`~/Library/Application Support/McBopomofoLM/Models/<set id>/`（dev bundle id 用 `McBopomofoLM.dev`，XCTest 用暫存資料夾）。set id 是 bundle 裡 `runtime-manifest.txt` 的 sha256 前 16 個 hex 字元（`SlothERuntime.mm:1565`）。
- **建立**（`EnsureModelStore`，`SlothEEngine.cpp:1063`）：在 `<root>/.lock` 的 flock 下進行。
  - 已有的副本每個檔案都相符：沿用，什麼都不動。
  - 損壞或不是資料夾：只有沒人持有它的 lease 時才換掉（先改名成 `.corrupt-*` 再刪）；有人在用就報錯，這次從 bundle 載入。
  - 新副本先放進 `.tmp-<pid>-<亂數>`：同一個 APFS volume 上用 `clonefile(2)`（幾乎不佔空間，並去掉 quarantine 屬性），否則一般複製；核對後用 `renamex_np(RENAME_EXCL)` 發布。別人先發布了就用別人的（核對過）。
  - 副本發布後**永不修改**，改了快取就失效。
- **lease**：每個從某個副本載入的 runtime，整個生命週期都對 `<root>/.leases/<set id>.lock` 持有共享 flock。
- **清理**（`CollectModelStoreGarbage`，`:1236`）：只有 runtime 在自己載入成功後做一次，`install` 不做。刪掉其他 set 和 `.tmp-*`／`.corrupt-*`，但保留：自己的 set、bundle **現在**指名的 set（重新讀檔）、有人持有 lease 的 set。只刪 lstat 看到是真資料夾、名稱符合的項目，不跟著符號連結走（`Models` 本身可以是符號連結，會照著走）。

### 放置檢查

Core ML 在 ANE 編譯失敗時會默默改用 CPU，所以載入後逐一檢查每個函式（`SlothECoreML.mm:48`–`:93`、`:532`–`:580`）：

1. **MLComputePlan**：不計 `const` 運算，算每個函式的預估成本有多少比例排在 Neural Engine 上。encoder 門檻 99%（實測 100%），decoder 80%（實測 85.6–94.9%，其餘是 token id 的 gather）。低於門檻 → `not_on_ane`；做不出 plan（300 s 逾時）→ `plan_failed`。
2. **延遲探測**：先喚醒一次，再跑 5 次取中位數。encoder 上限 10 ms；decoder 上限 $15 + (T - 16) \times 0.3125$ ms（t96 是 40 ms）（`SlothECoreML.mm:182`）。超過 → `probe_slow`。

啟動時 `probe_slow` 不算失敗：runtime 維持「載入中」，30 s 後再載入一次，還是太慢才算失敗（`SlothERuntime.mm:906`–`:930`、`:1005`–`:1022`）。

### lazy 函式與 eager pass

- **啟動時**只載 encoder 全部函式和 decoder 的 t16（`SlothERuntime.mm:660`–`:664`）。`install --compile-now` 和測試會載入全部 t 和 n 函式。
- **lazy 載入**（`SlothECoreML.mm:364`–`:400`、`:230`–`:262`）：request 需要還沒載入的函式時，這次不決定，並在 load queue 排一個載入。載入前先用 manifest 快照核對 decoder 檔案（`integrity`）；載入後做自己的放置檢查。`plan_failed` 重試一次；失敗的函式在這個 process 裡不再試，那些 request 維持 encoder-only。
- **eager pass**（v3.4n，`SlothERuntime.mm:1069`–`:1151`）：decoder 載入成功後，依需要的順序 t32、n16、t64、n32、t96、n64、n96 逐一編譯進 ANE 快取，**編完立刻釋放**，不做放置檢查，失敗也不標記失敗。目的是讓之後的 lazy 載入都是快取命中，而不是打字途中才編譯。每一步是 load queue 上獨立的 block，使用者觸發的 lazy 載入最多等一個函式。
- 「AI 選字」關掉時，eager pass 和延遲的重試都暫停，打開時從停下的地方繼續（`SlothERuntime.mm:1155`–`:1191`）。

### 清快取的標記與自我修復

編譯快取在 `~/Library/Caches/<bundle id>/com.apple.e5rt.e5bundlecache`。看起來像快取壞掉的情況：`plan_failed`，或 `not_on_ane` 而且 ANE 比例 < 1%（`SlothERuntime.mm:1538`）。

- **啟動時**：encoder 或 decoder 的載入出現這種失敗時，清快取（每個 process 最多一次）再載一次（`SlothERuntime.mm:894`、`:993`、`:1546`）。
- **自我修復標記**（`slothe-clear-ane-cache-at-next-start`）：lazy 載入的 t 函式重試後仍然這樣失敗時，由 runtime 寫入，請**下次啟動**清快取。每個 `CFBundleVersion` 最多清一次（`slothe-ane-cache-cleared-for-version`），避免一直不穩的 Mac 每次啟動都重新編譯 3 分鐘（`SlothERuntime.mm:706`–`:723`）。
- **安裝標記**（`slothe-clear-ane-cache-after-install`）：一定照做，沒有次數限制。`install` 只在副本是新建的、或建不起來時寫（模型沒變的重新安裝不寫，快取保留）（`Source/InstallCommand.swift:113`–`:139`）。副本建不起來、從 bundle 載入時，runtime 也寫一次（每個版本一次）（`SlothERuntime.mm:1748`）。
- **消費**（`SlothERuntime.mm:1298`–`:1474`）：在 `slothe-ane-cache.lock` 的 flock 下一次完成：用 rename 搶下標記 → 清快取 → 寫次數上限 → 刪掉搶下的檔。找到之前沒處理完的 `.claimed-*`，就當成還沒處理的要求。清不掉時搶下的檔留著，下次再處理。

### 退回的層次

| 狀況 | 行為 |
|---|---|
| encoder 不能用（檔案、載入、放置、探測任一失敗） | 原版小麥注音；decoder 標為 `no_encoder` |
| decoder 不能用 | encoder-only 的 in-walk |
| 某個 t 函式沒載入或失敗 | 需要它的 request 不決定（encoder-only），其他照常 |
| n 函式沒有 | 修正選字窗用 T1 規則；修正項等之後重試 |
| 副本建不起來 | 從 app 裡載入，並要求一次乾淨的重新編譯 |
| 背景結果 30 ms 內沒到（送出、開窗） | 用畫面上的結果；只有 encoder 階段到了就用它（`partial`） |

---

## 實際範例

讀音：ㄏㄨㄟˋ ㄗㄞˋ ㄐㄧㄚ ㄧˇ ㄊㄠˇ ㄌㄨㄣˋ（目標：會再加以討論）。這是 Common Voice dev 的 `cv08868`。以下的數字取自 repo 裡的 fixture 和測試記錄；decoder 分數沒有存下來，所以 decoder 那一步只描述它的條件，不給數字。

**1. 原版 walk**：`McBopomofoTests/SlothEFixtures/inwalk_parity.jsonl` 和 `v2_parity.jsonl` 記錄的斷詞是 會在｜加以｜討論，原版和補詞前的 in-walk 都是「會在加以討論」。

**2. encoder**：`v302_dev_walks.jsonl` 存的 25M encoder log-prob（v3.0.2 製作 fixture 時的快取；app 現在的 ANE 結果可能略有不同〔推論〕），每個位置的前兩名：

| 位置 | 讀音 | 第一 | 第二 |
|---|---|---|---|
| 0 | ㄏㄨㄟˋ | 會 −0.0016 | 慧 −7.93 |
| 1 | ㄗㄞˋ | **再 −0.0698** | 在 −2.7217 |
| 2 | ㄐㄧㄚ | 加 −0.0079 | 家 −5.38 |
| 3 | ㄧˇ | 以 −0.0086 | 乙 −5.50 |
| 4 | ㄊㄠˇ | 討 −0.0003 | 套 −8.45 |
| 5 | ㄌㄨㄣˋ | 論 −0.0001 | 侖 −10.22 |

encoder 在位置 1 偏好「再」約 2.65 nats，跟 `SlothE/eval-2026-10/PLAN.md:16` 寫的一致。

**3. 補詞前（v3.3 以前的詞庫）**：節點 [0, 2) 的 unigram 裡一音一字的只有「會在」，aligned 候選只有 1 個，不產生 decoder request（`v2_parity.jsonl` 這句的 `calls` 是空的）。[2, 4) 和 [4, 6) 也各只有「加以」「討論」一個（`rerank_parity.jsonl` 的 `span_candidates`）。所以 encoder 的偏好完全用不上，輸出「會在加以討論」。

**4. 補詞後的 in-walk**：「會再」的分數是「會在」的 −4.22800493 再減 0.1（`SlothE/eval-2026-10/tools/twins_added.txt`）。用上表的 encoder 分數算節點 [0, 2) 的兩條邊〔推論：用 fixture 數字手算〕：

$$
\begin{aligned}
w(\text{會在}) &= -4.2280 + 0.3 \times (-0.0016 - 2.7217) = -5.0450 \\
w(\text{會再}) &= -4.3280 + 0.3 \times (-0.0016 - 0.0698) = -4.3494
\end{aligned}
$$

「會再」高 0.70，in-walk 選「會再」。測試記錄 `SLOTHE_V34 會再加以討論 in-walk=會再加以討論`（`build/evidence/v34p3-test-full.log` 等，`McBopomofoTests/SlothEDecoderTests.mm:630`）證實了這個結果。「再」和「在」不是異體字（`variants.tsv` 沒有這組），guard 不介入。

**5. decoder 閘門**：只有節點 [0, 2) 有兩個 aligned 候選，產生一個 request：

- values = [會再, 會在]（依 neural 排序），enc = [−0.0715, −2.7233]；
- 左文 = 帶前文（測試裡是空的）＋ 左邊文字（空）；
- 打分字串 = 「會再加以討論」「會在加以討論」（右文「加以討論」4 字，沒到 24 字上限）；
- $b_k = \text{enc}_k + 1 \cdot \text{dec}_k$。

測試的最終結果是「會再加以討論」。shown 已經是「會再」，pin 只能改成別的值，所以結果表示 $b(\text{會再}) \ge b(\text{會在})$，也就是 $\text{dec}(\text{會在加以討論}) - \text{dec}(\text{會再加以討論}) \le 2.65$〔推論〕。

同一個 buffer 改用 `V2()` 閘門（只打分「會再」「會在」、左文空、λ 2）時，結果是「會在加以討論」（工作區的 `reports/v34-app.md:53`，以及同一行測試記錄的 `v2-gate=`）。兩個候選時，pin 成立等於 $b(\text{會在}) > b(\text{會再})$，即 $2 \times (\text{dec}(\text{會在}) - \text{dec}(\text{會再})) > 2.65$，decoder 只看兩個字時偏好「會在」超過約 1.33 nats〔推論〕。看到右邊的「加以討論」之後，這個偏好就不夠推翻 encoder 了。

**6. 逐鍵的過程**〔推論：依程式碼的流程描述，中間結果沒有記錄〕：

- 打完 ㄏㄨㄟˋ：只有 1 個讀音，沒有帶前文時顯示原版 walk。
- 打完 ㄗㄞˋ：開始有 in-walk；這時 [0, 2) 的 request 沒有右文，打分字串就是候選字本身（λ 仍是 1）。
- 之後每個音節：主執行緒先用上一輪的分數做暫定重選，前綴裡的 pin 帶過來，畫面不會跳回原版的選擇；背景再用更長的右文重新打分。v3.4 的鍵盤測試在逐鍵打完、不等停頓的情況下，組字區就是「會再加以討論」（`McBopomofoTests/SlothEV34Tests.swift:38`、`:72`–`:79`）。
- 按 Enter：settle 最多等 30 ms 拿最終結果再送出。送出後，6 個（讀音, 字）配對進入帶前文。

**7. 選字窗**（補詞前的資料，`v302_dev_units.jsonl`）：游標在 ㄗㄞˋ 上時，候選依原版順序是 在家、會在、在、再、載…。步驟 1 之後是 會在、在家、在、再…（aligned 區塊只有「會在」）。插隊計算的 $A$：會在 −12.86、在 −13.68、再 −13.79、在家 −15.84。節點跨度 [0, 2) 上除了「會在」沒有別的候選，所以沒有 $S^*$；「在」的模擬結果跟畫面相同，不算；$X$ 是「再」。沒有 $S^*$ 時 $X$ 不用比 δ，直接排第一：再、會在、在家、在、載…（fixture 的 `want`）。

---

## 程式碼與測試位置

### 原始碼

| 部分 | 檔案 | 主要函式／類別 |
|---|---|---|
| encoder、詞表、異體字、UNK、manifest、模型副本 | `Source/SlothE/SlothEEngine.{h,cpp}` | `Vocabulary`、`VariantTable`、`Engine::forward`、`ScoreCandidate`、`RerankOrder`、`VerifyRuntimeFiles`、`EnsureModelStore`、`CollectModelStoreGarbage` |
| in-walk、decoder 決策、選字窗 | `Source/SlothE/SlothEWalk.{h,cpp}` | `SlothEGrid::rescoredWalk`、`InWalkNeuralScore`、`BuildDecoderRequests`、`DecideDecoderPin`、`DemoteWalkValue`、`pinnedEntries`、`JumpAheadOrder`、`InWalkTotal` |
| 背景 pipeline、帶前文 | `Source/SlothE/SlothEPipeline.{h,cpp}` | `RunPipeline`、`BuildFullBufferInput`、`RunPauseGate`、`DecoderScoreCache`、`ProvisionalResult`、`CarriedContext`、`NewBufferResetReason` |
| decoder、tokenizer | `Source/SlothE/SlothEDecoder.{h,cpp}` | `BpeTokenizer`、`Decoder::score`、`Decoder::nextTokenLogProbs`、`firstCharLogProb` |
| 打錯救回 | `Source/SlothE/SlothETypo.{h,cpp}` | `TypoCandidates`、`ScoreTypoWindow`、`TypoWindowOrder`、`ScoreLegalTypo`、`DecideLegalTypo` |
| Core ML 後端、放置檢查 | `Source/SlothE/SlothECoreML.{h,mm}` | `LoadCoreMLEncoder`、`LoadCoreMLDecoder`、`CheckPlacement`、`PlanANEPercent` |
| runtime：佇列、載入、lazy、eager、標記、記錄 | `Source/SlothE/SlothERuntime.{h,mm}` | `SlothERuntime`（`submitComputeJob`、`_loadOnLoadQueue`、`_startEagerCompilePass`、`_consumeANECacheClearMarker`、`_prepareModelStore`） |
| key handler 的接點 | `Source/KeyHandler.mm:2814`–`:4134` | `_slothEUpdateForCurrentGrid`、`_slothESubmitPipelineFromKey`、`_slothESettleWaiting`、`_slothECandidateOrder`、`_slothEStartTypoJob`、`_slothEPauseTimerFired`、`_slothEStartLegalJobAt` |
| 偏好設定 | `Source/Preferences.swift:485`–`:600` | `slothERerankEnabled`、`slothEDecoderEnabled`、`slothEPauseGateEnabled`、`slothETypoEntryEnabled` |
| 安裝 | `Source/InstallCommand.swift` | `prepareModels` |
| 「再」雙胞胎詞 | `Source/Data/zai-twins.txt`、`Source/Data/curation/compilers/postprocess.py:132` | `add_twins` |
| 參考產生器、離線評估 | `SlothE/v2/`、`SlothE/eval-2026-10/` | v2 Core ML 參考產生器、v3.4 的評估 |

### 測試（`McBopomofoTests/`）

| 部分 | 主要測試類別 |
|---|---|
| 音節對應、重排序、異體字、執行環境 | `SlothETests`（`SlothETests.mm`） |
| in-walk、選字窗、插隊、帶前文、停頓複查、修正項規則（對 walk2 等離線研究的 parity） | `SlothEInWalkTests`（`SlothEInWalkTests.mm`） |
| tokenizer、decoder 分數、整句打分、分數快取、lazy 函式、manifest | `SlothEDecoderTests`、`SlothELazyDecoderTests`、`SlothEModelFileTests`、`SlothEMemoryTests`（`SlothEDecoderTests.mm`） |
| 背景結果、送出等待、暫定重選、放置檢查、安裝 | `SlothEPipelineTests`、`SlothEAsyncTests`、`SlothEKeyHandlerTests` |
| 帶前文、修正選字窗（key handler） | `SlothECarriedContextTests`、`SlothETypoWindowTests`（`SlothEKeyHandlerTests.swift`） |
| 停頓複查、修正項、英文模式 | `SlothEPauseGateTests`、`SlothETypoEntryTests`、`SlothEEnglishModeTests`（`SlothEV32Tests.swift`）、`SlothEV32ParityTests` |
| v3.4 整句打分和「再」 | `SlothEV34KeyHandlerTests`、`SlothEV34PreferenceTests`（`SlothEV34Tests.swift`）、`SlothEV34EvalTests`（選用） |
| 模型副本 | `SlothEModelStoreTests`、`V34pStoreRootTests`、`V34pInstallTests`、`V34pRuntimeStoreTests`、`V34pModelTests`、`V34pCacheHitTests`（預設跳過） |
| 安裝、標記、eager pass | `V34nInstallCommandTests`、`V34nMarkerTests`、`V34nModelTests`、`ANEColdStartTests`（預設跳過） |

載入模型、跑很久的測試清單見工作區的 `.claude/rules/mcbopomofo-lm.md`；只在動到載入程式時才需要跑。

---

**文件版本**：1.0
**最後更新**：2026-10-15
**適用版本**：McBopomofoLM v3.4p（`lm` 分支）
