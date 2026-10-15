# 2026-10 離線評估：在／再、0c D512、decoder 整句打分

日期：2026-10-13。評估對象是 v3.1.2（config A'：25M encoder 參與斷詞，β 0.3；pred_q35_60m decoder，λ 2，τ 0.5，取前 3 名）。

## 結論

1. **decoder 改成整句打分**：打分字串用「候選字＋後面的 buffer 文字」，前文要套用前面已決定的 pin，λ 1、τ 0.5。在新測試集上，聊天和新聞都比現行版高 **+0.34pt**，95% CI 都是 [+0.18, +0.50]。「再」的選對率從約 67% 提高到約 79%。
2. **補「再」雙胞胎詞**：在詞庫補上「<前綴>再」。這次加了 18 個，例如 會再、就再、你再、應該再。在 ㄗㄞˋ 的位置修好 28 個、改壞 1 個，p < 0.001；一般句子不受影響。
3. **D512（0c 的 `Funmula/0c-sequence-d512`）分兩種用法看：**
   - **放在 decoder 的位置：不導入。** 疊在現行 decoder 上，聊天 +0.15pt、新聞 +0.25pt，都顯著，但聊天沒達到事先定的 0.2pt 門檻；導入要多跑一個 130 MB 的 runtime（ONNX CPU 或轉 Core ML）。D512 單獨取代 decoder 時，在 dev 上不比現行版好。
   - **拿掉 encoder 和詞庫斷詞、只用 D512 解碼（事後追加）：** 聊天組跟設定 D 打平（96.44%），新聞組 97.53%，比設定 D 高 +1.14pt [+0.80, +1.48]。「再」選對率 84%／80%，但「在」被錯改成「再」也變多（聊天 34 → 48）。要上線需要 KV cache 加上 ANE 的增量解碼，還要保留使用者詞庫等功能。建議的下一步是「兩案並陳」混合版，詳見 [D512.md](D512.md) 第 4.6 節和第 5 節。
4. **斷詞零碎**（例如「那週是馬丁路德金連假」被打成「那州市馬丁路德金蓮價」）是詞庫的詞頻和收詞問題，例如「連假」−7.47、「廉價」−4.90，而且沒收「馬丁路德金」。換哪個 decoder 都修不好，要靠更新詞頻和使用者詞庫。

在 Mac 上要實作和驗證的項目，見最後一節「app 修改規格」。

## 方法

- **離線流程**（`tools/`）：
  - 斷詞用 repo 的 C++ 程式（gramambular2 + `SlothEWalk`）編成 `walk_cli`。
  - encoder 用 PyTorch 跑 HF 上的 fp32 master `slothe.pt`。
  - decoder 用 transformers 跑 `pred_q35_60m`。
  - D512 用 onnxruntime 跑。
- **對 repo fixture 的驗證**：

  | 項目 | 結果 |
  |---|---|
  | 用 fixture 的 encoder 分數時，stock walk 和最終 walk | 500/500 一致 |
  | 改用 PyTorch encoder 時，in-walk／最終輸出 | 499/500、496/500 一致（fp32 對 ANE fp16 的差異） |
  | decoder 決策 | 1103/1104 一致 |
- **沒模擬到的**：carried context（分段送出時帶前文）、typo window、使用者選字記憶。
- **dev 集**：repo 的 500 句 Common Voice dev（`McBopomofoTests/SlothEFixtures`），共 4,352 字。正確答案用 `tools/match_gold.py` 拿注音去比對 Common Voice zh-TW 原文找回來，500/500 都找到。這組只拿來調參數。
- **新測試集**（`testset/`）：2026-10-01 到 13 發布的文字，D512（訓練資料截止 2026-09-01）和 Sloth 模型都沒看過。
  - 新聞：公視、三立、ETtoday、風傳媒；中央社、自由、聯合、報導者、關鍵評論網的 robots.txt 禁止 AI 使用，沒有抓。
  - 聊天：PTT 22 個版；Dcard 被 Cloudflare 擋住。
  - 句子以子句為單位，6 到 32 個漢字，已排除 Common Voice 出現過的句子。
  - 注音用 g2pW 標註，再對齊到詞庫的合法讀音：一 統一標成ㄧ，不 統一標成ㄅㄨˋ，遇到詞庫裡的詞就用該詞的讀音。
  - 分成兩組：「一般」是不含「再」的句子，各隨機抽 1,500 句；「ㄗㄞˋ」是所有含「在」或「再」的句子。
  - 聊天組裡用錯「在／再」的 33 句已排除。另外人工抽查 120 句，沒有發現用錯的。
- **事先登記**：5 種設定和判定規則都先寫在 `PLAN.md`，測試集只跑一次。跑測試集之前，整個流程經過 3 輪審查，意見已併入 `PLAN.md`。

## 結果

### dev（500 句，`results/*.log`）

| 設定 | 字元正確率 | 整句全對 |
|---|---|---|
| 原版小麥注音 | 94.35% | 67.0% |
| 只有斷詞（不跑 decoder） | 96.05% | 75.2% |
| 現行（span，λ2，τ.5） | 96.35% | 76.8% |
| 現行＋依序前文 | 96.37% | 77.0% |
| D512 取代 decoder（span） | 96.09% | 75.2% |
| decoder 整句打分（λ1，τ.5） | 96.78% | 78.8% |
| decoder 整句打分＋0.25·D512 | 96.88% | 79.8% |

- β 掃描：β 0.3／0.5／0.7／1.0 加 decoder 後是 96.35／96.23／95.91／95.77%，維持 0.3。
- 整句打分在 λ 1 到 1.5、τ 0.4 到 0.6 之間結果都差不多。二折交叉驗證：整句打分 96.74%，加上 0.25·D512 是 96.92%。

### 新測試集：一般句子（`results/{chat,news}_analysis.txt`）

| 設定 | 聊天 字元／整句 | 新聞 字元／整句 |
|---|---|---|
| A 原版小麥注音 | 93.69% / 58.0% | 92.89% / 53.6% |
| B 現行 app | 96.10% / 71.1% | 96.05% / 70.2% |
| C 整句打分＋依序前文，λ1 τ.5 | 96.44% / 73.8% | 96.39% / 72.6% |
| D C＋雙胞胎詞 | 96.44% / 73.8% | 96.39% / 72.6% |
| E D＋0.25·D512 | 96.59% / 74.9% | 96.64% / 74.2% |

- 聊天組 1,497 句、16,366 字；新聞組 1,499 句、18,318 字。
- C 對 B：+0.34pt，95% CI [+0.18, +0.50]，兩組相同。聊天修好 113 字、改壞 57 字；新聞修好 131 字、改壞 69 字。
- E 對 D：聊天 +0.15pt [+0.07, +0.23]；新聞 +0.25pt [+0.17, +0.34]。

### 新測試集：在／再（ㄗㄞˋ 的所有位置）

| | 聊天「再」選對（1,353） | 新聞「再」選對（1,234） | 「在」錯成「再」（聊天／新聞） |
|---|---|---|---|
| A | 412 (30%) | 466 (38%) | 12 / 9 |
| B 現行 | 919 (68%) | 812 (66%) | 31 / 5 |
| C | 1,065 (79%) | 960 (78%) | 33 / 2 |
| D | 1,082 (80%) | 971 (79%) | 34 / 2 |
| E | 1,090 (81%) | 976 (79%) | 33 / 3 |

- McNemar 檢定，D 對 C：聊天修好 17、改壞 1（p = 1.5e-4）；新聞修好 11、改壞 0（p = 0.001）。
- 新聞組 D 剩下的 261 個「再→在」：
  - 189 個在單字節點，多半是子句開頭的「再…」。前面的「先…，」被子句切分切掉了；app 打字時有 carried context，表現應該會比這好。
  - 56 個被鎖在沒有「再」雙胞胎的詞裡，例如 也不在、也在、如果在、甚至在。
  - 16 個是詞庫裡有正確的詞，但沒選到。

### C 剩下的錯（一般句子）

| 原因 | 聊天 | 新聞 |
|---|---|---|
| 正確字在候選前三名，decoder 沒選 | 327 | 336 |
| 該處詞庫沒有正確的詞（收詞不足） | 133 | 139 |
| 正確字不在 encoder 前三名 | 85 | 152 |
| 詞庫有正確的詞，但詞頻導致斷詞切錯 | 37 | 34 |

## app 修改規格（已在 v3.4 實作，實作和 Mac 上的驗證見 `reports/v34-app.md`）

實作位置：`Source/SlothE/SlothEPipeline.cpp`（`RunPipeline`、`BuildFullBufferInput`）、`SlothEWalk.h`（`DecoderParams`）、`Source/Data/zai-twins.txt` 加 `Postprocess.txt` 的 `add-twins`。隱藏開關（`defaults write`，沒有選單）：`SlothEDecoderFullBuffer` = NO 回到 v3.3 的逐鍵 gate（λ 2、只打分候選字）；`SlothEDecoderSuffixCapChars` 設右邊文字的字數上限（預設 24）。停頓複查（v3.2a）預設改為關閉，開關保留。

1. **`SlothEPipeline` 的 decoder 階段**：
   - request 由左到右依序處理。每個 request 的 context = `decoderContext` + 套用前面 pin 之後的 buffer 左文。
   - 打分字串 = 候選字 + 套用 pin 之後的右文，右文最多取 `suffixCapChars` 個字（預設 24，可設定）。
   - `request.values` 保持原本的候選字，打分字串另外組。`DecideDecoderPin` 會比對 shown/raw 和異體字 guard，而且 pin 必須對得到 grid 裡的 unigram。
   - λ 1、τ 0.5。
   - 超過 t96 時，先從左邊刪 context；候選字加 suffix 還是太長，就退回只打分候選字本身，不要跳過整格。
   - 分數快取的 key 用 (context, 打分字串)，容量從 256 調大。
   - 加一個 `fullBuffer` 開關，保留舊路徑給 v2 parity 測試用。
   - typo window 的左文改成用已套用 pin 的文字。
2. **雙胞胎詞**：前綴清單放在 Source/Data，建置時產生，不要手改 data.txt。規則是：data.txt 有「<前綴>在」（讀音以 ㄗㄞˋ 結尾）而沒有「<前綴>再」時，補一條，分數 = 在 的分數 − 0.1（log10）。
   - 這次的 18 條見 `tools/twins_added.txt`，對應的前綴清單在 `tools/make_twins.py`。
   - 擴充時只加「再」當副詞合文法的前綴，例如 也、也不、如果、甚至、可、要、都、能、別、請、我們。不要加「在」本身是詞的一部分的，例如 實在、現在、存在、所在、在家、在野、人在。
3. **驗證**：
   - 看 `latency.log` 比較延遲：每次按鍵的 decoder 呼叫數，以及 t64／t96 的時間。
   - 手動打：會再加以討論、請再試一次、等一下再說、我在家、他還在睡覺。
4. **之後另外做**：常用新詞的詞頻更新，例如 連假；0c 的標點模型 `Funmula/0c-punct-d512`（MIT）可以另外評估。

## 重現方式

```bash
export EVAL_WORK=~/.cache/mcbpmflm-eval
SlothE/eval-2026-10/tools/setup.sh
# 步驟：建 venv、下載固定版本的模型、下載 g2pW 模型、make data.txt、編 walk_cli
cd $EVAL_WORK && venv/bin/python -I <repo>/SlothE/eval-2026-10/tools/val_walk.py $EVAL_WORK   # 預期 500/500

# 測試集只公開索引，沒有原文。重建：
venv/bin/python -I <repo>/SlothE/eval-2026-10/tools/rebuild_testset.py <repo>/SlothE/eval-2026-10/testset/news_index.jsonl news_lab.jsonl
venv/bin/python -I <repo>/SlothE/eval-2026-10/tools/rebuild_testset.py <repo>/SlothE/eval-2026-10/testset/chat_index.jsonl chat_lab.jsonl
# 雙胞胎詞庫：
python3 -I <repo>/SlothE/eval-2026-10/tools/make_twins.py <repo>/Source/Data/data.txt data_twins.txt twins_added.txt
# 跑 5 種設定，再分析：
venv/bin/python -I <repo>/SlothE/eval-2026-10/tools/run_fresh.py news_lab.jsonl news_out.json
python3 -I <repo>/SlothE/eval-2026-10/tools/analyze_fresh.py news_lab.jsonl news_out.json
```

- `rebuild_testset.py` 會重新抓每個網址，把網頁內容只留漢字，再用句長和 SHA-1 找回原句。文章被修改或刪除的句子會列為 missing，所以重建後的數字可能跟上面略有差異。
- dev 相關腳本（`dev_variants.py`、`grid_dev.py`、`full_dev.py`、`beta_sweep.py`）要在 `$EVAL_WORK` 裡有 `dev_gold.jsonl`。產生方式是先用 sparse clone 抓 Common Voice repo 的 `server/data/zh-TW`，再跑 `tools/match_gold.py <repo> <zh-TW 目錄> dev_gold.jsonl`。

## 檔案

- `D512.md`：0c D512 reranker 的完整評估報告。
- `PLAN.md`：背景、發現、事先登記的設定和判定規則（英文）。
- `results/`：彙總數字，不含任何原文。
- `testset/{news,chat}_index.jsonl`：每句的 id、來源、網址、發布日期、所屬組別（`general`／`zai`）、長度、SHA-1、注音。新聞和 PTT 原文有著作權，所以不放進 repo。
- 收集測試句用的程式不放在 repo 裡；repo 只留 `testset/` 的索引和 `tools/rebuild_testset.py`。
- `tools/`：離線評估工具，路徑設定在 `evalcfg.py`。D512 單獨解碼用 `d512_beam.py`（可以中斷後接著跑，執行緒數用環境變數 `D512_THREADS` 設定），分析用 `analyze_d512only.py`。
- 模型授權：Sloth 模型（`Luigi/sloth-ime-models` @ `e7d13c9`）是 Apache-2.0；D512（`Funmula/0c-sequence-d512` @ `d512-pf512-20260926`）是 MIT；g2pW（`G2PWModel-v2-onnx`）是 Apache-2.0。
