# McBopomofoTests/

小麥注音 LM 的測試（XCTest 與 Swift Testing，同一個 target `McBopomofoTests`，host 是 app 本身）：上游原有的測試，加上這個 fork 新增的 SlothE、v3.x 功能、輸入來源與模式提示測試。

## 迴歸測試與 Lab

- **迴歸測試**：這個資料夾裡的測試檔。保護已發布的行為，改版前要全部通過。
- **Lab**（`Lab/`）：研究和量測用，不是迴歸測試。三個類別都要設環境變數才會跑，沒設就自己 skip：
  - `ANEColdStartTests`：安裝後第一次啟動的冷編譯，以及編譯期間打字的延遲。會清掉這台機器的編譯快取，只在 dev bundle id 的 host 上跑，要單獨跑。
  - `V34pCacheHitTests`：換掉 app 以後，模型副本讓編譯快取仍然命中。分三個階段，各用一次 xcodebuild，中間換掉 host app。
  - `SlothEV34EvalTests`（`.mm`）：在 app 的 pipeline 上重做 v3.4 的雲端評估、量每個音節的延遲、逐階段探測一個句子。輸入（標好注音的測試集、沒有「再」補詞的 data.txt）不在 repo 裡，見 `SlothE/eval-2026-10/`。
- 有幾個選用的量測留在迴歸類別裡，因為它們跟同類別的測試共用 setUp：`SLOTHE_BENCH=1` 會打開 `SlothEPipelineTests` 的 5 個、`SlothEInWalkTests` 的 1 個，以及 `SlothEDecoderTests.mm` 裡一個測試的重新編譯段。

## 三個 test plan

repo 根目錄有三個 plan，scheme `McBopomofo` 都列得出來：

| plan | 用途 | 跑什麼 |
|---|---|---|
| `McBopomofo`（scheme 預設） | 本機發佈前的完整驗收 | 全部，只跳過 Lab 的三個類別。包含 10 個載入模型的測試 |
| `McBopomofoLab` | 研究與量測 | 只選 Lab 的三個類別 |
| `McBopomofoCI` | GitHub Actions | 同 `McBopomofo`，另外跳過在乾淨 clone 上跑不過的測試：需要 `SlothE/Bundle/SlothE/` 裡那些沒有進 git 的模型檔（`*.mlmodelc`、`enc25m_embed_f16.bin`、`dec_tokenizer.json`）的測試 |

```bash
# 完整驗收（預設 plan）
xcodebuild test -project McBopomofo.xcodeproj -scheme McBopomofo
# 沒有模型檔時，跟 CI 一樣
xcodebuild test -project McBopomofo.xcodeproj -scheme McBopomofo -testPlan McBopomofoCI
# Lab：加上環境變數（xcodebuild 要求 TEST_RUNNER_ 前綴）
TEST_RUNNER_SLOTHE_COLD_START_TEST=1 xcodebuild test -project McBopomofo.xcodeproj -scheme McBopomofo \
    -testPlan McBopomofoLab -only-testing:McBopomofoTests/ANEColdStartTests
```

新增一個要用真模型的測試時，同時把它加進 `McBopomofoCI.xctestplan` 的 `skippedTests`，否則 CI 會失敗（沒有模型時它會直接失敗，有的甚至會當掉 host）。

## Lab 的環境變數

| 變數 | 作用 |
|---|---|
| `SLOTHE_COLD_START_TEST=1` | 打開 `ANEColdStartTests` |
| `SLOTHE_STORE_HIT_TEST=1`、`SLOTHE_STORE_PHASE=1\|2\|3` | 打開 `V34pCacheHitTests` 的某個階段，每個階段各跑一次 |
| `V34_REPLICATE_OUT`、`V34_LABELED_DIR`、`V34_OLD_DATA`（`V34_SETS`，預設 `chat,news`） | `SlothEV34EvalTests` 重現雲端設定 B、C、D。輸出資料夾、`<set>_lab.jsonl` 所在資料夾、沒有「再」補詞的 data.txt |
| `V34_LATENCY_OUT`（`V34_LATENCY_BUFFERS`，預設 60） | `SlothEV34EvalTests` 量每個音節的 decoder 延遲 |
| `V34_PROBE_OUT`（`V34_PROBE_READINGS`、`V34_PROBE_CHUNK`、`V34_PROBE_NAME`） | `SlothEV34EvalTests` 逐階段探測一個句子 |
| `V34_PROBE_OUT`、`V34_PROBE_TRACE_READINGS`（`V34_PROBE_SLOTS`、`V34_PROBE_NAME`） | `SlothEV34EvalTests` 依前綴逐步追蹤 |

迴歸類別裡的選用變數：`SLOTHE_BENCH=1`（延遲、記憶體、冷編譯量測）；`SLOTHE_PARITY_OUT=<檔案>`（`SlothEV32ParityTests` 的錄製模式，用來重錄 `v32_parity_golden.jsonl`）。

## 其他

- 載入模型的測試（`SlothEPipelineTests`、`SlothELazyDecoderTests` 等 10 個）約占完整測試的 18 分鐘；同一時間只能跑一個，否則 ANE 編譯會互相干擾。沒動到載入程式時，用 `-skip-testing` 跳過它們。
- fixture 在 `SlothEFixtures/`，說明見該資料夾。
- `V33ReleaseTests` 會讀 `Source/McBopomofo-Info.plist` 和安裝程式的 Info.plist 檢查版本號，每次改版要同步更新；`V33EnglishIndicatorTests` 也用 `#filePath` 讀 Info.plist，搬動測試檔案時要留意。
- 安裝程式的邏輯測試不在這裡，在 `Source/Installer/InstallerKit/Tests`（`swift test`）。
- 授權：同 repo（MIT）。
