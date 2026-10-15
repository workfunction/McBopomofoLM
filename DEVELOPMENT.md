# McBopomofoLM 開發者說明

給想了解實作細節、自行編譯、或回報問題的人。一般介紹見 [README](README.markdown)。

## 模型與選字流程

- **SlothE-T 25M encoder**（三元權重、雙向）：對一整串注音的每個位置算出每個合法字的機率，加進小麥注音 walk（Viterbi 斷詞選字）的分數裡。
- **SlothE decoder `pred_q35_60m`**（Qwen3.5 架構，60M）：從 encoder 前 3 名候選中挑出最合理的一個，只在把握夠高時才改掉 walk 的選擇（λ 1、τ 0.5）。v3.4 起是**整句打分**：每個候選連同它右邊的字（最多 24 字）一起打分，前文用前面已經決定的字。它不看正確答案。
- **「再」補詞**（v3.4）：詞庫有「會在」「就在」「你在」卻沒有對應的「再」詞，「在」會被鎖在詞裡，模型沒機會選「再」。建置詞庫時，為 18 個副詞前綴補上「再」詞（`Source/Data/zai-twins.txt` 加上 `Postprocess.txt` 的 `add-twins`），分數比「在」低一點，預設仍是「在」，由模型決定。
  - 18 個詞：不會再、他再、她再、你再、還能再、還會再、會再、就再、將再、先再、只能再、只好再、只會再、最好再、才會再、也會再、也許再、應該再。
- **選字窗**：依模型分數重新排序，目前畫面上已選的字往後放。斷詞斷錯時，如果某個不同長度的候選（例如把「建制」改成「建置」只要選單字「置」）選了之後整句分數明顯較高，就插隊到第一個。只打一個音節時照原版詞頻排序。
- **上屏後帶前文**（v3.1）：在記憶體裡記住最近送出的字和注音，下一段打字時交給兩個模型當前文。換 app、換輸入框、移動游標、閒置 2 分鐘就清掉；只存在記憶體，不寫檔。
- **打錯救回**（v3.1）：打出不合法的讀音時，讀音留在組字區，繼續打會覆蓋同一格（聲母蓋聲母、韻母蓋韻母）。按空白鍵或 ↓ 跳出修正選字窗，候選是「按錯鄰鍵、漏打或多打一個符號」可能想打的字，由 decoder 和 encoder 一起排序。
- **選字窗的錯音修正**（v3.2，預設關閉）：打成另一個合法讀音時，選字窗最後一格提供修正。
- **每個按鍵的流程**：每打完一個音節，主執行緒先用上一輪的分數立刻重新選一次字並顯示（provisional），同時把這一輪交給背景的序列佇列：encoder 讀整串注音（加上前文）→ 用詞頻加 encoder 分數重新 walk → decoder 只對有 2 個以上候選字的位置打分、決定要不要改字 → 結果送回主執行緒套用。打字太快時，過期的那一輪會直接放棄。
- 兩個模型**只在 ANE 上跑**，沒有 CPU 版。載入前逐一核對檔案大小和 SHA-256，也檢查是否真的排到 ANE 上；任何一步失敗就退回原版小麥注音的行為，並把原因寫進記錄。

### 隱藏設定（`defaults write org.openvanilla.inputmethod.McBopomofoLM …`）
- `SlothEDecoderFullBuffer` = NO：回到 v3.3 的逐鍵判斷（λ 2，只對候選字本身打分）。
- `SlothEDecoderSuffixCapChars`：右邊最多看幾個字（預設 24）。
- 「停頓時整句複查」（輸入選單 → 進階）：v3.4 起預設關閉；沒設定過的話，`SlothEDecoderFullBuffer` 關掉時它預設是開。

## 效果

封存測試集：Common Voice zh-TW 1,500 句，設定在開封前寫定，只開封一次。

| | 字元正確率 | 整句全對 |
|---|---|---|
| 原版小麥注音 | 95.25% | 70.4% |
| McBopomofoLM（25M encoder＋decoder） | **96.55%** | **79.1%** |

- **分段打字**（每次上屏兩三個音節，依實際使用的分布切段）：原版 88.6%；每段從零開始的 v3.0 是 88.2%；**帶前文的 v3.1 是 92.7%**，整句全對 43% → 60%。
- **v3.4 的整句打分和「再」補詞**（2026 年 10 月新收集的聊天和新聞文字，各約 1,500 句）：字元正確率都提高約 0.34 個百分點（95% CI [+0.18, +0.50]）；「再」選對率約 67% → 79%。評估資料和方法在 `SlothE/eval-2026-10/`。
- **打錯救回**（模擬每個音節按錯一鍵）：正確的字排進修正選字窗前 3 名的比例 69%，第一頁（9 個）84%。
- ANE 放置：encoder 100%；decoder 各長度版本 86–95%。

## 效能與系統資源

M2、16 GB，2026-10-02 到 10-13 的實際使用（v3.2–v3.3）：

- 按鍵處理函式耗時（不含 app 重繪）：P50 1.5 ms、P95 4.4 ms、P99 10.5 ms（n = 82,022）。
- 模型在背景跑完一輪（encoder＋decoder）：P50 8.8 ms、P95 24 ms、P99 39 ms（n = 22,305），不會擋住打字。v3.4 的整句打分讓長句變慢：30 個音節時 P95 從 9 ms 變成 34 ms（測試主機）。
- CPU：閒置 0%；31 分鐘的正常使用約 41 秒 CPU。
- 記憶體（輸入法 process 的 footprint）：平常 76–200 MB；第一次準備模型時暫時約 600–700 MB。模型檔以 mmap 映射（總共 825 MB、駐留約 118 MB，主要是 encoder 的嵌入表 114 MB），這類 file-backed 記憶體不計入 footprint，記憶體吃緊時 macOS 可以先釋放。Neural Engine 的權重由系統驅動程式管理，不算在輸入法的帳上。整體實際占用估計約 300–400 MB；「活動監視器」只顯示 footprint，所以看起來很低。
- 硬碟：app 160 MB（其中模型 140 MB）；模型副本（見下一節）在 APFS 上是 clone，幾乎不另外占空間；Neural Engine 編譯快取約 1.2 GB（`~/Library/Caches/org.openvanilla.inputmethod.McBopomofoLM/com.apple.e5rt.e5bundlecache`）。

## 安裝流程與模型準備

- `McBopomofoLM install`（安裝程式也是呼叫它）只做三件事，幾秒就好：核對模型檔案（size + SHA-256）、確認模型副本、註冊輸入法。它**不編譯模型**；模型檔案損壞時印 `model files damaged: …`、以狀態 1 結束，不註冊（安裝程式會還原舊版）。
  - 想像 v3.4.0 之前那樣在安裝時就編譯完（約 3 分鐘，有進度條）：`install --compile-now`。
- **模型副本**：Neural Engine 的編譯快取跟著模型檔案（路徑和檔案身分），不跟著 app 或執行檔（實驗見工作區的 `reports/ane-cache-keying.md`）。所以輸入法把模型 clone 到 `~/Library/Application Support/McBopomofoLM/Models/<set id>/`（set id 是 `runtime-manifest.txt` 的 SHA-256 前 16 碼），建立後不再更動，從這份副本載入。
  - 更新 app 時，模型沒變的話副本和快取都原封不動，AI 幾秒內就緒（實測約 6 秒）。
  - 副本以 app 內有簽章的清單核對（啟動時讀一次，存在記憶體）；`Models/.lock` 加 `RENAME_EXCL` 發佈；每個副本有使用中的租約，舊副本只在輸入法載入成功後、確定沒人在用時才刪。
  - `Models` 可以是符號連結（例如測試時指到別處），輸入法會照連結走。
  - 磁碟空間不夠、建不起副本時，改從 app 裡載入，並在這個版本內要求一次乾淨的重新編譯。
- **第一次使用時**：輸入法在背景編譯模型（utility QoS），準備好之前用一般選字；輸入選單的「AI 選字」旁寫「（準備中，約 3 分鐘）」，好了通知「AI 模型已就緒」。實測裝完約 1 分 17 秒 AI 開始選字，約 2.5 分鐘全部準備完。之後依序預先編譯較長的 decoder 函式；「AI 選字」關掉時暫停。
- **清快取的標記**：分成安裝和自我修復兩種，以 rename 搶下並在同一把 flock 裡處理；自我修復每個版本最多一次。只有新建副本或建不起副本時，`install` 才寫安裝標記；模型沒變的重新安裝不寫，快取留著。
- 系統更新後快取是否失效由 macOS 決定；失效時輸入法會在背景重新編譯。

### 確認模型跑在 ANE 上

打開「寫入診斷記錄」（輸入選單的「進階」區，或偏好設定的「進階」分頁），它會立刻寫入目前的模型載入結果。再看 `~/Library/Application Support/McBopomofoLM/latency.log` 裡 `R` 開頭的行：`encoder ane ok`、`decoder ane ok`，後面是 ANE 執行的運算比例。

### 用終端機移除

先在「系統設定 → 鍵盤 → 輸入方式」移除「注音 LM」，然後：

```bash
pkill -x McBopomofoLM; rm -rf ~/Library/Input\ Methods/McBopomofoLM.app
# 模型副本和編譯快取都可以再生
rm -rf ~/Library/Application\ Support/McBopomofoLM/Models ~/Library/Caches/org.openvanilla.inputmethod.McBopomofoLM
# 選做：連設定和使用者詞庫一起刪
rm -rf ~/Library/Application\ Support/McBopomofoLM
defaults delete org.openvanilla.inputmethod.McBopomofoLM
```

安裝程式的「移除」會刪掉 app、模型副本和編譯快取（可再生，不另外詢問），使用者資料（設定和使用者詞庫）另外詢問，預設保留。

## 診斷記錄

「寫入診斷記錄」**預設關閉**；關閉時完全不寫檔，也不建立資料夾。打開後寫到 `~/Library/Application Support/McBopomofoLM/latency.log`：時間、每個按鍵的耗時、組字區長度和各項計數、正在輸入的 app 的 bundle ID、中英文切換、選了修正選字窗的第幾個、前文被清掉的原因、模型載入和 Neural Engine 配置的結果，以及每次模式提示的座標（`H` 行）。**絕不記錄你輸入的文字。** 記錄只在你的電腦上，回報問題時由你決定要不要附上。關掉會立刻停止寫入並關閉檔案，已寫的內容不會刪除。

## 使用細節

- **單點 Shift**：單獨按一下 Shift（不同時按別的鍵）切換中英文。游標旁的提示：系統強調色的小膠囊，macOS 26 以上是玻璃效果，停約 2 秒；app 沒提供游標位置時顯示在滑鼠旁。選單列的「中／A」只在「注音 LM」是目前的輸入法、而且「單點 Shift 切換中英文」開著時出現。英文模式在換視窗、換 app 時保留；切到別的輸入法再切回來時，一律回到中文。
- **進階選項**（輸入選單「進階」標題底下的四個）給開發者和想細調的人用。三個常用選項也在「小麥注音 LM 偏好設定」裡。
- 安裝程式只開啟「注音 LM」，不開啟「傳統注音 LM」；想用的話自己在系統設定加入。

## 發佈檔案與核對

- 每版的發佈說明（含 SHA-256）在 Releases 頁面和工作區的 `release/<版本>/NOTES.md`。
- 安裝程式（universal：arm64 + x86_64，Intel 的 Mac 只會看到說明）把 IME 的 zip 放在 `Contents/Resources/NotarizedArchives/McBopomofoLM-r<build>.zip`，zip 裡是 ad-hoc 簽署的 `McBopomofoLM.app`（沒有 Apple 公證）。用 `codesign --verify --deep --strict McBopomofoLM.app` 驗證簽章結構。
- 打包：`scripts/build-installer.sh <McBopomofoLM.app> <輸出資料夾>`。它檢查 IME 與安裝程式的版本號相同、程式覆蓋率已關閉（`__llvm_profile` 符號為 0），把 IME 壓成上面那個 zip 放進安裝程式，建置 Release 並輸出 `McBopomofoLM-Installer-<版本>.zip`。
- 安裝程式的邏輯在 `Source/Installer/InstallerKit`（`swift test` 跑單元測試）。`McBopomofoInstaller --dry-run <資料夾>` 只做解壓縮、拷貝、移除隔離標記並檢查結果，不安裝、不註冊。

## Core ML 模型

轉好的模型放在 Hugging Face：[workfunction/McBopomofoLM-models](https://huggingface.co/workfunction/McBopomofoLM-models)。

- `runtime/`：app 實際打包的檔案，包括編譯好的 `.mlmodelc`、嵌入表、詞表和 `runtime-manifest.txt`。
- `mlpackage/`：可以重新編譯的原檔。encoder 做 2-bit palettize（權重是三元的，所以沒有損失）；decoder 是 fp16。兩個都是 multifunction，分成多個固定長度的版本：decoder 的 t16–t96 給字串打分，n16–n96 輸出下一個字的機率（用於打錯救回）。
- 版本：HF tag `v3.1.2` 是目前的模型（v3.3、v3.4 都沒換模型）；`v2.1.1` 是舊版用的。

## 自行編譯

需求：Apple Silicon、macOS 15 以上；Xcode 26 以上。

1. 下載模型，放進 bundle 目錄：
   ```bash
   hf download workfunction/McBopomofoLM-models --revision v3.1.2 --include "runtime/*" --local-dir /tmp/mcbpmf-lm
   cp -R /tmp/mcbpmf-lm/runtime/ SlothE/Bundle/SlothE/
   ```
2. 編譯：
   ```bash
   xcodebuild -project McBopomofo.xcodeproj -scheme McBopomofo -configuration Release \
     -derivedDataPath build/DerivedData -destination 'platform=macOS,arch=arm64' build
   ```
3. 安裝：把產出的 `McBopomofoLM.app` 照 README「用終端機」的後三行指令安裝。

## 測試

- XCTest 和 Swift Testing：`xcodebuild test -project McBopomofo.xcodeproj -scheme McBopomofo`。完整跑一次約 22–28 分鐘，其中 10 個載入模型的測試占約 18 分鐘（每次重新 build，ANE 快取就失效）。
- 同一時間只能有一個測試程式載入模型：兩個同時編譯 ANE 會出現 `ANECCompile() FAILED`。
- 冷啟動和模擬更新的測試預設跳過，要另外打開（`ANEColdStartTests`、`V34pCacheHitTests`）。

## 跟上游的關係

- `master` 追蹤 [openvanilla/McBopomofo](https://github.com/openvanilla/McBopomofo)，`lm` 是這個 fork 的開發分支。
- bundle id、輸入法 ID、偏好設定和使用者詞庫目錄都跟原版分開，也移除了所有連網路徑（例如更新檢查）。
