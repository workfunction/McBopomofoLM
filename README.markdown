# McBopomofoLM：小麥注音 × 本機神經模型

這是 [小麥注音輸入法（McBopomofo）](https://github.com/openvanilla/McBopomofo) 的 fork。它在原本的選字引擎上，加了兩個在 Apple Neural Engine（ANE）上執行的小型神經模型，讓選字更準。打字方式、詞庫和設定都跟小麥注音一樣。安裝後它是另一個獨立的輸入法「注音 LM」，可以跟原版小麥注音並存。

模型來自 [Sloth 輸入法（vieenrose/sloth-zhuyin-linux）](https://github.com/vieenrose/sloth-zhuyin-linux)。它是 Linux 上的神經注音輸入法（ibus／fcitx5）。模型權重由作者發布在 Hugging Face：[Luigi/sloth-ime-models](https://huggingface.co/Luigi/sloth-ime-models)。這個 fork 把其中兩個模型轉成 Core ML，接進小麥注音的 C++ 選字引擎。

## 一般使用者

### 需求

- Apple 晶片的 Mac（M1 以上）。Intel 的 Mac 不能用，安裝程式會告訴你。
- macOS 15 以上。

### 安裝（約 5 分鐘，其中約 3 分鐘在等電腦準備 AI 模型）

這個輸入法沒有 Apple 的開發者簽章（沒有付 Apple 的年費），所以第一次打開安裝程式時，macOS 會擋下來，要你到「系統設定」放行一次。照下面做就好，不需要用終端機。

1. 到 [Releases](https://github.com/workfunction/McBopomofoLM/releases) 下載最新的 `McBopomofoLM-v3.3.0.zip`。到「下載項目」雙擊它解壓縮，會得到「安裝程式」。
2. 雙擊「安裝程式」。macOS 會跳出「Apple 無法驗證…」之類的警告，按「完成」（**不要**按「移到垃圾桶」）。
3. 打開「系統設定 → 隱私權與安全性」，往下捲到「安全性」。會看到一行「已阻擋「安裝程式」…」，按旁邊的「仍要打開」。
   - 這個按鈕只會在你剛剛被擋下之後的一段時間（約一小時）內出現。找不到的話，回到步驟 2 再雙擊一次安裝程式，然後馬上回來看。
   - 按下去要輸入這台 Mac 的**管理員密碼**（或用 Touch ID）。如果你用的是標準（非管理員）帳號，請管理員幫你輸入密碼。
   - 之後 macOS 會再問一次，按「打開」。
4. 安裝程式的視窗出現後，按「同意並安裝」。視窗裡的進度條會跑約 3 分鐘，這段時間電腦在為神經網路引擎（Neural Engine）編譯 AI 模型，**請不要關掉視窗**。看到「安裝成功」就完成了。
5. 加入輸入法：打開「系統設定 → 鍵盤 → 文字輸入 → 輸入方式 → 編輯…」，按「＋」，選「繁體中文」，加入「注音 LM」（咖啡色的 ㄅ）。安裝程式已經替你啟用它，如果在選單列右上角的輸入選單裡已經看到「注音 LM」，這一步可以跳過。
6. 用你平常的方式（例如 Control + 空白鍵，或鍵盤上的地球儀鍵）切換到「注音 LM」，開始打字。

裝好之後不要搬動或改名這個輸入法，因為 macOS 的 AI 模型快取綁定它的位置。

**如果找不到「仍要打開」（備案）**：打開「終端機」，輸入下面這行（最後有一個空白），再把「安裝程式」從 Finder 拖進終端機視窗，按 Return；之後再雙擊安裝程式就不會被擋了：

```bash
xattr -dr com.apple.quarantine 
```

### 使用

- **單點 Shift 切換中英文**：單獨按一下 Shift（不要同時按別的鍵）就在中文和英文之間切換，打到一半的字不會被送出。英文模式下，選單列右上角會多一個咖啡色的「A」圖示。第一次切到英文時會有一則說明。不想要的話，到輸入選單把「單點 Shift 切換中英文」取消打勾。
- **AI 選字**：輸入選單裡的「AI 選字」是總開關。關掉就跟原版小麥注音一樣選字。
- **選字窗提供錯音修正**：預設關閉。打開後，選字窗最後一格會提供「可能按錯鍵」的修正。
- **輸入選單**裡的「進階」是給開發者和想細調的人用的，一般不用動。
- 三個常用選項也在「小麥注音 LM 偏好設定」裡。
- **第一次使用時**：如果電腦還在準備 AI 模型，會出現「AI 模型準備中，約需 3 分鐘，期間使用一般選字」，準備好會再通知「AI 模型已就緒」。準備期間照常可以打字，只是用一般的選字。

### 隱私

- 這個輸入法**完全離線**：更新檢查等所有連網的功能，都在編譯時就移除了，不會傳送你輸入的任何東西。
- 為了讓選字更準，輸入法會在記憶體裡記住最近送出的幾個字和注音（「上屏後帶前文」）。這些只存在記憶體，不寫入任何檔案；換 app、換輸入框、移動游標或閒置 2 分鐘就清掉。
- 「寫入診斷記錄」（輸入選單 → 進階，也在偏好設定的「進階」分頁）**預設關閉**。關閉時完全不寫檔案，也不會建立資料夾。打開之後，會在 `~/Library/Application Support/McBopomofoLM/latency.log` 記錄每個按鍵的耗時、組字區的長度，以及你正在輸入的 app 的 bundle ID（例如 `com.apple.Notes`），**絕不記錄你輸入的文字**。這份記錄只在你的電腦上，沒有任何程式會傳出去；回報問題時，你可以自己決定要不要附上。

### 更新與移除

- **更新**：下載新版，再打開新版的安裝程式。它會先關掉舊版再安裝，目前正在使用「注音 LM」的話，會先幫你切換到 ABC。
- **移除**：打開安裝程式，按「移除」。它會停用輸入法、把輸入法丟進垃圾桶，並詢問要不要一併刪除設定和使用者詞庫（`~/Library/Application Support/McBopomofoLM`，預設保留）。

## 做了什麼

- **SlothE-T 25M encoder**（三元權重、雙向）：對一整串注音的每個位置算出每個合法字的機率，加進小麥注音 walk（Viterbi 斷詞選字）的分數裡。
- **SlothE decoder `pred_q35_60m`**（Qwen3.5 架構，60M）：從 encoder 前 3 名候選中挑出最合理的一個，只在把握夠高時才改掉 walk 的選擇。它只看左邊已經轉出來的字，不看正確答案。
- **選字窗**：依模型分數重新排序，把目前畫面上已選的字往後放。斷詞斷錯時，如果某個不同長度的候選（例如把「建制」改成「建置」只要選單字「置」）選了之後整句分數明顯較高，就把它插隊到第一個。只打一個音節時照原版詞頻排序；冷僻字不再並列排在最前面。
- **上屏後帶前文**（v3.1）：輸入法在記憶體裡記住最近送出的字和注音，下一段打字時交給兩個模型當前文，所以兩三個字就送出也選得準。換 app、換輸入框、移動游標、閒置 2 分鐘就清掉；只存在記憶體，不寫檔、不記 log。
- **打錯救回**（v3.1）：打出不合法的讀音時，讀音留在組字區，繼續打會覆蓋同一格（聲母蓋聲母、韻母蓋韻母）。按空白鍵或 ↓ 會跳出修正選字窗，候選是「按錯鄰鍵、漏打或多打一個符號」可能想打的字，由 decoder（依左文預測下一個字）和 encoder 一起排序；漏打韻母直接按聲調的情況會加分。
- 兩個模型**只在 ANE 上跑**，沒有 CPU 版。載入前會逐一核對檔案大小和 SHA-256，也會檢查是否真的排到 ANE 上。任何一步失敗，就退回原版小麥注音的行為，並把原因寫進 log。

## 效果

封存測試集：Common Voice zh-TW 1,500 句。設定在開封前就寫定，而且只開封一次。

| | 字元正確率 | 整句全對 |
|---|---|---|
| 原版小麥注音 | 95.25% | 70.4% |
| McBopomofoLM（25M encoder＋decoder） | **96.55%** | **79.1%** |

- **分段打字**（模擬每次上屏只有兩三個音節，依實際使用的分布切段；Common Voice test 集，字元正確率）：原版 88.6%；每段從零開始的 v3.0 是 88.2%，跟原版差不多；**帶前文的 v3.1 是 92.7%**，整句全對 43% → 60%。所以 v3.1 的「帶前文」很重要。
- **打錯救回**（模擬每個音節按錯一鍵）：非法讀音的修正選字窗裡，正確的字排進前 3 名的比例 69%，第一頁（9 個）84%。
- 延遲（M2，按鍵到畫面）：P50 8.6 ms，P95 19 ms。模型在背景逐音節執行，不會卡住按鍵處理。
- ANE 放置：encoder 100%；decoder 各長度版本 86–95%。

## 開發者

以下是用終端機安裝、自行編譯、模型和發佈檔案的細節。

### 用終端機安裝（不用安裝程式）

發佈的 zip 裡只有安裝程式。要從終端機安裝，可以把安裝程式內嵌的 IME 取出（模型已經包在裡面），或是自己編譯（見下面「自行編譯」）：

```bash
mkdir -p /tmp/mcbpmf && unzip -q -d /tmp/mcbpmf "安裝程式.app/Contents/Resources/NotarizedArchives/"*.zip
cd /tmp/mcbpmf   # 裡面就是 McBopomofoLM.app
```

需求同上。

0. 先切換到別的輸入法（例如 ABC），等步驟 3 跑完再切回來。
1. 移除隔離標記。這個 app 只做了 ad-hoc 簽署，沒有經過 Apple 公證，不移除的話 macOS 會擋住：
   ```bash
   xattr -dr com.apple.quarantine McBopomofoLM.app
   ```
2. 放進輸入法目錄（如果裝過舊版，會先刪掉）：
   ```bash
   pkill -x McBopomofoLM; rm -rf ~/Library/Input\ Methods/McBopomofoLM.app
   ditto McBopomofoLM.app ~/Library/Input\ Methods/McBopomofoLM.app
   ```
3. 為 Neural Engine 編譯模型並註冊輸入法（約 3 分鐘，畫面會顯示進度條，請等它跑完；想看每一步的細節就加 `--verbose`）：
   ```bash
   ~/Library/Input\ Methods/McBopomofoLM.app/Contents/MacOS/McBopomofoLM install
   ```
   最後一行出現 `both models run on the Neural Engine` 就是成功了。
4. 到「系統設定 → 鍵盤 → 輸入方式 → 編輯… → ＋ → 繁體中文」，加入「注音 LM」（咖啡色 ㄅ）。

裝好之後不要搬動或改名 app，因為 macOS 的 Neural Engine 編譯快取綁定 app 路徑。

**確認模型跑在 ANE 上**：打開 `latency.log`（要先在輸入選單 → 進階 → 寫入診斷記錄打開，預設不寫），`R` 開頭的行會寫 `encoder ane ok`、`decoder ane ok`，後面是 ANE 執行的運算比例。

**用終端機移除**：先在「系統設定 → 鍵盤 → 輸入方式」移除「注音 LM」，然後：
```bash
pkill -x McBopomofoLM; rm -rf ~/Library/Input\ Methods/McBopomofoLM.app
# 選做：連設定、詞庫、快取一起刪
rm -rf ~/Library/Application\ Support/McBopomofoLM ~/Library/Caches/org.openvanilla.inputmethod.McBopomofoLM
defaults delete org.openvanilla.inputmethod.McBopomofoLM
```

### 發佈檔案與核對

- 每版的發佈說明在 `release/<版本>/NOTES.md`，裡面有 IME 本體（`McBopomofoLM.app/Contents/MacOS/McBopomofoLM`）和 zip 的 SHA-256。下載後可以核對：
  ```bash
  shasum -a 256 McBopomofoLM-v3.3.0.zip
  ```
- 安裝程式把 IME 的 zip 放在自己的 `Contents/Resources/NotarizedArchives/`，zip 裡是 ad-hoc 簽署的 `McBopomofoLM.app`。用 `codesign --verify --deep --strict McBopomofoLM.app` 驗證簽章結構。
- 發佈內容不含任何 Apple 公證；這是 v3.3 的設計（見 `PLAN.md` §22）。

### Core ML 模型

轉好的模型放在 Hugging Face：[workfunction/McBopomofoLM-models](https://huggingface.co/workfunction/McBopomofoLM-models)。

- `runtime/`：app 實際打包的檔案，包括編譯好的 `.mlmodelc`、嵌入表和詞表，以及 `runtime-manifest.txt`。
- `mlpackage/`：可以重新編譯的原檔。encoder 做 2-bit palettize，因為權重是三元的，所以沒有損失；decoder 是 fp16。兩個都是 multifunction，分成多個固定長度的版本。decoder 有兩組函式：t16–t96 給字串打分數，n16–n96 輸出下一個字的完整機率（v3.1 新增，用於打錯救回）。
- 版本：HF 的 tag `v3.1.2` 對應本版的模型（v3.3 沒有改模型）；`v2.1.1` 是舊版用的模型。

### 自行編譯

需求：Apple Silicon、macOS 15 以上（執行）；Xcode 26 以上（編譯）。

1. 下載模型，放進 bundle 目錄（`--revision` 要跟你編譯的版本一致，編舊版 v2.1.1 就用 `v2.1.1`）：
   ```bash
   hf download workfunction/McBopomofoLM-models --revision v3.1.2 --include "runtime/*" --local-dir /tmp/mcbpmf-lm
   cp -R /tmp/mcbpmf-lm/runtime/ SlothE/Bundle/SlothE/
   ```
2. 編譯：
   ```bash
   xcodebuild -project McBopomofo.xcodeproj -scheme McBopomofo -configuration Release \
     -derivedDataPath build/DerivedData -destination 'platform=macOS,arch=arm64' build
   ```
3. 安裝：編譯產出的 `McBopomofoLM.app` 照上面「用終端機安裝」的步驟 2–4 安裝（自己編譯的不需要步驟 1）。

安裝後不要搬動或改名 app，因為 macOS 的 ANE 編譯快取綁定 app 路徑。

### 跟上游的關係

- `master` 追蹤 [openvanilla/McBopomofo](https://github.com/openvanilla/McBopomofo)，`lm` 是這個 fork 的開發分支。
- bundle id、輸入法 ID、偏好設定和使用者詞庫目錄都跟原版分開，也移除了所有連網路徑（例如更新檢查）。

## 授權

- 小麥注音：MIT（[LICENSE.txt](LICENSE.txt)），本 fork 的程式碼沿用。
- SlothE 模型權重：Apache-2.0（[Luigi/sloth-ime-models](https://huggingface.co/Luigi/sloth-ime-models)）；`char2id` 字表：Apache-2.0（[Luigi/slothing-web](https://huggingface.co/spaces/Luigi/slothing-web)）；異體字對照：衍生自 OpenCC，Apache-2.0。詳見 [NOTICE](SlothE/Bundle/SlothE/NOTICE.txt)。

感謝小麥注音的開發者，也感謝 Sloth 輸入法作者公開模型與研究。
