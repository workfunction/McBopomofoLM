# McBopomofoLM：小麥注音 × 本機神經模型

這是 [小麥注音輸入法（McBopomofo）](https://github.com/openvanilla/McBopomofo) 的 fork。它在原本的選字引擎上，加了兩個在 Apple Neural Engine（ANE）上執行的小型神經模型，讓選字更準。打字方式、詞庫和設定都跟小麥注音一樣。安裝後它是另一個獨立的輸入法「注音 LM」，可以跟原版小麥注音並存。

模型來自 [Sloth 輸入法（vieenrose/sloth-zhuyin-linux）](https://github.com/vieenrose/sloth-zhuyin-linux)。它是 Linux 上的神經注音輸入法（ibus／fcitx5）。模型權重由作者發布在 Hugging Face：[Luigi/sloth-ime-models](https://huggingface.co/Luigi/sloth-ime-models)。這個 fork 把其中兩個模型轉成 Core ML，接進小麥注音的 C++ 選字引擎。

## 做了什麼

- **SlothE-T 25M encoder**（三元權重、雙向）：對一整串注音的每個位置算出每個合法字的機率，加進小麥注音 walk（Viterbi 斷詞選字）的分數裡。
- **SlothE decoder `pred_q35_60m`**（Qwen3.5 架構，60M）：從 encoder 前 3 名候選中挑出最合理的一個，只在把握夠高時才改掉 walk 的選擇。它只看左邊已經轉出來的字，不看正確答案。
- **候選窗**：依模型分數重新排序，把目前畫面上已選的字往後放。
- 兩個模型**只在 ANE 上跑**，沒有 CPU 版。載入前會逐一核對檔案大小和 SHA-256，也會檢查是否真的排到 ANE 上。任何一步失敗，就退回原版小麥注音的行為，並把原因寫進 log。

## 效果

封存測試集：Common Voice zh-TW 1,500 句。設定在開封前就寫定，而且只開封一次。

| | 字元正確率 | 整句全對 |
|---|---|---|
| 原版小麥注音 | 95.25% | 70.4% |
| McBopomofoLM（25M encoder＋decoder） | **96.55%** | **79.1%** |

- 延遲（M2，按鍵到畫面）：P50 8.6 ms，P95 19 ms。模型在背景逐音節執行，不會卡住按鍵處理。
- ANE 放置：encoder 100%；decoder 各長度版本 86–95%。

## Core ML 模型

轉好的模型放在 Hugging Face：[workfunction/McBopomofoLM-models](https://huggingface.co/workfunction/McBopomofoLM-models)。

- `runtime/`：app 實際打包的檔案，包括編譯好的 `.mlmodelc`、嵌入表和詞表，以及 `runtime-manifest.txt`。
- `mlpackage/`：可以重新編譯的原檔。encoder 做 2-bit palettize，因為權重是三元的，所以沒有損失；decoder 是 fp16。兩個都是 multifunction，分成多個固定長度的版本。

## 自行編譯

需求：Apple Silicon、macOS 15 以上（執行）；Xcode 26 以上（編譯）。

1. 下載模型，放進 bundle 目錄：
   ```bash
   hf download workfunction/McBopomofoLM-models --include "runtime/*" --local-dir /tmp/mcbpmf-lm
   cp -R /tmp/mcbpmf-lm/runtime/ SlothE/Bundle/SlothE/
   ```
2. 編譯：
   ```bash
   xcodebuild -project McBopomofo.xcodeproj -scheme McBopomofo -configuration Release \
     -derivedDataPath build/DerivedData -destination 'platform=macOS,arch=arm64' build
   ```
3. 安裝：把 `McBopomofoLM.app` 放到 `~/Library/Input Methods/`，執行一次 `McBopomofoLM.app/Contents/MacOS/McBopomofoLM install`。這一步會先把模型編譯給 ANE（約 1–3 分鐘），再註冊輸入法。最後一行出現 `both models run on the Neural Engine` 就表示成功。
4. 到「系統設定 → 鍵盤 → 輸入方式」加入「注音 LM」。

安裝後不要搬動或改名 app，因為 macOS 的 ANE 編譯快取綁定 app 路徑。

## 跟上游的關係

- `master` 追蹤 [openvanilla/McBopomofo](https://github.com/openvanilla/McBopomofo)，`lm` 是這個 fork 的開發分支。
- bundle id、輸入法 ID、偏好設定和使用者詞庫目錄都跟原版分開，也移除了所有連網路徑（例如更新檢查）。

## 授權

- 小麥注音：MIT（[LICENSE.txt](LICENSE.txt)），本 fork 的程式碼沿用。
- SlothE 模型權重：Apache-2.0（[Luigi/sloth-ime-models](https://huggingface.co/Luigi/sloth-ime-models)）；`char2id` 字表：Apache-2.0（[Luigi/slothing-web](https://huggingface.co/spaces/Luigi/slothing-web)）；異體字對照：衍生自 OpenCC，Apache-2.0。詳見 [NOTICE](SlothE/Bundle/SlothE/NOTICE.txt)。

感謝小麥注音的開發者，也感謝 Sloth 輸入法作者公開模型與研究。
