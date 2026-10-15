# SlothE/

McBopomofoLM 的 SlothE 模型相關檔案：執行時資料、測試 fixture 的產生腳本，以及離線評估。

- `Bundle/SlothE/`：打包進 app 的執行時檔案（詞表、字表、manifest）。模型本體另外下載，見 `Bundle/README.md`。
- `make_*.py`、`v2/`、`v31b/make_typo_fixture.py`：重新產生 `McBopomofoTests/SlothEFixtures/` 各個 fixture 的腳本。輸入是作者工作區裡的離線研究結果（腳本開頭的 docstring 有列出），不在這個 repo 裡，所以外部開發者無法直接重跑；fixture 已經提交，一般用不到。
- `prep_resources_v2.py`、`v31b/build_dec_tn.py`：把 Core ML 模型編譯、整理成 `Bundle/SlothE/` 的打包腳本。公開的成品放在 Hugging Face（見 DEVELOPMENT.md「Core ML 模型」）。
- `ThirdParty/opencc/`：異體字對照表的原始表（Apache-2.0）。
- `eval-2026-10/`：2026 年 10 月的離線評估（在／再、D512、整句打分），入口是它自己的 README.md。
- 這個資料夾只有 `Bundle/SlothE/` 會被 Xcode 專案引用（Resources 資料夾參照）；其餘都不參與建置，也不被測試執行。
- 授權：程式碼同 repo（MIT，見根目錄 LICENSE.txt）；模型與資料的授權見 `Bundle/SlothE/NOTICE.txt`。
