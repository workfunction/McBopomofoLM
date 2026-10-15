# SlothE/eval-2026-10/tools/

離線評估工具，用來重現 `../README.md` 裡的數字。不屬於 app，也不參與任何 target 的建置或測試。

- 路徑集中在 `evalcfg.py`；產生的東西（模型、venv、`walk_cli`、快取）放在 `$EVAL_WORK`（預設 `~/.cache/mcbpmflm-eval`）。
- 一次性設定：`setup.sh`（建 venv、下載固定版本的模型與 g2pW、編 `walk_cli`）；套件版本固定在 `requirements.txt`。
- 主流程：`pipeline.py`（斷詞、encoder、decoder、釘選）→ `run_fresh.py`（跑 5 種設定）→ `analyze_fresh.py`（差距與 95% 信賴區間）。
- 測試句：`rebuild_testset.py` 依 `../testset/` 的索引重抓網頁並重建句子；`label.py` 把 g2pW 的注音對齊到詞庫的合法讀音。
- 驗證：`val_walk.py` 對 `McBopomofoTests/SlothEFixtures` 重跑斷詞，預期 500/500 一致。
- dev 掃描：`dev_variants.py`、`grid_dev.py`、`full_dev.py`、`beta_sweep.py`，需要 `$EVAL_WORK/dev_gold.jsonl`（由 `match_gold.py` 產生）。
- D512 單獨解碼：`d512.py`、`d512_beam.py`、`analyze_d512only.py`。
- 「再」雙胞胎詞：`make_twins.py` 從 `Source/Data/data.txt` 產生補詞後的詞庫；`twins_added.txt` 是評估時補的 18 條（app 實際使用的是 `Source/Data/zai-twins.txt`）。
- 讀 repo 內的 `Source/Data/data.txt`、`SlothE/Bundle/SlothE`、`McBopomofoTests/SlothEFixtures`；`build_walk_cli.sh` 用 repo 的 C++ 原始碼編譯。
- 授權：程式碼同 repo（MIT）；用到的模型授權見 `../README.md` 最後一節。
