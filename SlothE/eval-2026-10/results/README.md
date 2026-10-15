# SlothE/eval-2026-10/results/

評估的彙總輸出。只有數字，沒有任何新聞或 PTT 原文。

- `news_analysis.txt`、`chat_analysis.txt`：新測試集上 A 到 E 五種設定的字元／整句正確率、成對差距與 95% 信賴區間、在／再的混淆表（`tools/analyze_fresh.py` 產生）。
- `news_d512only_analysis.txt`、`chat_d512only_analysis.txt`：D512 單獨解碼的對照（`tools/analyze_d512only.py` 產生）。
- `full_dev.log`、`grid_dev.log`、`beta_sweep.log`、`dev_variants.log`：500 句 Common Voice dev 上的參數掃描，只用來調參（腳本是 `tools/` 裡的同名檔）。
- 怎麼讀：見 `../README.md` 的「結果」一節。這些檔案是 2026-10-13 的輸出，不會自動更新。
