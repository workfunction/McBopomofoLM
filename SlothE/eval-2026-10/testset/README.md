# SlothE/eval-2026-10/testset/

評估用句子的索引，沒有原文。

- `news_index.jsonl`：3,356 筆（公視、三立、ETtoday、風傳媒）；`chat_index.jsonl`：4,368 筆（PTT 22 個版）。發布日期 2026-10-01 起。
- 欄位：`id`、`source`、`url`、`published`、`pools`（分組，`general`／`zai`）、`n`（漢字數）、`sha1`（原句漢字的 SHA-1）、`readings`（注音）；聊天組另有 `kind`（`body`／`push`）。
- 為什麼沒有原文：新聞和 PTT 文章有著作權，不放進 repo。檢查過：除了 `source` 欄位，檔案裡沒有任何漢字。
- 重建：`python3 -I SlothE/eval-2026-10/tools/rebuild_testset.py testset/news_index.jsonl news_lab.jsonl`。它會重抓每個網址（每個網址至少間隔 1.2 秒，帶瀏覽器樣式的 User-Agent，PTT 另帶 `over18=1` cookie），用長度加 SHA-1 找回原句；文章被修改或刪除的句子列為 missing，所以重建後的數字可能略有差異。
- 執行 `rebuild_testset.py` 前，請自行確認各網站的 robots.txt 與使用條款允許這樣抓取；腳本不會替你檢查。重建出來的原文有著作權，只供重現這份評估，不要轉貼、散布，也不要放進 repo。收集當時哪些來源因為 robots.txt 沒有抓，見 `../README.md` 的「方法」一節。
- 用途與結果：見 `../README.md`。這些索引只用來一次性評估，不是 app 或測試的輸入。
