# SlothE/v31b/

v3.1b（打錯字視窗用的 next-token 函式）相關檔案。

- `build_dec_tn.py`：把 decoder 的 t16–t96（算分）和 n16–n96（next-token）合併成一個 multifunction 套件，並寫出 `dec_next_first.tsv`。
- `dec_next_first.tsv`：「字 → 開頭是該字的 token id」對照表（96 KB）。`SlothE/prep_resources_v2.py` 讀它，複製成 `SlothE/Bundle/SlothE/dec_next_first.tsv`，兩份內容完全相同。
- `check_dec_tn.py`：驗證合併後的 t* 函式輸出跟舊版逐位元相同（CPU_ONLY 與 CPU_AND_NE）。
- `make_typo_fixture.py`：產生 `McBopomofoTests/SlothEFixtures/v31b_typo_dev.jsonl`（8,548 列）。
- 重跑：三支腳本需要作者工作區的 `ane/dec`、`sloth`、`typo`（`$W`，見各自的 docstring），外部開發者無法直接重跑。`dec_next_first.tsv` 與已提交的 fixture 是現成的結果。
- 誰用：`dec_next_first.tsv` 被 `prep_resources_v2.py` 讀取；`build_dec_tn.py`、`check_dec_tn.py` 只在 `SlothETests.mm` 的註解裡被引用，其餘腳本也沒有被建置或測試呼叫。
