# McBopomofoTests/SlothEFixtures/

SlothE 測試的參考資料：17 個 JSONL，合計約 25 MB。這是資料夾參照，整個複製進測試 bundle 的 Resources（`McBopomofo.xcodeproj` 的 `SlothEFixtures`）。

- 句子來源：Mozilla Common Voice zh-TW（CC0）的 500 句 dev（`sid` 為 `cvNNNNN`）。沒有新聞或 PTT 文字。Common Voice 以 CC0 釋出；我們沒有逐句核對授權。
- 對照：`SlothETests.mm` ← `syllable_map`、`rerank_parity`、`logprob_probe`、`v2_logprob_probe`；`SlothEDecoderTests.mm` ← `v2_dec_long`、`v2_parity`、`v2_tokenizer`、`walk2_dev_final2_Aprime.walks`。
- `SlothEInWalkTests.mm` ← `inwalk_parity`、`v2_parity`、`v302_*`（3 個）、`v31a_dev_chunks`、`v31b_typo_dev`、`v32a_dev_full_gate`、`v32b_dev_t2`。
- 其他：`v32_parity_golden` 給 `SlothEV32ParityTests`；`inwalk_parity` 也給 `SlothEAsyncTests`、`SlothEPipelineTests`；`v31a_dev_chunks` 也給 `SlothEV32Tests` 和 `SlothEV34EvalTests`。
- 產生：`SlothE/make_*.py`、`SlothE/v2/`、`SlothE/v31b/make_typo_fixture.py`。`v32_parity_golden.jsonl` 是在乾淨的 v3.1.2 樹上跑 `SlothEV32ParityTests` 並設 `TEST_RUNNER_SLOTHE_PARITY_OUT` 錄下來的。
- 不要手改；改了模型、詞庫或打分規則以後，要用原腳本重產並說明差異。
- 授權：同 repo（MIT）；句子的出處見上。
