# SlothE/v2/

v2（Core ML／ANE）測試 fixture 的產生腳本。

- `make_v2_reference.py`：用 app 打包的兩個 `.mlmodelc`（`enc25m`、`dec60m`）在 coremltools 上重跑 walk2 的 config A'，寫出 `McBopomofoTests/SlothEFixtures/` 的 `v2_parity.jsonl`、`v2_logprob_probe.jsonl`、`v2_tokenizer.jsonl`。
- `make_v2_fixtures.py`：用同一組 Core ML 模型重算 `rerank_parity.jsonl`、`logprob_probe.jsonl`，並產生 `v2_dec_long.jsonl`（長前文的 decoder 呼叫）。
- 誰用：產生的 fixture 由 `SlothETests.mm`、`SlothEDecoderTests.mm`、`SlothEInWalkTests.mm` 讀取。這兩支腳本本身不被建置或測試呼叫，只在測試註解裡被引用。
- 重跑：需要作者工作區的 walk2、ane、sloth 資料夾和 Python venv（`$W`，見各腳本的 docstring），外部開發者無法直接重跑；fixture 已經提交。
- `work/` 是暫存資料夾（約 3 MB），已列入 `.gitignore`，不要提交。
