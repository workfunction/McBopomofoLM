# SlothE/Bundle/SlothE/

app 的 Resources 資料夾參照（`McBopomofo.xcodeproj` 裡的 `SlothE`），建置時原樣複製成 `Contents/Resources/SlothE/`。

- 已提交：`char2id.tsv`、`syl_vocab.tsv`、`syl2legal.bin`、`variants.tsv`、`dec_next_first.tsv`（字表、音節表、合法讀音遮罩、異體字、decoder 首字表），加上 `runtime-manifest.txt`、`MANIFEST.txt`、`NOTICE.txt`、`OpenCC-LICENSE.txt`。
- 不提交（見根目錄 `.gitignore`）：`enc25m.mlmodelc/`、`dec60m.mlmodelc/`、`enc25m_embed_f16.bin`、`dec_tokenizer.json`，合計約 137 MB。
- 取得模型：`hf download workfunction/McBopomofoLM-models --revision v3.1.2 --include "runtime/*" --local-dir /tmp/mcbpmf-lm`，再 `cp -R /tmp/mcbpmf-lm/runtime/ SlothE/Bundle/SlothE/`（DEVELOPMENT.md「自行編譯」）。
- 完整性：app 啟動時依 `runtime-manifest.txt` 逐檔核對大小與 sha256，不符的模型不載入；換了檔案就要重新產生這份清單（`SlothE/prep_resources_v2.py`）。
- 誰用：`Source/SlothE/SlothEEngine.cpp`、`SlothERuntime.mm`；測試 `SlothETests.mm`、`SlothEModelStoreTests.mm`、`V34pTests.swift`。
- 授權：Sloth 模型與 `char2id` 字表是 Apache-2.0，異體字表衍生自 OpenCC（Apache-2.0）；細節見 `NOTICE.txt`。
