# SlothE/ThirdParty/opencc/

- 內容：OpenCC 的 `TWVariants.txt`、`HKVariants.txt`（異體字對照表），加上 Apache-2.0 授權文字（`LICENSE.txt`）和 `NOTICE.txt`。
- 用途：`SlothE/Bundle/SlothE/variants.tsv`（異體字分組，會進 app）是從這兩張表整理出來的；這裡留原始表當出處。app、測試和腳本都不讀這個資料夾。
- 來源：https://github.com/BYVoid/OpenCC （Apache-2.0）。
- `LICENSE.txt` 跟 `SlothE/Bundle/SlothE/OpenCC-LICENSE.txt` 內容完全相同；後者會隨 app 一起發佈。
- 從原始表整理出 `variants.tsv` 的腳本不在這個 repo 裡（在作者的離線研究環境）。要更新時，請放入新版原始表並同步更新 `NOTICE.txt`。
