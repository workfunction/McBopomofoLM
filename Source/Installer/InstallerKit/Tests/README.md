# Source/Installer/InstallerKit/Tests/

InstallerKit（安裝程式中不依賴 AppKit 的邏輯）的單元測試，使用 Swift Testing，共 16 個檔案。

- 跑法：`cd Source/Installer/InstallerKit && swift test`（DEVELOPMENT.md「發佈檔案與核對」）。這些測試不在 Xcode 的 `McBopomofo` scheme 裡，也不在目前的 CI workflow 裡。
- 涵蓋：版本比較與封存檔查找、安裝順序（`OrchestratorTests`）、備份舊版、移除隔離標記、`--dry-run`、輸入來源註冊流程、模型資料夾與編譯快取、路徑安全、本地化字串等。
- `Fixtures.swift` 提供暫存資料夾（`TempDir`），測試的檔案操作都在暫存資料夾裡進行。
- `LocalizationTests` 以 `#filePath` 定位 `Source/Installer/*.lproj`；搬動安裝程式的資料夾時要同步修改。
- 測試對象在 `../Sources/InstallerKit`，由 Xcode 專案的 McBopomofoInstaller target 以本機套件引用。
- 授權：MIT，見根目錄 LICENSE.txt（各檔案開頭有標示）。
