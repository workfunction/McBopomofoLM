# scripts/

發佈安裝程式用的 shell 腳本。

- `build-installer.sh <McBopomofoLM.app> <輸出資料夾>`：檢查 IME 與安裝程式的版本號一致、簽章有效；把 IME 壓成 `Source/Installer/NotarizedArchives/McBopomofoLM-r<build>.zip`，建置 Release 的 McBopomofoInstaller，輸出 `McBopomofoLM-Installer-<版本>.zip`。結果是 ad-hoc 簽署，沒有 Apple 公證。
- `check-build-installer-guards.sh`：`build-installer.sh` 清理規則的自我檢查。它在暫存資料夾裡用假的 xcodebuild、codesign、lipo 跑完整個流程，不碰真的 repo；全部通過會印出 OK，失敗時結束碼為 1。
- 選用的環境變數：`DEVELOPER_DIR`（指定 Xcode）、`SOURCE_PACKAGES_DIR`（已解析好的 Swift 套件，可離線建置）、`KEEP_ARCHIVE=1`（建置後保留 zip）。
- 誰用：DEVELOPMENT.md「發佈檔案與核對」；`McBopomofoTests/V33ReleaseTests.swift` 檢查它依賴的版本號規則；`Source/Installer/.gitignore` 對應它放進 `NotarizedArchives/` 的 zip。
- 改了 `build-installer.sh` 之後，請重跑 `check-build-installer-guards.sh`。
- 授權：同 repo（MIT）。
