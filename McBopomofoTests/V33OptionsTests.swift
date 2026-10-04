// McBopomofoLM v3.3b: the menu regrouped for ordinary users, the same options in the preferences
// window, the diagnostic-log switch (D6: default OFF, nothing written and no directory created
// while OFF, opened and closed at run time) and the zh-Hant / English wording.
//
// No model is loaded and nothing is installed or registered: the controller is built without an
// IMK server, the runtimes have no models, and every log lives in a temporary directory.

import Cocoa
import XCTest

@testable import McBopomofo

private let kDiagnosticKey = "SlothEDiagnosticLogEnabled"

/// Strings of one `.lproj` of the app bundle.
private func stringsTable(_ localization: String) throws -> [String: String] {
    let path = try XCTUnwrap(
        Bundle.main.path(forResource: "Localizable", ofType: "strings", inDirectory: nil, forLocalization: localization),
        "no Localizable.strings for \(localization)")
    return try XCTUnwrap(NSDictionary(contentsOfFile: path) as? [String: String])
}

class V33OptionsTestCase: XCTestCase {
    private var savedDefaults: [String: Any?] = [:]
    private let keys = [
        kDiagnosticKey, "SlothERerankEnabled", "SlothEShiftToggleEnabled", "SlothETypoEntryEnabled",
        "SlothEDecoderEnabled", "SlothEPauseGateEnabled", "SlothEDemoteShownCandidate",
    ]
    private var savedHook: (() -> Void)?
    var startLoadingCalls = 0

    override func setUp() {
        super.setUp()
        for key in keys { savedDefaults[key] = UserDefaults.standard.object(forKey: key) }
        for key in keys { UserDefaults.standard.removeObject(forKey: key) }
        savedHook = Preferences.slothEStartLoading
        startLoadingCalls = 0
        Preferences.slothEStartLoading = { [unowned self] in self.startLoadingCalls += 1 }
    }

    override func tearDown() {
        Preferences.slothEStartLoading = savedHook ?? { SlothERuntime.shared.startLoading() }
        for key in keys {
            if let value = savedDefaults[key] ?? nil {
                UserDefaults.standard.set(value, forKey: key)
            } else {
                UserDefaults.standard.removeObject(forKey: key)
            }
        }
        super.tearDown()
    }

    func makeController() throws -> McBopomofoInputMethodController {
        try XCTUnwrap(McBopomofoInputMethodController(server: nil, delegate: nil, client: nil))
    }
}

// MARK: - menu

final class V33MenuTests: V33OptionsTestCase {
    private func title(_ key: String) -> String { NSLocalizedString(key, comment: "") }

    func testThreeOptionsForEverybodyThenAnAdvancedSubmenu() throws {
        let controller = try makeController()
        let menu = try XCTUnwrap(controller.menu())
        let top = menu.items.map { $0.title }
        // The first four entries after the three upstream toggles: the plain options, then "Advanced".
        let first = try XCTUnwrap(top.firstIndex { $0.hasPrefix(title("AI Candidate Selection")) })
        XCTAssertEqual(top[first + 1], title("Tap Shift to Switch Chinese/English"))
        XCTAssertEqual(top[first + 2], title("Offer Syllable Corrections in the Candidate Window"))
        XCTAssertEqual(top[first + 3], title("Advanced"))

        let advanced = try XCTUnwrap(menu.items[first + 3].submenu)
        let actionTitles = advanced.items.filter { $0.action != nil }.map { $0.title }
        XCTAssertEqual(
            actionTitles,
            [
                title("Decoder Sentence Review"), title("Sentence Check After a Pause"),
                title("List Current Word Last in the Candidate Window"), title("Write Diagnostic Log"),
            ])
        // The three developer switches are not in the top level any more.
        for gone in ["Decoder Sentence Review", "Sentence Check After a Pause", "List Current Word Last in the Candidate Window", "Write Diagnostic Log"] {
            XCTAssertFalse(top.contains(title(gone)), "\(gone) is still in the top level")
        }
        // No upstream-style test names left anywhere in the menu.
        for item in menu.items + advanced.items {
            XCTAssertFalse(item.title.contains("SlothE"), item.title)
        }
        // A submenu's items get no target from the framework: ours name the controller.
        for item in advanced.items where item.action != nil {
            XCTAssertTrue(item.target === controller, item.title)
        }
    }

    func testTheStatesFollowThePreferences() throws {
        Preferences.slothERerankEnabled = true
        Preferences.slothEShiftToggleEnabled = true
        Preferences.slothETypoEntryEnabled = false
        Preferences.slothEDecoderEnabled = true
        Preferences.slothEPauseGateEnabled = false
        Preferences.slothEDemoteShownCandidate = true
        Preferences.changeSlothEDiagnosticLogEnabled(false)
        let controller = try makeController()
        let menu = try XCTUnwrap(controller.menu())
        func state(_ key: String) -> NSControl.StateValue? {
            var found: NSControl.StateValue?
            func walk(_ m: NSMenu) {
                for item in m.items {
                    if item.title.hasPrefix(title(key)) { found = item.state }
                    if let sub = item.submenu { walk(sub) }
                }
            }
            walk(menu)
            return found
        }
        XCTAssertEqual(state("AI Candidate Selection"), .on)
        XCTAssertEqual(state("Tap Shift to Switch Chinese/English"), .on)
        XCTAssertEqual(state("Offer Syllable Corrections in the Candidate Window"), .off)
        XCTAssertEqual(state("Decoder Sentence Review"), .on)
        XCTAssertEqual(state("Sentence Check After a Pause"), .off)
        XCTAssertEqual(state("List Current Word Last in the Candidate Window"), .on)
        XCTAssertEqual(state("Write Diagnostic Log"), .off)
    }

    func testTheActionsToggleThePreferences() throws {
        let controller = try makeController()
        Preferences.slothERerankEnabled = true
        controller.toggleSlothERerank(nil)  // off: nothing to load
        XCTAssertFalse(Preferences.slothERerankEnabled)
        XCTAssertEqual(startLoadingCalls, 0)
        controller.toggleSlothERerank(nil)  // on: the models start loading
        XCTAssertTrue(Preferences.slothERerankEnabled)
        XCTAssertEqual(startLoadingCalls, 1)

        XCTAssertFalse(Preferences.slothEDiagnosticLogEnabled, "default OFF")
        controller.toggleSlothEDiagnosticLog(nil)
        XCTAssertTrue(Preferences.slothEDiagnosticLogEnabled)
        controller.toggleSlothEDiagnosticLog(nil)
        XCTAssertFalse(Preferences.slothEDiagnosticLogEnabled)
    }

    func testATitleSuffixFollowsTheLanguage() {
        XCTAssertEqual(McBopomofoInputMethodController.menuTitle("AI 選字", suffix: "（準備中）"), "AI 選字（準備中）")
        XCTAssertEqual(McBopomofoInputMethodController.menuTitle("AI Candidate Selection", suffix: "(preparing)"), "AI Candidate Selection (preparing)")
    }

    func testPlainBopomofoModeShowsNoneOfThem() throws {
        let controller = try makeController()
        controller.keyHandler.inputMode = .plainBopomofo
        let menu = try XCTUnwrap(controller.menu())
        XCTAssertFalse(menu.items.contains { $0.title.hasPrefix(title("AI Candidate Selection")) })
        XCTAssertFalse(menu.items.contains { $0.title == title("Advanced") })
    }
}

// MARK: - preferences window

final class V33PreferencesWindowTests: V33OptionsTestCase {
    func testTheWindowModelSetsTheSamePreferencesAsTheMenu() {
        MainActor.assumeIsolated { checkTheWindowModel() }  // XCTest runs a test method on the main thread
    }

    @MainActor
    private func checkTheWindowModel() {
        let model = PreferencesViewModel()
        Preferences.slothERerankEnabled = false
        model.slothERerankEnabled = true
        XCTAssertTrue(Preferences.slothERerankEnabled)
        XCTAssertEqual(startLoadingCalls, 1, "turning the master switch on starts loading, as the menu does")
        model.slothERerankEnabled = true  // no change, no second load
        XCTAssertEqual(startLoadingCalls, 1)
        model.slothERerankEnabled = false
        XCTAssertFalse(Preferences.slothERerankEnabled)
        XCTAssertEqual(startLoadingCalls, 1)

        model.slothEShiftToggleEnabled = false
        XCTAssertFalse(Preferences.slothEShiftToggleEnabled)
        XCTAssertFalse(model.slothEShiftToggleEnabled)
        model.slothEShiftToggleEnabled = true
        XCTAssertTrue(Preferences.slothEShiftToggleEnabled)

        model.slothETypoEntryEnabled = true
        XCTAssertTrue(Preferences.slothETypoEntryEnabled)
        model.slothETypoEntryEnabled = false
        XCTAssertFalse(Preferences.slothETypoEntryEnabled)

        XCTAssertFalse(model.slothEDiagnosticLogEnabled, "default OFF")
        model.slothEDiagnosticLogEnabled = true
        XCTAssertTrue(Preferences.slothEDiagnosticLogEnabled)
        model.slothEDiagnosticLogEnabled = false
        XCTAssertFalse(Preferences.slothEDiagnosticLogEnabled)
    }

    func testTheNewSettingsPostTheNotificationsTheRuntimesListenTo() {
        var general = 0
        var diagnostic = 0
        let a = NotificationCenter.default.addObserver(forName: .slothEPreferencesDidChange, object: nil, queue: nil) { _ in general += 1 }
        let b = NotificationCenter.default.addObserver(forName: .slothEDiagnosticLogPreferenceDidChange, object: nil, queue: nil) { _ in diagnostic += 1 }
        defer {
            NotificationCenter.default.removeObserver(a)
            NotificationCenter.default.removeObserver(b)
        }
        Preferences.changeSlothEShiftToggleEnabled(false)
        XCTAssertEqual(general, 1)
        XCTAssertEqual(diagnostic, 0)
        Preferences.changeSlothEDiagnosticLogEnabled(true)
        XCTAssertEqual(diagnostic, 1)
        XCTAssertEqual(general, 1, "the diagnostic switch does not make the key handlers re-walk")
    }
}

// MARK: - wording

final class V33WordingTests: V33OptionsTestCase {
    private let usedKeys = [
        "AI Candidate Selection", "Tap Shift to Switch Chinese/English",
        "Offer Syllable Corrections in the Candidate Window", "Advanced", "Decoder Sentence Review",
        "Sentence Check After a Pause", "List Current Word Last in the Candidate Window", "Write Diagnostic Log",
        "McBopomofo LM:", "Diagnostics:", "(loading)", "(model failed to load)",
        "The log records only per-key timings, buffer lengths and the bundle identifier of the app you type in. It never records the text you type.",
        "About McBopomofo…", "McBopomofo Preferences",
    ]

    func testEveryNewStringExistsInEveryLanguage() throws {
        for localization in ["Base", "en", "zh-Hant"] {
            let table = try stringsTable(localization)
            for key in usedKeys {
                XCTAssertNotNil(table[key], "\(localization) lacks \(key)")
            }
        }
    }

    func testZhHantNamesForOrdinaryUsers() throws {
        let zh = try stringsTable("zh-Hant")
        XCTAssertEqual(zh["AI Candidate Selection"], "AI 選字")
        XCTAssertEqual(zh["Tap Shift to Switch Chinese/English"], "單點 Shift 切換中英文")
        XCTAssertEqual(zh["Offer Syllable Corrections in the Candidate Window"], "選字窗提供錯音修正")
        XCTAssertEqual(zh["Advanced"], "進階")
        XCTAssertEqual(zh["Write Diagnostic Log"], "寫入診斷記錄")
        XCTAssertEqual(zh["List Current Word Last in the Candidate Window"], "選字窗把目前的字排最後")
        // The app is called 小麥注音 LM in its own windows.
        XCTAssertEqual(zh["About McBopomofo…"], "關於小麥注音 LM…")
        XCTAssertEqual(zh["McBopomofo Preferences"], "小麥注音 LM 偏好設定")
        XCTAssertEqual(zh["Issues were found in the following user phrase files:"], "以下是使用者自訂詞庫檔案中發現的問題：")
    }

    func testEnglishNamesFollowTheNewName() throws {
        for localization in ["Base", "en"] {
            let table = try stringsTable(localization)
            XCTAssertEqual(table["About McBopomofo…"], "About McBopomofo LM…")
            XCTAssertEqual(table["McBopomofo Preferences"], "McBopomofo LM Preferences")
            XCTAssertEqual(table["AI Candidate Selection"], "AI Candidate Selection")
        }
    }

    func testZhHantUsesTaiwanTermsOnly() throws {
        let zh = try stringsTable("zh-Hant")
        // 候選窗 is 選字窗; 用戶 is 使用者; and the usual mainland-only words never appear.
        let banned = ["候選窗", "用戶", "視頻", "軟件", "信息", "默認", "設置", "菜單", "窗口", "程序", "激活", "登錄", "文件夾", "內存", "鼠標", "網絡", "服務器", "數據"]
        for (key, value) in zh {
            for word in banned {
                XCTAssertFalse(value.contains(word), "zh-Hant '\(key)' contains \(word)")
            }
        }
        // The user-phrase templates written into a new user's folder as well.
        for name in ["template-data", "template-data-plain-bpmf"] {
            let path = try XCTUnwrap(Bundle.main.path(forResource: name, ofType: "txt", inDirectory: nil, forLocalization: "zh-Hant"))
            let text = try String(contentsOfFile: path, encoding: .utf8)
            XCTAssertFalse(text.contains("用戶"), name)
        }
    }

    func testTheMenuNamesAreNotTheDeveloperNamesAnyMore() throws {
        for localization in ["Base", "en", "zh-Hant"] {
            let table = try stringsTable(localization)
            for key in table.keys {
                XCTAssertFalse(key.hasPrefix("SlothE"), "\(localization) still has the old key \(key)")
            }
            for (key, value) in table where key.contains("SlothE") || value.contains("SlothE") {
                XCTFail("\(localization): \(key) = \(value)")
            }
        }
    }
}

// MARK: - the diagnostic log (D6)

final class V33DiagnosticLogTests: V33OptionsTestCase {
    private var root: URL!

    override func setUp() {
        super.setUp()
        root = FileManager.default.temporaryDirectory.appendingPathComponent("v33-diag-\(UUID().uuidString)", isDirectory: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: root)
        super.tearDown()
    }

    /// `<root>/McBopomofoLM/latency.log`, in a directory that does not exist yet.
    private var logPath: String { root.appendingPathComponent("McBopomofoLM").appendingPathComponent("latency.log").path }
    private var logDirectory: String { root.appendingPathComponent("McBopomofoLM").path }

    private func makeApplicationRuntime() -> SlothERuntime {
        SlothERuntime(forApplicationWithResourcePath: "/nonexistent", logPath: logPath)
    }

    private func noteSomething(_ runtime: SlothERuntime) {
        runtime.noteInputModeChange("en", source: "tap")
        runtime.noteBufferStart(withContextLength: 3, reset: "none", selection: "ok", client: "com.example.app", passThrough: 0)
        runtime.noteLegalEntryEvent("offered", slot: 9, candidates: 12, readyMilliseconds: 1.5, trigger: "pause")
        runtime.flushLog()
    }

    private func logText() -> String { (try? String(contentsOfFile: logPath, encoding: .utf8)) ?? "" }

    func testOffByDefaultNothingIsWrittenAndNoDirectoryIsCreated() {
        XCTAssertFalse(Preferences.slothEDiagnosticLogEnabled)
        XCTAssertFalse(SlothERuntime.diagnosticLogPreference)
        let runtime = makeApplicationRuntime()
        XCTAssertFalse(runtime.diagnosticLogEnabled)
        noteSomething(runtime)
        runtime.noteStockWalkMilliseconds(1)
        runtime.beginKeystroke()
        runtime.endKeystroke(withBufferLength: 4, handlerMilliseconds: 1.0, state: "on")
        runtime.flushLog()
        XCTAssertFalse(FileManager.default.fileExists(atPath: logDirectory), "no directory while OFF")
        XCTAssertFalse(FileManager.default.fileExists(atPath: logPath))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path), "nothing under the parent either")
    }

    func testTurningItOnOpensTheLogAtRunTimeAndOffClosesIt() throws {
        let runtime = makeApplicationRuntime()
        noteSomething(runtime)
        XCTAssertFalse(FileManager.default.fileExists(atPath: logDirectory))

        Preferences.changeSlothEDiagnosticLogEnabled(true)  // the runtime follows the notification
        XCTAssertTrue(runtime.diagnosticLogEnabled)
        runtime.flushLog()
        XCTAssertTrue(FileManager.default.fileExists(atPath: logPath), "opened at once: header and build marker")
        XCTAssertTrue(logText().hasPrefix("# McBopomofoLM latency log v4"))
        XCTAssertTrue(logText().contains("# McBopomofoLM v3.3"), "build marker")

        noteSomething(runtime)
        let onText = logText()
        XCTAssertTrue(onText.contains("\nM\t"), onText)
        XCTAssertTrue(onText.contains("com.example.app"))
        let size = try XCTUnwrap(FileManager.default.attributesOfItem(atPath: logPath)[.size] as? Int)

        Preferences.changeSlothEDiagnosticLogEnabled(false)
        XCTAssertFalse(runtime.diagnosticLogEnabled)
        noteSomething(runtime)
        XCTAssertEqual(try XCTUnwrap(FileManager.default.attributesOfItem(atPath: logPath)[.size] as? Int), size, "nothing written while OFF")

        // Closed, not just muted: remove the file and the directory, note more, nothing comes back.
        try FileManager.default.removeItem(atPath: logDirectory)
        noteSomething(runtime)
        XCTAssertFalse(FileManager.default.fileExists(atPath: logDirectory), "a closed log does not reopen by itself")

        // And on again: opened afresh (the file was deleted, so header and marker again).
        Preferences.changeSlothEDiagnosticLogEnabled(true)
        runtime.flushLog()
        XCTAssertTrue(FileManager.default.fileExists(atPath: logPath))
        XCTAssertTrue(logText().hasPrefix("# McBopomofoLM latency log v4"))
        noteSomething(runtime)
        XCTAssertTrue(logText().contains("com.example.app"))
    }

    func testAnApplicationRuntimeStartsAsThePreferenceSaysAndWritesNoTypedText() {
        Preferences.changeSlothEDiagnosticLogEnabled(true)
        let runtime = makeApplicationRuntime()
        XCTAssertTrue(runtime.diagnosticLogEnabled)
        runtime.noteBufferStart(withContextLength: 5, reset: "none", selection: "ok", client: "com.example.típing", passThrough: 1)
        runtime.flushLog()
        let text = logText()
        XCTAssertTrue(text.contains("com.example.t"), "ASCII part of the bundle id")
        XCTAssertFalse(text.contains("í"), "anything but ASCII is dropped from the client id")
        Preferences.changeSlothEDiagnosticLogEnabled(false)
        runtime.flushLog()
    }

    func testARuntimeWithoutAPathNeverLogsWhateverTheSwitchSays() {
        let runtime = SlothERuntime(resourcePath: "/nonexistent", logPath: nil)
        XCTAssertFalse(runtime.diagnosticLogEnabled)
        runtime.diagnosticLogEnabled = true
        XCTAssertFalse(runtime.diagnosticLogEnabled)
        noteSomething(runtime)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
    }

    func testARuntimeMadeWithAPathStillLogsLikeBefore() {
        // The existing tests (and the install flow of earlier versions) pass a path and expect lines.
        let runtime = SlothERuntime(resourcePath: "/nonexistent", logPath: logPath)
        XCTAssertTrue(runtime.diagnosticLogEnabled)
        noteSomething(runtime)
        XCTAssertTrue(logText().contains("com.example.app"))
        runtime.diagnosticLogEnabled = false
        let size = (try? FileManager.default.attributesOfItem(atPath: logPath)[.size] as? Int) ?? -1
        noteSomething(runtime)
        XCTAssertEqual((try? FileManager.default.attributesOfItem(atPath: logPath)[.size] as? Int) ?? -2, size)
    }

    func testTheRealLogPathIsNotTouchedByTheTests() {
        // Under XCTest the shared runtime has no log path at all.
        XCTAssertNil(SlothERuntime.shared.logPath)
        XCTAssertTrue(SlothERuntime.defaultLatencyLogPath.hasSuffix("/McBopomofoLM/latency.log"))
    }
}
