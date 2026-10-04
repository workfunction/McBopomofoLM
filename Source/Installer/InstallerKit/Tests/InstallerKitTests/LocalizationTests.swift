// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation
import Testing

/// The installer's strings: every key the code asks for exists in zh-Hant and en, the two files
/// agree on keys and format specifiers, and the zh-Hant text keeps to Taiwan wording.
@Suite("Installer strings")
struct LocalizationTests {
    /// .../Source/Installer, found from this file's path.
    private static let installerDirectory: String = {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<4 { url.deleteLastPathComponent() }
        return url.path
    }()

    private func strings(_ language: String) throws -> [String: String] {
        let path = Self.installerDirectory + "/\(language).lproj/Localizable.strings"
        let plist = try #require(NSDictionary(contentsOfFile: path) as? [String: String])
        return plist
    }

    private func keysUsedInCode() throws -> Set<String> {
        var keys: Set<String> = []
        let regex = try NSRegularExpression(pattern: #"NSLocalizedString\("((?:[^"\\]|\\.)*)""#)
        for file in [
            "AppDelegate.swift", "ArchiveUtil.swift", "InputSourceControl.swift", "main.swift",
        ] {
            let source = try String(
                contentsOfFile: Self.installerDirectory + "/" + file, encoding: .utf8)
            let range = NSRange(source.startIndex..., in: source)
            for match in regex.matches(in: source, range: range) {
                if let r = Range(match.range(at: 1), in: source) {
                    keys.insert(String(source[r]).replacingOccurrences(of: "\\\"", with: "\""))
                }
            }
        }
        return keys
    }

    private func specifiers(_ text: String) -> [String] {
        let regex = try! NSRegularExpression(pattern: #"%(?:\d+\$)?[@dsf]"#)
        let range = NSRange(text.startIndex..., in: text)
        return regex.matches(in: text, range: range).map {
            // "%1$@" and "%@" both count as an object argument.
            String(text[Range($0.range, in: text)!]).last.map(String.init) ?? ""
        }.sorted()
    }

    @Test func everyKeyTheCodeUsesExistsInBothLanguages() throws {
        let en = try strings("en")
        let zh = try strings("zh-Hant")
        // The auto-close message lives in Localizable.stringsdict.
        let keys = try keysUsedInCode().subtracting([
            "The installer will close itself in %d second(s)."
        ])
        #expect(keys.count > 40)
        for key in keys {
            #expect(en[key] != nil, "en is missing: \(key)")
            #expect(zh[key] != nil, "zh-Hant is missing: \(key)")
        }
    }

    @Test func bothLanguagesHaveTheSameKeysAndFormatSpecifiers() throws {
        let en = try strings("en")
        let zh = try strings("zh-Hant")
        #expect(Set(en.keys) == Set(zh.keys))
        for (key, value) in zh {
            #expect(
                specifiers(value) == specifiers(key), "format specifiers differ in zh-Hant: \(key)")
            #expect(
                specifiers(en[key] ?? "") == specifiers(key),
                "format specifiers differ in en: \(key)")
        }
    }

    @Test func chineseTextUsesTheProductNameAndTaiwanWording() throws {
        let zh = try strings("zh-Hant")
        // PRC terms and simplified characters must not appear.
        let banned = ["軟件", "信息", "默認", "視頻", "文件夾", "設置", "登錄", "软件", "信息", "设置", "文件夹", "登录"]
        for (key, value) in zh {
            for word in banned {
                #expect(!value.contains(word), "\(word) in zh-Hant: \(key)")
            }
            // The product is 小麥注音 LM, never the bare old name.
            if value.contains("小麥注音") {
                #expect(value.contains("小麥注音 LM"), "name without LM: \(key)")
            }
        }
        #expect(zh["Remove"] == "移除")
    }

    @Test func englishTextCallsTheProductMcBopomofoLM() throws {
        let en = try strings("en")
        for (key, value) in en where value.contains("McBopomofo") {
            #expect(value.contains("McBopomofo LM"), "name without LM: \(key)")
        }
    }

    // MARK: Hard-coded English

    /// Lines that hand a string literal straight to something the user reads. Empty literals and
    /// the format-only `[%d/%d]` do not count.
    static func hardCodedUserText(in source: String) -> [String] {
        let sinks = try! NSRegularExpression(
            pattern:
                #"((messageText|informativeText|stringValue|title|message|buttonTitle|withTitle)\s*[:=]\s*"[^"]|appendLog\(\s*"[^"]|\.failed\(\s*title:\s*"[^"])"#
        )
        var found: [String] = []
        for line in source.split(separator: "\n", omittingEmptySubsequences: true) {
            let text = String(line)
            if text.contains("NSLocalizedString") { continue }
            if sinks.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil {
                found.append(text.trimmingCharacters(in: .whitespaces))
            }
        }
        return found
    }

    @Test func theLintCatchesHardCodedEnglish() {
        let bad = """
            alert.messageText = "Internal Error"
            alert.informativeText = "devMode installer, expected archive name"
            alert.addButton(withTitle: "Terminate")
            appendLog("Installing")
            runAlertPanel(title: "Oops", message: "x", buttonTitle: "OK")
            """
        #expect(Self.hardCodedUserText(in: bad).count == 5)
        let good = """
            alert.messageText = NSLocalizedString("Internal Error", comment: "")
            agreeText.stringValue = ""
            appendLog(String(format: "[%d/%d] %@", 1, 2, title), bold: true)
            """
        #expect(Self.hardCodedUserText(in: good).isEmpty)
    }

    @Test func noUserVisibleTextInTheInstallerIsHardCoded() throws {
        for file in [
            "AppDelegate.swift", "ArchiveUtil.swift", "InputSourceControl.swift", "main.swift",
        ] {
            let source = try String(
                contentsOfFile: Self.installerDirectory + "/" + file, encoding: .utf8)
            #expect(
                Self.hardCodedUserText(in: source).isEmpty,
                "\(file): \(Self.hardCodedUserText(in: source))")
        }
    }

    @Test func theEnglishTextOfAnErrorIsNeverShown() throws {
        // InstallerError.description and String(describing:) are English; only NSLog may use them.
        for file in ["AppDelegate.swift", "ArchiveUtil.swift"] {
            let source = try String(
                contentsOfFile: Self.installerDirectory + "/" + file, encoding: .utf8)
            for line in source.split(separator: "\n")
            where line.contains(".description") || line.contains("String(describing:") {
                #expect(line.contains("NSLog"), "\(file): \(line)")
            }
        }
    }
}
