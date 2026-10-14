// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

/// What `McBopomofoLM install` prints when stdout is not a terminal (see Source/InstallCommand.swift in the IME),
/// as events the installer can show in the user's language. Lines that are not recognised yield no event, so the
/// window never shows raw English tool output.
///
/// Since v3.4n `install` no longer compiles the models (the input method does, in the background, at its first
/// start), so there is no progress to report any more: the only facts are whether the model files are intact.
public enum InstallOutput {
    public enum Event: Equatable, Sendable {
        /// "McBopomofoLM: model files verified (size + sha256) in 98 ms."
        case modelFilesVerified(milliseconds: Double)
        /// "McBopomofoLM: ERROR: model files damaged: <first problem>". The command then exits with
        /// status 1 before it registers anything.
        case modelFilesDamaged(detail: String)
    }

    public static func parse(_ line: String) -> Event? {
        let text = line.trimmingCharacters(in: .whitespaces)
        if let m = match(#"^McBopomofoLM: model files verified \(size \+ sha256\) in ([0-9.]+) ms\.$"#, text),
            m.count == 2, let milliseconds = Double(m[1])
        {
            return .modelFilesVerified(milliseconds: milliseconds)
        }
        let damagedPrefix = "McBopomofoLM: ERROR: model files damaged: "
        if text.hasPrefix(damagedPrefix) {
            return .modelFilesDamaged(detail: String(text.dropFirst(damagedPrefix.count)))
        }
        return nil
    }

    private static func match(_ pattern: String, _ text: String) -> [String]? {
        guard let regex = try? NSRegularExpression(pattern: pattern),
            let result = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text))
        else { return nil }
        return (0..<result.numberOfRanges).map {
            Range(result.range(at: $0), in: text).map { String(text[$0]) } ?? ""
        }
    }
}
