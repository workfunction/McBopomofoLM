// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

/// What `McBopomofoLM install` prints when stdout is not a terminal (see Source/main.swift and
/// SlothERuntime.loadForInstall in the IME), as events the installer can show in the user's
/// language. Lines that are not recognised yield no event, so the window never shows raw
/// English tool output.
public enum InstallOutput {
    public enum Model: Equatable, Sendable {
        case encoder
        case decoder
    }

    public enum Event: Equatable, Sendable {
        /// "McBopomofoLM: preparing the SlothE models for the Neural Engine ..."
        case preparing
        /// The compile cache of the models was cleared; they compile from scratch.
        case cacheCleared
        /// "SlothE-T 25M encoder (enc25m.mlmodelc):"
        case encoderStarted
        /// "SlothE decoder (dec60m.mlmodelc):"
        case decoderStarted
        /// "  encoder L8 ready in 4.6 s" / "  decoder t16 ready in 4.2 s"
        case functionReady(Model, name: String, seconds: Double)
        /// "  encoder ON the Neural Engine: loaded in 53.6 s (...)"
        case modelLoaded(Model, seconds: Double)
        /// "Models ready in 184.9 s: ..."
        case modelsReady(seconds: Double)
        /// "McBopomofoLM: both models run on the Neural Engine."
        case allOnNeuralEngine
        /// "McBopomofoLM: WARNING: not every model can run on the Neural Engine; ..." The input
        /// method still works but without the models that cannot run there.
        case degraded
    }

    public static func parse(_ line: String) -> Event? {
        let text = line.trimmingCharacters(in: .whitespaces)
        if text.hasPrefix("McBopomofoLM: preparing the SlothE models") { return .preparing }
        if text.hasPrefix("Cleared this app's Neural Engine compile cache") { return .cacheCleared }
        if text.hasPrefix("SlothE-T") && text.contains("encoder") { return .encoderStarted }
        if text.hasPrefix("SlothE decoder") { return .decoderStarted }
        if text == "McBopomofoLM: both models run on the Neural Engine." {
            return .allOnNeuralEngine
        }
        if text.hasPrefix("McBopomofoLM: WARNING: not every model can run on the Neural Engine") {
            return .degraded
        }
        if let m = match(#"^(encoder|decoder) (\S+) ready in ([0-9.]+) s"#, text), m.count == 4,
            let seconds = Double(m[3])
        {
            return .functionReady(model(m[1]), name: m[2], seconds: seconds)
        }
        if let m = match(#"^(encoder|decoder) ON the Neural Engine: loaded in ([0-9.]+) s"#, text),
            m.count == 3, let seconds = Double(m[2])
        {
            return .modelLoaded(model(m[1]), seconds: seconds)
        }
        if let m = match(#"^Models ready in ([0-9.]+) s"#, text), m.count == 2,
            let seconds = Double(m[1])
        {
            return .modelsReady(seconds: seconds)
        }
        return nil
    }

    private static func model(_ word: String) -> Model { word == "encoder" ? .encoder : .decoder }

    private static func match(_ pattern: String, _ text: String) -> [String]? {
        guard let regex = try? NSRegularExpression(pattern: pattern),
            let result = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text))
        else { return nil }
        return (0..<result.numberOfRanges).map {
            Range(result.range(at: $0), in: text).map { String(text[$0]) } ?? ""
        }
    }
}
