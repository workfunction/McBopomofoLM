// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation
import Testing

@testable import InstallerKit

/// The lines are the ones `McBopomofoLM install` prints (Source/main.swift, SlothERuntime.mm).
@Suite("Install output")
struct InstallOutputTests {
    @Test func knownLinesBecomeEvents() {
        let cases: [(String, InstallOutput.Event)] = [
            (
                "McBopomofoLM: preparing the SlothE models for the Neural Engine (first time: about 3 minutes)...",
                .preparing
            ),
            (
                "Cleared this app's Neural Engine compile cache: the models compile from scratch.",
                .cacheCleared
            ),
            ("SlothE-T 25M encoder (enc25m.mlmodelc):", .encoderStarted),
            ("SlothE decoder (dec60m.mlmodelc):", .decoderStarted),
            ("  encoder L8 ready in 4.6 s", .functionReady(.encoder, name: "L8", seconds: 4.6)),
            ("  decoder t16 ready in 4.2 s", .functionReady(.decoder, name: "t16", seconds: 4.2)),
            (
                "  encoder ON the Neural Engine: loaded in 53.6 s (ANE compile included when not cached)",
                .modelLoaded(.encoder, seconds: 53.6)
            ),
            (
                "  decoder ON the Neural Engine: loaded in 131.3 s",
                .modelLoaded(.decoder, seconds: 131.3)
            ),
            (
                "Models ready in 184.9 s: encoder on the Neural Engine, decoder on the Neural Engine.",
                .modelsReady(seconds: 184.9)
            ),
            ("McBopomofoLM: both models run on the Neural Engine.", .allOnNeuralEngine),
            (
                "McBopomofoLM: WARNING: not every model can run on the Neural Engine; see above. Those models are not used (no encoder = stock McBopomofo; no decoder = encoder-only). The input method retries at every start.",
                .degraded
            ),
        ]
        for (line, event) in cases {
            #expect(InstallOutput.parse(line) == event, "\(line)")
        }
    }

    @Test func otherLinesShowNothing() {
        for line in [
            "  files verified (size + sha256) in 98 ms",
            "  placement (MLComputePlan), estimated cost on the Neural Engine: L8 100.0%, 29.3 s",
            "  latency probe: 0.85 ms per call (limit 10 ms)",
            "Registering input source org.openvanilla.inputmethod.McBopomofoLM at file:///x",
            "", "tty",
        ] {
            #expect(InstallOutput.parse(line) == nil, "\(line)")
        }
    }
}
