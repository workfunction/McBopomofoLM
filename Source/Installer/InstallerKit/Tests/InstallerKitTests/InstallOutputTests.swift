// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation
import Testing

@testable import InstallerKit

/// The lines are the ones `McBopomofoLM install` prints (Source/InstallCommand.swift in the IME).
@Suite("Install output")
struct InstallOutputTests {
    @Test func knownLinesBecomeEvents() {
        let cases: [(String, InstallOutput.Event)] = [
            (
                "McBopomofoLM: model files verified (size + sha256) in 98 ms.",
                .modelFilesVerified(milliseconds: 98)
            ),
            (
                "  McBopomofoLM: model files verified (size + sha256) in 101 ms.",
                .modelFilesVerified(milliseconds: 101)
            ),
            (
                "McBopomofoLM: ERROR: model files damaged: dec60m.mlmodelc/weights/weight.bin: size 1 instead of 2",
                .modelFilesDamaged(detail: "dec60m.mlmodelc/weights/weight.bin: size 1 instead of 2")
            ),
        ]
        for (line, event) in cases {
            #expect(InstallOutput.parse(line) == event, "\(line)")
        }
    }

    @Test func otherLinesShowNothing() {
        for line in [
            // what `install --compile-now` prints for developers: the installer never asks for it
            "McBopomofoLM: preparing the SlothE models for the Neural Engine (first time: about 3 minutes)...",
            "SlothE decoder (dec60m.mlmodelc):",
            "  decoder t16 ready in 4.2 s",
            "Models ready in 184.9 s: encoder on the Neural Engine, decoder on the Neural Engine.",
            "PROGRESS 0.4210 decoder-t16",
            "McBopomofoLM: both models run on the Neural Engine.",
            "McBopomofoLM: the models are not compiled now; the input method prepares them in the background the first time it starts (about 3 minutes). Until they are loaded the usual candidate selection is used.",
            "Registering input source org.openvanilla.inputmethod.McBopomofoLM at file:///x",
            "McBopomofoLM: model files verified (size + sha256) in x ms.",
            "", "tty",
        ] {
            #expect(InstallOutput.parse(line) == nil, "\(line)")
        }
    }
}
