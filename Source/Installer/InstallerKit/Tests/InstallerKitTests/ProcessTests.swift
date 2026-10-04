// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation
import Testing

@testable import InstallerKit

@Suite("Child process output")
struct ProcessTests {
    @Test func splitterHandlesChunksAndPartialLines() {
        var splitter = LineSplitter()
        #expect(splitter.feed(Data("one\ntw".utf8)) == ["one"])
        #expect(splitter.feed(Data("o\n\nthree".utf8)) == ["two", ""])
        #expect(splitter.flush() == "three")
        #expect(splitter.flush() == nil)
    }

    @Test func splitterKeepsMultiByteCharactersCutByAReadBoundary() {
        let bytes = Array("小麥注音\n".utf8)
        var splitter = LineSplitter()
        var lines: [String] = []
        for byte in bytes { lines += splitter.feed(Data([byte])) }
        #expect(lines == ["小麥注音"])
    }

    @Test func splitterDropsCarriageReturns() {
        var splitter = LineSplitter()
        #expect(splitter.feed(Data("a\r\nb\r\n".utf8)) == ["a", "b"])
    }

    @Test func streamsLinesFromBothPipesAndReturnsTheStatus() throws {
        let collected = LineTail(limit: 100)
        let errors = LineTail(limit: 100)
        let status = try ProcessRunner.runStreaming(
            "/bin/sh", ["-c", "echo one; echo two >&2; printf tail; exit 3"]
        ) { line in
            (line.isError ? errors : collected).add(line.text)
        }
        #expect(status == 3)
        #expect(collected.snapshot.sorted() == ["one", "tail"])
        #expect(errors.snapshot == ["two"])
    }

    @Test func linesKnowWhichPipeTheyCameFrom() throws {
        let collector = Collector<OutputLine>()
        _ = try ProcessRunner.runStreaming(
            "/bin/sh", ["-c", "echo one; echo two >&2; printf tail"]
        ) { line in
            collector.add(line)
        }
        let lines = collector.all
        #expect(lines.filter { !$0.isError }.map(\.text).sorted() == ["one", "tail"])
        #expect(lines.filter { $0.isError }.map(\.text) == ["two"])
    }

    @Test func childSeesAPipeNotATerminal() throws {
        // `McBopomofoLM install` prints its detailed lines when stdout is not a tty.
        let seen = LineTail(limit: 10)
        _ = try ProcessRunner.runStreaming(
            "/bin/sh", ["-c", "if [ -t 1 ]; then echo tty; else echo pipe; fi"]
        ) { line in
            seen.add(line.text)
        }
        #expect(seen.snapshot == ["pipe"])
    }

    @Test func runCollectsLargeOutputWithoutHanging() throws {
        // More than a pipe buffer on stdout and on stderr.
        let script =
            "BEGIN { for (i = 0; i < 20000; i++) { print \"xxxxxxxxxxxxxxxxxxxxxxxx\"; print \"yyyyyyyyyyyyyyyyyyyy\" > \"/dev/stderr\" } }"
        let result = try ProcessRunner.run("/usr/bin/awk", [script])
        #expect(result.status == 0)
        #expect(result.stdout.split(separator: "\n").count == 20000)
        #expect(result.stderr.split(separator: "\n").count == 20000)
    }
}
