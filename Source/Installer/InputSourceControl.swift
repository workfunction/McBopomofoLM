// Copyright (c) 2022 and onwards The McBopomofo Authors.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

import Carbon
import Cocoa
import InputSourceHelper
import InstallerKit

/// The Text Input Sources service behind `InputSourceOperations`: the only place the installer
/// calls TIS. Everything runs on the main thread and only ever on the user's own session; the
/// installer's tests use a fake and never reach this file.
@MainActor
struct SystemInputSources: InputSourceOperations {
    private func string(_ source: TISInputSource, _ key: CFString) -> String? {
        guard let pointer = TISGetInputSourceProperty(source, key) else { return nil }
        return Unmanaged<CFString>.fromOpaque(pointer).takeUnretainedValue() as String
    }

    private func info(_ source: TISInputSource) -> InputSourceInfo? {
        guard let id = string(source, kTISPropertyInputSourceID) else { return nil }
        return InputSourceInfo(
            id: id, bundleID: string(source, kTISPropertyBundleID),
            modeID: string(source, kTISPropertyInputModeID),
            enabled: InputSourceHelper.inputSourceEnabled(for: source))
    }

    private func source(id: String) -> TISInputSource? {
        InputSourceHelper.inputSource(for: id)
    }

    func allSources() -> [InputSourceInfo] {
        InputSourceHelper.allInstalledInputSources().compactMap(info)
    }

    func currentSource() -> InputSourceInfo? {
        guard let current = TISCopyCurrentKeyboardInputSource()?.takeRetainedValue() else {
            return nil
        }
        return info(current)
    }

    func select(id: String) -> Bool {
        guard let source = source(id: id) else { return false }
        return TISSelectInputSource(source) == noErr
    }

    func selectASCIICapable() -> Bool {
        guard let ascii = TISCopyCurrentASCIICapableKeyboardInputSource()?.takeRetainedValue()
        else { return false }
        return TISSelectInputSource(ascii) == noErr
    }

    func enable(id: String) -> Bool {
        guard let source = source(id: id) else { return false }
        return InputSourceHelper.enable(inputSource: source)
    }

    func disable(id: String) -> Bool {
        guard let source = source(id: id) else { return false }
        return InputSourceHelper.disable(inputSource: source)
    }
}
