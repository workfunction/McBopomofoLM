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

import Cocoa
import InstallerKit

// `McBopomofoInstaller --dry-run <folder> [--simulate-quarantine]` runs the unzip, copy and
// quarantine steps into <folder> and exits, without a window and without touching the system.
switch DryRunOptions.parse(Array(CommandLine.arguments.dropFirst())) {
case .notRequested:
    break
case .usageError(let usage):
    FileHandle.standardError.write(Data((usage + "\n").utf8))
    exit(2)
case .run(let options):
    let code = DryRun.run(
        options: options, resourcePath: Bundle.main.resourcePath ?? "",
        infoDictionary: Bundle.main.infoDictionary ?? [:]
    ) { line in
        FileHandle.standardOutput.write(Data((line + "\n").utf8))
    }
    exit(code)
}

exit(NSApplicationMain(CommandLine.argc, CommandLine.unsafeArgv))
