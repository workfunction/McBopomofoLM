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
import Foundation
import InstallerKit

/// Finds the input method the installer carries: the zip under
/// Resources/NotarizedArchives/$appName-r$bundleVersion.zip (the lookup rules live in
/// InstallerKit's ArchiveLocator, where they are tested), or, for a developer build, the app
/// bundle itself in Resources.
struct ArchiveUtil {
    var appName: String
    var targetAppBundleName: String

    init(appName: String, targetAppBundleName: String) {
        self.appName = appName
        self.targetAppBundleName = targetAppBundleName
    }

    func resolve() -> ArchiveLocator.Resolution? {
        guard let resourcePath = Bundle.main.resourcePath,
            let build = Bundle.main.infoDictionary?[kCFBundleVersionKey as String] as? String
        else { return nil }
        return ArchiveLocator.resolve(
            resourcePath: resourcePath, appName: appName, bundleName: targetAppBundleName,
            build: build)
    }

    /// What the install steps unpack, or nil when the installer is built wrongly.
    func payloadSource() -> PayloadInstaller.Source? {
        switch resolve() {
        case .archive(let path): return .archive(path: path)
        case .devBundle(let path): return .appBundle(path: path)
        case .failure, .none: return nil
        }
    }

    /// Alerts and quits when the installer carries no usable input method (a build mistake, not
    /// something a user can fix but by downloading the installer again). Returns true when the
    /// installer is sound.
    func validateOrQuit() -> Bool {
        guard let resolution = resolve() else { return false }
        guard case .failure(let failure) = resolution else { return true }
        NSLog("McBopomofoLM installer: no usable input method inside: %@", String(describing: failure))
        let alert = NSAlert()
        alert.alertStyle = .informational
        alert.messageText = NSLocalizedString("This installer is incomplete", comment: "")
        alert.informativeText = NSLocalizedString("The input method inside this installer is missing or does not belong to it, so nothing can be installed. Please download the installer again.", comment: "")
        alert.addButton(withTitle: NSLocalizedString("Quit", comment: ""))
        alert.runModal()
        NSApp.terminate(nil)
        return false
    }
}
