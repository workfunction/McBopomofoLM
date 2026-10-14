// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

/// Names shared by the installer app, the dry run and the tests.
public enum InstallerConstants {
    /// The executable name and the base of the archive name (`McBopomofoLM-r<build>.zip`).
    public static let appName = "McBopomofoLM"
    /// The bundle the installer puts under `~/Library/Input Methods`.
    public static let appBundleName = "McBopomofoLM.app"
    /// The bundle identifier of the input method.
    public static let bundleID = "org.openvanilla.inputmethod.McBopomofoLM"
    /// The only input mode the installer enables (plain Bopomofo stays off, PLAN.md section 22).
    public static let bopomofoModeID = "org.openvanilla.inputmethod.McBopomofoLM.Bopomofo"
    /// The input mode the installer never enables.
    public static let plainBopomofoModeID = "org.openvanilla.inputmethod.McBopomofoLM.PlainBopomofo"
    /// The input source id of the ASCII keyboard the installer switches to before it stops the
    /// running input method.
    public static let asciiKeyboardID = "com.apple.keylayout.ABC"
    /// The arguments of the `install` child: verify the model files, register the input source (v3.4n: it
    /// does not compile the models, so it takes seconds and has no progress to print).
    public static let installArguments = ["install"]
    /// The first macOS version the input method (and so the installer) runs on.
    public static let minimumMajorOSVersion = 15

    /// The real install location. Only the installer app itself ever writes here.
    public static var inputMethodsDirectory: String {
        (NSHomeDirectory() as NSString).appendingPathComponent("Library/Input Methods")
    }

    /// `~/Library/Caches`.
    public static var cachesDirectory: String {
        (NSHomeDirectory() as NSString).appendingPathComponent("Library/Caches")
    }

    /// The Neural Engine compile cache of the input method (v3.3k: the removal deletes it).
    public static var compileCacheDirectory: String {
        (cachesDirectory as NSString).appendingPathComponent(bundleID)
    }

    /// The input method's user data folder (user phrases, settings, the optional diagnostics log).
    public static var userDataDirectory: String {
        (NSHomeDirectory() as NSString)
            .appendingPathComponent("Library/Application Support/McBopomofoLM")
    }
}
