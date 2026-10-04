// Copyright (c) 2026 and onwards The McBopomofo Authors. MIT License, see LICENSE.txt.

import Foundation

/// Why this Mac cannot run McBopomofoLM.
public enum SystemIssue: Equatable {
    /// An Intel Mac: the input method and its models need Apple silicon.
    case notAppleSilicon
    /// macOS older than the supported version (`current` is "major.minor.patch").
    case osTooOld(current: String)
}

public enum SystemCheck {
    public static func issues(
        appleSilicon: Bool, os: OperatingSystemVersion,
        minimumMajor: Int = InstallerConstants.minimumMajorOSVersion
    ) -> [SystemIssue] {
        var result: [SystemIssue] = []
        if !appleSilicon {
            result.append(.notAppleSilicon)
        }
        if os.majorVersion < minimumMajor {
            result.append(
                .osTooOld(current: "\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)"))
        }
        return result
    }

    /// True on Apple silicon, also for an x86_64 slice running under Rosetta (`hw.optional.arm64`
    /// reports the hardware). Missing key (an Intel Mac) or 0 is false.
    public static func isAppleSilicon() -> Bool {
        var value: Int32 = 0
        var size = MemoryLayout<Int32>.size
        guard sysctlbyname("hw.optional.arm64", &value, &size, nil, 0) == 0 else { return false }
        return value == 1
    }

    public static func currentIssues() -> [SystemIssue] {
        issues(appleSilicon: isAppleSilicon(), os: ProcessInfo.processInfo.operatingSystemVersion)
    }
}
