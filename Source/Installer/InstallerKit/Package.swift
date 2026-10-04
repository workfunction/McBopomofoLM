// swift-tools-version:6.0
// InstallerKit: the testable, AppKit-free logic of the McBopomofoLM installer
// (version compare, archive lookup, install order, quarantine removal, dry run).

import PackageDescription

let package = Package(
    name: "InstallerKit",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "InstallerKit", targets: ["InstallerKit"])
    ],
    targets: [
        .target(name: "InstallerKit"),
        .testTarget(name: "InstallerKitTests", dependencies: ["InstallerKit"]),
    ]
)
