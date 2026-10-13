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
import InputSourceHelper
import InstallerKit

private let kTargetBin = InstallerConstants.appName
private let kTargetBundle = InstallerConstants.appBundleName
private let kDestinationPartial = "~/Library/Input Methods/"
private let kTargetPartialPath = "~/Library/Input Methods/McBopomofoLM.app"

private let kTranslocationRemovalTickInterval: TimeInterval = 0.5
private let kTranslocationRemovalDeadline: TimeInterval = 60.0

/// How the install ended (before anything is shown).
private enum InstallResult {
    /// Installed. The outcome carries the warnings and the log-out notice, together.
    case done(InstallOutcome)
    case failed(title: String, message: String)
}

/// A step that failed outside InstallerKit.
private struct StepFailure: Error {
    let title: String
    let message: String
}

@objc(AppDelegate)
class AppDelegate: NSWindowController, NSApplicationDelegate {
    @IBOutlet weak private var actionButton: NSButton!
    @IBOutlet weak private var cancelButton: NSButton!
    @IBOutlet private var textView: NSTextView!
    @IBOutlet weak private var progressSheet: NSWindow!
    @IBOutlet weak private var progressIndicator: NSProgressIndicator!
    @IBOutlet weak var welcomeText: NSTextField!
    @IBOutlet weak var agreeText: NSTextField!

    private var removeButton: NSButton?
    private var statusSpinner: NSProgressIndicator?
    /// The model progress panel (v3.3i): a sentence, the percentage and a determinate bar, above
    /// the log. Built in code, like the controls above.
    private var progressPanelStack: NSStackView?
    private var modelProgressBar: NSProgressIndicator?
    private var modelProgressLabel: NSTextField?
    private var modelProgressPercent: NSTextField?
    private var modelProgress = ModelProgressState()
    /// The title of the step that runs now: the sentence while the bar is still indeterminate.
    private var currentStepTitle = ""
    /// The install command exited 0 but said not every model runs on the Neural Engine.
    private var modelsDegraded = false
    /// The install button title and the line under the license, as the xib has them.
    private var installButtonTitle = ""
    private var installAgreementLine = ""

    private let inputSources = SystemInputSources()

    private var archiveUtil: ArchiveUtil?
    private var installingVersion = ""
    private var installingShortVersion = ""
    private var existing: ExistingInstall = .none
    /// Something of McBopomofo LM is on this Mac (see `detectRemovable`).
    private var removable = false

    /// An install or a removal is running: the window must not close and the app must not quit.
    private var busy = false
    private var installed = false

    func runAlertPanel(title: String, message: String, buttonTitle: String) {
        let alert = NSAlert()
        alert.alertStyle = .informational
        alert.messageText = title
        alert.informativeText = message
        alert.addButton(withTitle: buttonTitle)
        alert.runModal()
    }

    // MARK: Launch

    func applicationDidFinishLaunching(_ notification: Notification) {
        if !checkSystemRequirements() {
            return
        }
        guard let installingVersion = Bundle.main.infoDictionary?[kCFBundleVersionKey as String] as? String,
              let versionString = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String else {
            runAlertPanel(
                title: NSLocalizedString("Fatal Error", comment: ""),
                message: NSLocalizedString("No installable packagess found.", comment: ""),
                buttonTitle: NSLocalizedString("Abort", comment: ""))
            NSApp.terminate(nil)
            return
        }
        self.installingVersion = installingVersion
        self.installingShortVersion = versionString
        self.archiveUtil = ArchiveUtil(appName: kTargetBin, targetAppBundleName: kTargetBundle)
        #if DEBUG
        let snapshotDirectory = ProcessInfo.processInfo.environment["MCBPMF_INSTALLER_SNAPSHOT"]
        #else
        let snapshotDirectory: String? = nil
        #endif
        guard snapshotDirectory != nil || archiveUtil?.validateOrQuit() == true else { return }

        cancelButton.nextKeyView = actionButton
        actionButton.nextKeyView = cancelButton
        if let cell = actionButton.cell as? NSButtonCell {
            window?.defaultButtonCell = cell
        }

        var attrStr = NSAttributedString()
        if let rtfPath = Bundle.main.url(forResource: "License", withExtension: "rtf"),
           let rtfData = try? Data(contentsOf: rtfPath),
           let rtf = NSAttributedString(rtf: rtfData, documentAttributes: nil) {
            attrStr = rtf
        }

        let mutableAttrStr = NSMutableAttributedString(attributedString: attrStr)
        mutableAttrStr.addAttribute(.foregroundColor, value: NSColor.controlTextColor, range: NSMakeRange(0, mutableAttrStr.length))
        textView.textStorage?.setAttributedString(mutableAttrStr)
        textView.setSelectedRange(NSMakeRange(0, 0))
        window?.title = String(format: NSLocalizedString("%@ (for version %@, r%@)", comment: ""), window?.title ?? "", versionString, installingVersion)

        installButtonTitle = actionButton.title
        installAgreementLine = agreeText.stringValue
        makeRemoveButton()
        makeStatusSpinner()
        makeModelProgressPanel()
        refreshExistingInstall()

        #if DEBUG
        if let directory = snapshotDirectory {
            writeSnapshots(to: directory)
            exit(0)
        }
        #endif

        window?.center()
        window?.orderFront(self)
        NSApp.activate(ignoringOtherApps: true)
    }

    /// Intel Macs and macOS before 15 get a message and the installer quits. (On macOS 14 and
    /// earlier LaunchServices usually refuses the installer first, through LSMinimumSystemVersion;
    /// this is the second line.)
    private func checkSystemRequirements() -> Bool {
        let issues = SystemCheck.currentIssues()
        if issues.isEmpty { return true }
        var lines: [String] = []
        for issue in issues {
            switch issue {
            case .notAppleSilicon:
                lines.append(NSLocalizedString("McBopomofo LM needs a Mac with Apple silicon (M1 or later). Intel Macs are not supported.", comment: ""))
            case .osTooOld(let current):
                lines.append(String(format: NSLocalizedString("McBopomofo LM needs macOS %d or later. This Mac runs macOS %@. Please update macOS first, then open the installer again.", comment: ""), InstallerConstants.minimumMajorOSVersion, current))
            }
        }
        runAlertPanel(
            title: NSLocalizedString("This Mac cannot run McBopomofo LM", comment: ""),
            message: lines.joined(separator: "\n\n"),
            buttonTitle: NSLocalizedString("Quit", comment: ""))
        NSApp.terminate(nil)
        return false
    }

    /// What is installed now, by CFBundleVersion: an older build is an upgrade.
    private func detectExisting() -> ExistingInstall {
        let path = (kTargetPartialPath as NSString).expandingTildeInPath
        return VersionCheck.classify(
            installerBuild: installingVersion,
            installedBuild: VersionCheck.readVersions(ofAppAt: path).build,
            appExists: FileManager.default.fileExists(atPath: path))
    }

    private func refreshExistingInstall() {
        existing = detectExisting()
        removable = detectRemovable()
        applyExistingInstall()
    }

    /// Anything of McBopomofo LM left on this Mac: the app, a registered input source or a running
    /// process. Removal is offered while any is there, also after the app was deleted by hand.
    private func detectRemovable() -> Bool {
        UninstallPlan.isRemovable(
            appExists: FileManager.default.fileExists(atPath: (kTargetPartialPath as NSString).expandingTildeInPath),
            inputSourceCount: InputSourceFlow.sources(inputSources).count,
            processRunning: !NSRunningApplication.runningApplications(withBundleIdentifier: InstallerConstants.bundleID).isEmpty)
    }

    private func applyExistingInstall() {
        switch existing {
        case .older:
            actionButton.title = NSLocalizedString("Agree and Upgrade", comment: "")
            agreeText.stringValue = NSLocalizedString("By upgrading the software I agree to the terms above.", comment: "")
        case .same, .newer, .unreadable:
            actionButton.title = NSLocalizedString("Agree and Reinstall", comment: "")
            agreeText.stringValue = NSLocalizedString("By reinstalling the software I agree to the terms above.", comment: "")
        case .none:
            actionButton.title = installButtonTitle
            agreeText.stringValue = installAgreementLine
        }
        removeButton?.isHidden = !(existing.isInstalled || removable)
    }

    // MARK: Extra controls (built in code so both xibs stay as they are)

    private func makeRemoveButton() {
        guard let content = window?.contentView else { return }
        let button = NSButton(
            title: NSLocalizedString("Remove", comment: ""), target: self,
            action: #selector(removeAction(_:)))
        button.bezelStyle = .rounded
        button.translatesAutoresizingMaskIntoConstraints = false
        button.isHidden = !(existing.isInstalled || removable)
        content.addSubview(button)
        // The button keeps its natural width; the welcome line gives way if the two ever collide.
        button.setContentCompressionResistancePriority(.required, for: .horizontal)
        welcomeText.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)
        NSLayoutConstraint.activate([
            button.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -20),
            button.centerYAnchor.constraint(equalTo: welcomeText.centerYAnchor),
            button.widthAnchor.constraint(greaterThanOrEqualToConstant: 64),
            welcomeText.trailingAnchor.constraint(lessThanOrEqualTo: button.leadingAnchor, constant: -8),
        ])
        removeButton = button
    }

    private func makeStatusSpinner() {
        guard let content = window?.contentView else { return }
        let spinner = NSProgressIndicator()
        spinner.style = .spinning
        spinner.controlSize = .small
        spinner.isIndeterminate = true
        spinner.isDisplayedWhenStopped = false
        spinner.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(spinner)
        NSLayoutConstraint.activate([
            spinner.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 20),
            spinner.centerYAnchor.constraint(equalTo: agreeText.centerYAnchor),
        ])
        statusSpinner = spinner
    }

    /// The progress panel sits between the welcome line and the log. It lives in a stack view
    /// whose only view is hidden until an install starts, which makes it take no room: the window
    /// looks as the xib has it until then.
    private func makeModelProgressPanel() {
        guard let content = window?.contentView, let scroll = textView.enclosingScrollView else { return }
        // The two xibs fix the log's top differently: Base ties it to the welcome line (which is
        // pinned to the top), zh-Hant ties it to the welcome line and to the window's top (so the
        // welcome line is pinned only through the log). Both are replaced by "log below the panel
        // below the welcome line", with the welcome line kept where it was.
        let isScrollTop: (NSLayoutConstraint, AnyObject?) -> Bool = { c, item in
            ((c.firstItem as AnyObject?) === item && c.firstAttribute == .top)
                || ((c.secondItem as AnyObject?) === item && c.secondAttribute == .top)
        }
        let oldTops = content.constraints.filter { isScrollTop($0, scroll) }
        guard !oldTops.isEmpty else { return }
        var gapBelowWelcome: CGFloat = 8
        var scrollTopFromWindowTop: CGFloat?
        for c in oldTops {
            let other = ((c.firstItem as AnyObject?) === scroll ? c.secondItem : c.firstItem) as AnyObject?
            if other === welcomeText {
                gapBelowWelcome = c.constant
            } else if other === content {
                scrollTopFromWindowTop = c.constant
            }
        }
        let welcomeHasTop = content.constraints.contains { isScrollTop($0, welcomeText) }

        let label = NSTextField(labelWithString: "")
        label.font = NSFont.systemFont(ofSize: NSFont.systemFontSize)
        label.lineBreakMode = .byTruncatingTail
        label.translatesAutoresizingMaskIntoConstraints = false
        label.setContentCompressionResistancePriority(.defaultLow, for: .horizontal)

        let percent = NSTextField(labelWithString: "")
        percent.font = NSFont.monospacedDigitSystemFont(ofSize: NSFont.systemFontSize, weight: .regular)
        percent.alignment = .right
        percent.translatesAutoresizingMaskIntoConstraints = false
        percent.setContentHuggingPriority(.required, for: .horizontal)
        percent.setContentCompressionResistancePriority(.required, for: .horizontal)

        let bar = NSProgressIndicator()
        bar.style = .bar
        bar.minValue = 0
        bar.maxValue = 1
        bar.isIndeterminate = true
        bar.translatesAutoresizingMaskIntoConstraints = false

        let panel = NSView()
        panel.translatesAutoresizingMaskIntoConstraints = false
        panel.isHidden = true
        for view in [label, percent, bar] { panel.addSubview(view) }
        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: panel.leadingAnchor),
            label.topAnchor.constraint(equalTo: panel.topAnchor, constant: 10),
            percent.leadingAnchor.constraint(greaterThanOrEqualTo: label.trailingAnchor, constant: 8),
            percent.trailingAnchor.constraint(equalTo: panel.trailingAnchor),
            percent.firstBaselineAnchor.constraint(equalTo: label.firstBaselineAnchor),
            bar.leadingAnchor.constraint(equalTo: panel.leadingAnchor),
            bar.trailingAnchor.constraint(equalTo: panel.trailingAnchor),
            bar.topAnchor.constraint(equalTo: label.bottomAnchor, constant: 6),
            bar.bottomAnchor.constraint(equalTo: panel.bottomAnchor),
        ])

        let stack = NSStackView(views: [panel])
        stack.orientation = .vertical
        stack.alignment = .width
        stack.spacing = 0
        stack.detachesHiddenViews = true
        // Exactly as tall as what it holds (nothing while hidden): the log takes all the rest.
        stack.setContentHuggingPriority(.required, for: .vertical)
        stack.setContentCompressionResistancePriority(.required, for: .vertical)
        stack.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(stack)

        NSLayoutConstraint.deactivate(oldTops)
        var added = [
            stack.leadingAnchor.constraint(equalTo: scroll.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: scroll.trailingAnchor),
            stack.topAnchor.constraint(equalTo: welcomeText.bottomAnchor),
            scroll.topAnchor.constraint(equalTo: stack.bottomAnchor, constant: gapBelowWelcome),
        ]
        if !welcomeHasTop, let fromTop = scrollTopFromWindowTop {
            // zh-Hant: the welcome line sat (log top - gap - its height) below the window's top.
            added.append(welcomeText.topAnchor.constraint(
                equalTo: content.topAnchor, constant: fromTop - gapBelowWelcome - welcomeText.frame.height))
        }
        NSLayoutConstraint.activate(added)
        progressPanelStack = stack
        modelProgressBar = bar
        modelProgressLabel = label
        modelProgressPercent = percent
    }

    private func setModelProgressVisible(_ visible: Bool) {
        progressPanelStack?.arrangedSubviews.first?.isHidden = !visible
    }

    /// Shows `modelProgress`: indeterminate with the step's title until the first PROGRESS line,
    /// then a determinate bar, one sentence for what runs now and the percentage.
    private func refreshModelProgress() {
        guard let bar = modelProgressBar, let label = modelProgressLabel,
              let percent = modelProgressPercent
        else { return }
        let sentence: String
        switch modelProgress.phase {
        case .waiting:
            sentence = currentStepTitle
        case .compiling:
            sentence = NSLocalizedString("Compiling the models for the Neural Engine (about 3 minutes)…", comment: "")
        case .placementCheck:
            sentence = NSLocalizedString("Checking that the models run on the Neural Engine…", comment: "")
        case .finishing:
            sentence = NSLocalizedString("Almost done…", comment: "")
        case .done:
            sentence = modelsDegraded
                ? NSLocalizedString("The models are prepared, but not all of them run on the Neural Engine.", comment: "")
                : NSLocalizedString("The models are ready.", comment: "")
        case .failed:
            sentence = NSLocalizedString("Install Failed", comment: "")
        }
        label.stringValue = sentence
        label.textColor = modelProgress.phase == .failed ? NSColor.systemRed : NSColor.labelColor
        bar.setAccessibilityLabel(sentence)
        if modelProgress.isIndeterminate {
            percent.stringValue = ""
            bar.isIndeterminate = true
            bar.startAnimation(self)
        } else {
            bar.stopAnimation(self)
            bar.isIndeterminate = false
            bar.doubleValue = modelProgress.fraction ?? 0
            percent.stringValue = String(format: "%d%%", modelProgress.percent)
        }
        percent.textColor = modelProgress.phase == .failed ? NSColor.systemRed : NSColor.secondaryLabelColor
    }

    /// While something runs the status line moves right to make room for the spinner.
    private func setBusy(_ value: Bool) {
        busy = value
        cancelButton.isEnabled = !value
        actionButton.isEnabled = !value
        removeButton?.isEnabled = !value
        if value {
            removeButton?.isHidden = true
            statusSpinner?.startAnimation(self)
        } else {
            statusSpinner?.stopAnimation(self)
        }
        if let content = window?.contentView,
           let leading = content.constraints.first(where: { ($0.firstItem as AnyObject?) === agreeText && $0.firstAttribute == .leading }) {
            leading.constant = value ? 42 : 20
        }
    }

    // MARK: Install

    @IBAction func agreeAndInstallAction(_ sender: AnyObject) {
        if installed {
            // installation was successful, user clicked the "Close Installer" button
            endAppWithDelay()
            return
        }
        guard !busy else { return }
        if case .newer(let installedBuild) = existing, !confirmDowngrade(installedBuild: installedBuild) {
            return
        }
        startInstall()
    }

    private func confirmDowngrade(installedBuild: String) -> Bool {
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = NSLocalizedString("A newer version is already installed", comment: "")
        alert.informativeText = String(format: NSLocalizedString("The installed McBopomofo LM is r%@, newer than the r%@ in this installer. Replace it with the older version?", comment: ""), installedBuild, installingVersion)
        alert.addButton(withTitle: NSLocalizedString("Cancel", comment: ""))
        alert.addButton(withTitle: NSLocalizedString("Replace", comment: ""))
        return alert.runModal() == .alertSecondButtonReturn
    }

    private func startInstall() {
        setBusy(true)
        if existing.isInstalled {
            welcomeText.stringValue = NSLocalizedString("Upgrading McBopomofo LM…", comment: "")
        } else {
            welcomeText.stringValue = NSLocalizedString("Installing McBopomofo LM…", comment: "")
        }
        textView.textStorage?.setAttributedString(NSAttributedString())
        modelProgress = ModelProgressState()
        modelsDegraded = false
        currentStepTitle = ""
        setModelProgressVisible(true)
        refreshModelProgress()
        Task { @MainActor in
            let result = await self.runInstall()
            self.finishInstall(result)
        }
    }

    private func killOldProcess() async {
        _ = try? await background { try ProcessRunner.run("/usr/bin/killall", ["-9", kTargetBin]) }
    }

    /// Lets the run loop turn between two looks at the input source list.
    private func pause() {
        RunLoop.current.run(until: Date().addingTimeInterval(0.1))
    }

    private func recycle(_ path: String) async throws {
        _ = try await NSWorkspace.shared.recycle([URL(fileURLWithPath: path)])
    }

    private func runInstall() async -> InstallResult {
        guard let source = archiveUtil?.payloadSource() else {
            return .failed(
                title: NSLocalizedString("Fatal Error", comment: ""),
                message: NSLocalizedString("No installable packagess found.", comment: ""))
        }
        let installedPath = (kTargetPartialPath as NSString).expandingTildeInPath
        let payload = PayloadInstaller(
            source: source,
            destinationDirectory: (kDestinationPartial as NSString).expandingTildeInPath,
            stagingDirectory: (NSTemporaryDirectory() as NSString).appendingPathComponent(UUID().uuidString),
            expectedBuild: installingVersion, expectedShortVersion: installingShortVersion)
        // The old version is moved aside, not trashed, until the new one is in.
        let backup = OldVersionBackup(
            installedApp: installedPath,
            backupDirectory: (NSTemporaryDirectory() as NSString).appendingPathComponent("McBopomofoLM-previous-" + UUID().uuidString))

        // Look again: the window may have been open for a while.
        existing = detectExisting()
        let steps = InstallPlan.steps(
            existing: existing,
            oldProcessTranslocated: existing.isInstalled && appBundleTranslocatedToARandomizedPath(kTargetPartialPath),
            dryRun: false)

        // The same orchestration the tests run, with the real system behind it.
        let environment = InstallEnvironment(
            inputSources: inputSources,
            pause: { self.pause() },
            offMain: { work in try await self.background(work) },
            killOldProcess: { await self.killOldProcess() },
            waitForTranslocationRemoval: { await self.waitForTranslocationRemoval() },
            moveToTrash: { try await self.recycle($0.path) },
            willRun: { step, index, total in
                self.showStep(self.title(for: step), index: index, total: total)
                if step == .runInstall {
                    self.appendLog(NSLocalizedString("Preparing the models for the Neural Engine. The first time takes about 3 minutes; please keep this window open.", comment: ""))
                }
            },
            // Delivered on the main actor by the orchestrator, in the order the child printed them.
            onEvent: { event in self.show(event) },
            // The install command exited 0: the bar goes to 100 %.
            onInstallCommandSucceeded: { status in
                self.modelsDegraded = (status == .degraded)
                self.modelProgress.markSucceeded()
                self.refreshModelProgress()
            })
        do {
            return .done(
                try await InstallOrchestrator.run(
                    steps: steps, payload: payload, backup: backup, isUpgrade: existing.isInstalled,
                    environment: environment))
        } catch let failure as InstallFailure {
            return describe(failure)
        } catch {
            return describe(error)
        }
    }

    /// The failure as text, with what became of the old version: put back, or where it is.
    private func describe(_ failure: InstallFailure) -> InstallResult {
        guard case .failed(let title, var message) = describe(failure.underlying) else {
            return describe(failure.underlying)
        }
        switch failure.restore {
        case .notNeeded:
            break
        case .restored:
            message += "\n\n" + NSLocalizedString("The previous version was put back.", comment: "")
        case .failed(let error):
            message += "\n\n" + describeText(error)
        }
        return .failed(title: title, message: message)
    }

    private func show(_ event: InstallOutput.Event) {
        func seconds(_ value: Double) -> Int { Int(value.rounded()) }
        switch event {
        case .progress:
            // About twice a second: the bar moves, the log does not get a line for each.
            modelProgress.apply(event)
            refreshModelProgress()
        case .preparing, .functionReady:
            break
        case .cacheCleared:
            appendLog(NSLocalizedString("The compile cache of the models was cleared; they are prepared from scratch.", comment: ""), monospaced: true)
        case .encoderStarted:
            appendLog(NSLocalizedString("Preparing the encoder model…", comment: ""), monospaced: true)
        case .decoderStarted:
            appendLog(NSLocalizedString("Preparing the decoder model…", comment: ""), monospaced: true)
        case .modelLoaded(.encoder, let value):
            appendLog(String(format: NSLocalizedString("Encoder model ready (%d s).", comment: ""), seconds(value)), monospaced: true)
        case .modelLoaded(.decoder, let value):
            appendLog(String(format: NSLocalizedString("Decoder model ready (%d s).", comment: ""), seconds(value)), monospaced: true)
        case .modelsReady(let value):
            appendLog(String(format: NSLocalizedString("The models are ready (%d s).", comment: ""), seconds(value)), monospaced: true)
        case .allOnNeuralEngine:
            appendLog(NSLocalizedString("Both models run on the Neural Engine.", comment: ""), monospaced: true)
        case .degraded:
            appendLog(NSLocalizedString("Warning: not every model can run on the Neural Engine.", comment: ""), monospaced: true)
        }
    }

    /// Waits for Gatekeeper to let go of the translocated copy of the old IME (upstream's flow):
    /// up to a minute, with the sheet. False when it is still there afterwards.
    private func waitForTranslocationRemoval() async -> Bool {
        guard window?.responds(to: #selector(NSWindow.beginSheet(_:completionHandler:))) ?? false else {
            return true
        }
        progressIndicator.startAnimation(self)
        window?.beginSheet(progressSheet, completionHandler: nil)
        defer {
            window?.endSheet(progressSheet)
            progressIndicator.stopAnimation(self)
        }
        let start = Date()
        while Date().timeIntervalSince(start) < kTranslocationRemovalDeadline {
            if appBundleTranslocatedToARandomizedPath(kTargetPartialPath) == false {
                progressIndicator.doubleValue = 1.0
                return true
            }
            try? await Task.sleep(nanoseconds: UInt64(kTranslocationRemovalTickInterval * 1_000_000_000))
        }
        return false
    }

    private func finishInstall(_ result: InstallResult) {
        setBusy(false)
        switch result {
        case .done(let outcome):
            showSuccess(outcome)
        case .failed(let title, let message):
            modelProgress.markFailed()  // the panel shows the error state behind the alert
            refreshModelProgress()
            runAlertPanel(title: title, message: message, buttonTitle: NSLocalizedString("Cancel", comment: ""))
            endAppWithDelay()
        }
    }

    private func showSuccess(_ outcome: InstallOutcome) {
        let headlineAttr = [
            NSAttributedString.Key.font : NSFont.boldSystemFont(ofSize: NSFont.systemFontSize * 1.3),
            NSAttributedString.Key.foregroundColor : NSColor.textColor
        ]
        let bodyAttr = [
            NSAttributedString.Key.font : NSFont.systemFont(ofSize: NSFont.systemFontSize),
            NSAttributedString.Key.foregroundColor : NSColor.textColor
        ]
        let needsReading = !outcome.warnings.isEmpty || outcome.needsRelogin
        let headline = !needsReading
            ? NSLocalizedString("Installation Successful", comment: "")
            : NSLocalizedString("Installed, with a warning", comment: "")
        let message = NSMutableAttributedString(string: headline, attributes: headlineAttr)
        func paragraph(_ text: String) {
            message.append(NSAttributedString(string: "\n\n" + text, attributes: bodyAttr))
        }
        paragraph(NSLocalizedString("McBopomofo is ready to use.", comment: ""))
        // Both the log-out notice and the warnings are shown: neither hides the other.
        if outcome.needsRelogin {
            paragraph(NSLocalizedString("McBopomofo is upgraded, but please log out or reboot for the new version to be fully functional.", comment: ""))
        }
        for warning in outcome.warnings {
            switch warning {
            case .degradedModels:
                paragraph(NSLocalizedString("Not every model can run on the Neural Engine of this Mac. McBopomofo LM works, but without the models that cannot run there (without the encoder it behaves like McBopomofo; without the decoder it uses the encoder only). It tries again every time it starts.", comment: ""))
            case .modeNotEnabled:
                paragraph(NSLocalizedString("Input method may not be fully enabled. Please enable it through System Preferences > Keyboard > Input Sources.", comment: ""))
            case .otherModesStillOn:
                paragraph(NSLocalizedString("Another McBopomofo LM mode is still turned on. Turn it off in System Settings > Keyboard > Input Sources if you do not want it.", comment: ""))
            }
        }
        paragraph(NSLocalizedString("If McBopomofo LM is not in the input menu, log out and log in again, or add it in System Settings > Keyboard > Input Sources.", comment: ""))
        textView.textStorage?.setAttributedString(message)

        installed = true

        welcomeText.isHidden = true
        cancelButton.isHidden = true
        removeButton?.isHidden = true
        actionButton.title = NSLocalizedString("Close Installer", comment: "")
        actionButton.isEnabled = true

        // System Settings now asks the user whether to activate the IME; auto-close the installer
        // (only when nothing needs reading).
        if !needsReading {
            scheduleAutoClose()
        } else {
            agreeText.stringValue = ""
        }
    }

    // MARK: Remove

    @objc func removeAction(_ sender: AnyObject) {
        guard !busy, !installed else { return }
        let appExists = FileManager.default.fileExists(atPath: (kTargetPartialPath as NSString).expandingTildeInPath)
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = NSLocalizedString("Remove McBopomofo LM?", comment: "")
        alert.informativeText = appExists
            ? NSLocalizedString("This turns off the input method and moves McBopomofo LM from the Input Methods folder to the Trash. Your user phrases and settings stay where they are unless you choose otherwise in the next question.", comment: "")
            : NSLocalizedString("The McBopomofo LM app is gone, but the input method is still registered or running. This turns it off. Your user phrases and settings stay where they are unless you choose otherwise in the next question.", comment: "")
        alert.addButton(withTitle: NSLocalizedString("Remove", comment: ""))
        alert.addButton(withTitle: NSLocalizedString("Cancel", comment: ""))
        guard alert.runModal() == .alertFirstButtonReturn else { return }

        setBusy(true)
        welcomeText.stringValue = NSLocalizedString("Removing McBopomofo LM…", comment: "")
        textView.textStorage?.setAttributedString(NSAttributedString())
        Task { @MainActor in
            await self.runRemoval()
        }
    }

    private func runRemoval() async {
        let installedPath = (kTargetPartialPath as NSString).expandingTildeInPath
        let userData = InstallerConstants.userDataDirectory
        let appExists = FileManager.default.fileExists(atPath: installedPath)
        let steps = UninstallPlan.steps(
            userDataExists: FileManager.default.fileExists(atPath: userData),
            appExists: appExists)
        var userDataTrashed = false
        do {
            try await InstallRunner.runAsync(steps, willRun: { step, index, total in
                self.showStep(title(for: step), index: index, total: total)
            }, perform: { step in
                switch step {
                case .selectASCIIInputSource:
                    // Nothing is turned off, killed or trashed while the user is typing in it.
                    guard InputSourceFlow.leaveMcBopomofoLM(inputSources, pause: pause) else {
                        throw StepFailure(
                            title: NSLocalizedString("Remove Failed", comment: ""),
                            message: NSLocalizedString("McBopomofo LM is the input method in use and the installer could not switch away from it. Please choose another input method (for example ABC) in the input menu, then open the installer again. Nothing was changed.", comment: ""))
                    }
                case .disableInputSources:
                    let outcome = InputSourceFlow.disableAll(inputSources, pause: pause)
                    appendLog(String(format: NSLocalizedString("Input sources turned off: %d", comment: ""), outcome.disabled))
                    // The app is not killed or trashed while a source is still on.
                    guard outcome.isComplete else {
                        throw StepFailure(
                            title: NSLocalizedString("Remove Failed", comment: ""),
                            message: NSLocalizedString("Some input sources of McBopomofo LM could not be turned off. Turn them off in System Settings > Keyboard > Input Sources, then open the installer again. Nothing else was changed.", comment: ""))
                    }
                case .stopProcess:
                    await killOldProcess()
                case .moveToTrash:
                    do {
                        try await recycle(installedPath)
                    } catch {
                        throw StepFailure(
                            title: NSLocalizedString("Remove Failed", comment: ""),
                            message: String(format: NSLocalizedString("Cannot remove the existing version at %@, error: %@", comment: ""), installedPath, error.localizedDescription))
                    }
                case .askAboutUserData:
                    if askAboutUserData(path: userData) == .moveToTrash {
                        do {
                            try await recycle(userData)
                            userDataTrashed = true
                        } catch {
                            throw StepFailure(
                                title: NSLocalizedString("Remove Failed", comment: ""),
                                message: String(format: NSLocalizedString("Cannot move %@ to the Trash, error: %@", comment: ""), userData, error.localizedDescription))
                        }
                    }
                }
            })
        } catch {
            setBusy(false)
            if case .failed(let title, let message) = describe((error as? InstallRunner.Failure)?.underlying ?? error) {
                runAlertPanel(title: title, message: message, buttonTitle: NSLocalizedString("Cancel", comment: ""))
            }
            endAppWithDelay()
            return
        }
        setBusy(false)
        var message = appExists
            ? NSLocalizedString("McBopomofo LM has been moved to the Trash. If it still shows up in the input menu, please log out or restart your Mac.", comment: "")
            : NSLocalizedString("McBopomofo LM has been turned off. If it still shows up in the input menu, please log out or restart your Mac.", comment: "")
        if !userDataTrashed && FileManager.default.fileExists(atPath: userData) {
            message += "\n\n" + String(format: NSLocalizedString("Your user phrases and settings were kept in %@.", comment: ""), (userData as NSString).abbreviatingWithTildeInPath)
        }
        runAlertPanel(title: NSLocalizedString("McBopomofo LM Removed", comment: ""), message: message, buttonTitle: NSLocalizedString("OK", comment: ""))
        endAppWithDelay()
    }

    /// The user data folder is kept unless the user says otherwise: "Keep" is the default button.
    private func askAboutUserData(path: String) -> UserDataChoice {
        let alert = NSAlert()
        alert.alertStyle = .informational
        alert.messageText = NSLocalizedString("Also delete your user data?", comment: "")
        alert.informativeText = String(format: NSLocalizedString("Your user phrases and settings are in %@. Keep them if you may install McBopomofo LM again.", comment: ""), (path as NSString).abbreviatingWithTildeInPath)
        alert.addButton(withTitle: NSLocalizedString("Keep", comment: ""))
        alert.addButton(withTitle: NSLocalizedString("Move to Trash", comment: ""))
        return alert.runModal() == .alertSecondButtonReturn ? .moveToTrash : .keep
    }

    // MARK: Progress

    private func title(for step: InstallStep) -> String {
        switch step {
        case .selectASCIIInputSource: return NSLocalizedString("Switching to the ABC keyboard…", comment: "")
        case .moveOldAside: return NSLocalizedString("Setting the old version aside…", comment: "")
        case .stopOldProcess: return NSLocalizedString("Stopping the old version…", comment: "")
        case .waitForTranslocationRemoval: return NSLocalizedString("Waiting for the old version to stop…", comment: "")
        case .unzip: return NSLocalizedString("Unpacking McBopomofo LM…", comment: "")
        case .copy: return NSLocalizedString("Copying to ~/Library/Input Methods…", comment: "")
        case .removeQuarantine: return NSLocalizedString("Removing the download mark…", comment: "")
        case .verifyQuarantineFree: return NSLocalizedString("Checking the installed files…", comment: "")
        case .runInstall: return NSLocalizedString("Registering the input method and preparing its models…", comment: "")
        case .enableBopomofoMode: return NSLocalizedString("Turning on the Bopomofo input mode…", comment: "")
        case .discardOldBackup: return NSLocalizedString("Moving the old version to the Trash…", comment: "")
        }
    }

    private func title(for step: UninstallStep) -> String {
        switch step {
        case .selectASCIIInputSource: return NSLocalizedString("Switching to the ABC keyboard…", comment: "")
        case .disableInputSources: return NSLocalizedString("Turning off the input method…", comment: "")
        case .stopProcess: return NSLocalizedString("Stopping McBopomofo LM…", comment: "")
        case .moveToTrash: return NSLocalizedString("Moving McBopomofo LM to the Trash…", comment: "")
        case .askAboutUserData: return NSLocalizedString("Your user data", comment: "")
        }
    }

    private func showStep(_ title: String, index: Int, total: Int) {
        currentStepTitle = title
        if modelProgress.isIndeterminate, progressPanelStack?.arrangedSubviews.first?.isHidden == false {
            refreshModelProgress()
        }
        agreeText.stringValue = String(format: NSLocalizedString("Step %1$d of %2$d: %3$@", comment: ""), index + 1, total, title)
        appendLog(String(format: "[%d/%d] %@", index + 1, total, title), bold: true)
    }

    private func appendLog(_ text: String, bold: Bool = false, monospaced: Bool = false) {
        guard let storage = textView.textStorage else { return }
        let font: NSFont
        if monospaced {
            font = NSFont.monospacedSystemFont(ofSize: NSFont.smallSystemFontSize, weight: .regular)
        } else if bold {
            font = NSFont.boldSystemFont(ofSize: NSFont.systemFontSize)
        } else {
            font = NSFont.systemFont(ofSize: NSFont.systemFontSize)
        }
        storage.append(NSAttributedString(string: text + "\n", attributes: [.font: font, .foregroundColor: NSColor.textColor]))
        textView.scrollToEndOfDocument(nil)
    }

    /// The failure as text for the user, in their language. The error's English `description` is
    /// only written to the log.
    private func describe(_ error: Error) -> InstallResult {
        if let failure = error as? StepFailure {
            return .failed(title: failure.title, message: failure.message)
        }
        return .failed(title: NSLocalizedString("Install Failed", comment: ""), message: describeText(error))
    }

    private func describeText(_ error: Error) -> String {
        guard let installerError = error as? InstallerError else {
            return error.localizedDescription
        }
        NSLog("McBopomofoLM installer: %@", installerError.description)
        func status(_ value: Int32) -> String {
            String(format: NSLocalizedString("Exit status: %d", comment: ""), value)
        }
        func folder(_ path: String) -> String {
            String(format: NSLocalizedString("Folder: %@", comment: ""), (path as NSString).abbreviatingWithTildeInPath)
        }
        let headline: String
        let detail: String?
        switch installerError {
        case .cannotCreateDirectory(let path, _):
            headline = NSLocalizedString("Cannot unpack McBopomofo LM from the installer. Please download the installer again.", comment: "")
            detail = folder(path)
        case .unzipFailed(_, let code, _):
            headline = NSLocalizedString("Cannot unpack McBopomofo LM from the installer. Please download the installer again.", comment: "")
            detail = status(code)
        case .missingAppInArchive:
            headline = NSLocalizedString("Cannot unpack McBopomofo LM from the installer. Please download the installer again.", comment: "")
            detail = nil
        case .payloadMismatch(.linkEscapesBundle):
            headline = NSLocalizedString("The input method inside this installer contains a link that points outside of it, so nothing was installed. Please download the installer again.", comment: "")
            detail = nil
        case .payloadMismatch(let problem):
            headline = NSLocalizedString("The input method inside this installer does not match the installer's version. Please download the installer again.", comment: "")
            switch problem {
            case .buildVersion(let found, let expected):
                detail = String(format: NSLocalizedString("Version found: r%@, expected: r%@", comment: ""), found ?? "?", expected)
            case .shortVersion(let found, let expected):
                detail = String(format: NSLocalizedString("Version found: %@, expected: %@", comment: ""), found ?? "?", expected)
            default:
                detail = nil
            }
        case .copyFailed(_, let destination, let code, _):
            headline = NSLocalizedString("Cannot copy the file to the destination.", comment: "")
            detail = folder(destination) + " (" + status(code) + ")"
        case .destinationExists(let path):
            headline = NSLocalizedString("Cannot copy the file to the destination.", comment: "")
            detail = folder(path)
        case .quarantineRemovalFailed(_, let code, _):
            headline = NSLocalizedString("The download mark could not be removed from the installed files, so the input method would not start.", comment: "")
            detail = status(code)
        case .quarantineMarkFailed(_, let code, _):
            headline = NSLocalizedString("The download mark could not be removed from the installed files, so the input method would not start.", comment: "")
            detail = status(code)
        case .quarantineCheckIncomplete(_, let code, _):
            headline = NSLocalizedString("The installed files could not be checked completely, so the installer cannot tell whether the input method would start.", comment: "")
            detail = status(code)
        case .quarantineRemains(let paths):
            headline = NSLocalizedString("The download mark could not be removed from the installed files, so the input method would not start.", comment: "")
            detail = String(format: NSLocalizedString("%d item(s) still carry the mark, for example: %@", comment: ""), paths.count, paths.first ?? "")
        case .installCommandFailed(let code, _):
            headline = NSLocalizedString("McBopomofo LM was copied, but could not be registered as an input method.", comment: "")
            detail = status(code)
        case .cannotLeaveMcBopomofoLM:
            headline = NSLocalizedString("McBopomofo LM is the input method in use and the installer could not switch away from it. Please choose another input method (for example ABC) in the input menu, then open the installer again. Nothing was changed.", comment: "")
            detail = nil
        case .cannotMoveOldAside(let path, _):
            headline = NSLocalizedString("The installed version could not be moved aside, so nothing was changed.", comment: "")
            detail = folder(path)
        case .cannotRestoreOld(let backup, _):
            headline = NSLocalizedString("The previous version could not be put back automatically. You can find it here and move it back to ~/Library/Input Methods:", comment: "")
            detail = (backup as NSString).abbreviatingWithTildeInPath
        case .refusedDestination:
            headline = NSLocalizedString("Cannot activate the input method.", comment: "")
            detail = nil
        }
        return detail.map { headline + "\n\n" + $0 } ?? headline
    }

    /// Runs blocking work (a child process, file copies) off the main thread.
    private func background<T>(_ work: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    continuation.resume(returning: try work())
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    #if DEBUG
    /// Debug builds only. MCBPMF_INSTALLER_SNAPSHOT=<folder> draws the window in each state into
    /// PNG files and quits, without reading or touching anything on the system. The state of the
    /// first picture comes from MCBPMF_INSTALLER_SNAPSHOT_EXISTING = none | older | same.
    private func writeSnapshots(to directory: String) {
        guard let window = window, let content = window.contentView else { return }
        try? FileManager.default.createDirectory(atPath: directory, withIntermediateDirectories: true)
        func snap(_ name: String) {
            window.layoutIfNeeded()
            window.displayIfNeeded()
            RunLoop.current.run(until: Date().addingTimeInterval(1.2))  // let the bar finish its fill animation
            window.layoutIfNeeded()
            window.displayIfNeeded()
            guard let rep = content.bitmapImageRepForCachingDisplay(in: content.bounds) else { return }
            content.cacheDisplay(in: content.bounds, to: rep)
            let url = URL(fileURLWithPath: directory).appendingPathComponent(name + ".png")
            try? rep.representation(using: .png, properties: [:])?.write(to: url)
        }
        switch ProcessInfo.processInfo.environment["MCBPMF_INSTALLER_SNAPSHOT_EXISTING"] ?? "none" {
        case "older": existing = .older(installed: "2510")
        case "same": existing = .same(installed: installingVersion)
        default: existing = .none
        }
        applyExistingInstall()
        window.orderFront(nil)
        snap("1-license")

        setBusy(true)
        welcomeText.stringValue = NSLocalizedString(existing.isInstalled ? "Upgrading McBopomofo LM…" : "Installing McBopomofo LM…", comment: "")
        textView.textStorage?.setAttributedString(NSAttributedString())
        modelProgress = ModelProgressState()
        modelsDegraded = false
        setModelProgressVisible(true)
        refreshModelProgress()
        let steps: [InstallStep] = [.unzip, .copy, .removeQuarantine, .verifyQuarantineFree, .runInstall, .enableBopomofoMode]
        for (index, step) in steps.enumerated() where index < 5 {
            showStep(title(for: step), index: index, total: steps.count)
        }
        appendLog(NSLocalizedString("Preparing the models for the Neural Engine. The first time takes about 3 minutes; please keep this window open.", comment: ""))
        snap("2a-installing-waiting")  // before the first PROGRESS line: indeterminate
        show(.encoderStarted)
        show(.progress(ModelProgress(fraction: 0.05, step: "encoder-L64")))
        show(.modelLoaded(.encoder, seconds: 53.6))
        show(.decoderStarted)
        show(.progress(ModelProgress(fraction: 0.42, step: "decoder-t32")))
        snap("2-installing")
        show(.progress(ModelProgress(fraction: 0.71, step: "placement-decoder-t96")))
        snap("2c-placement-check")
        modelProgress.markFailed()
        refreshModelProgress()
        snap("5-failed")
        modelProgress = ModelProgressState()
        show(.progress(ModelProgress(fraction: 0.64, step: "decoder-n32")))
        modelProgress.markSucceeded()
        refreshModelProgress()

        setBusy(false)
        showSuccess(InstallOutcome())
        snap("3-done")
        modelsDegraded = true
        refreshModelProgress()
        showSuccess(InstallOutcome(warnings: [.degradedModels, .otherModesStillOn], needsRelogin: true))
        snap("4-done-with-warnings")
    }
    #endif

    // MARK: Quitting

    func endAppWithDelay() {
        DispatchQueue.main.asyncAfter(deadline: DispatchTime.now() + 0.1) {
            NSApp.terminate(self)
        }
    }

    @IBAction func cancelAction(_ sender: AnyObject) {
        NSApp.terminate(self)
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if busy {
            runAlertPanel(
                title: NSLocalizedString("Please wait", comment: ""),
                message: NSLocalizedString("The installer is still working. Quitting now could leave McBopomofo LM half installed. Please wait until it finishes.", comment: ""),
                buttonTitle: NSLocalizedString("OK", comment: ""))
            return .terminateCancel
        }
        return .terminateNow
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        if busy {
            NSSound.beep()
            return false
        }
        return true
    }

    func windowWillClose(_ Notification: Notification) {
        NSApp.terminate(self)
    }

    func updateAutoClosingMessage(seconds: Int) {
        agreeText.stringValue = String.localizedStringWithFormat(
            NSLocalizedString("The installer will close itself in %d second(s).", comment: ""),
            seconds)
    }

    func scheduleAutoClose() {
        var counter = 60
        updateAutoClosingMessage(seconds: counter)

        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] timer in
            counter -= 1
            if counter == 0 {
                timer.invalidate()
                self?.endAppWithDelay()
            } else {
                self?.updateAutoClosingMessage(seconds: counter)
            }
        }
    }
}
