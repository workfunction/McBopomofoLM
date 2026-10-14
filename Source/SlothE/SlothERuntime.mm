// SlothE-T runtime for the McBopomofoLM side-by-side test build.

#import "SlothERuntime.h"

#include <errno.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <stdio.h>
#include <sys/file.h>
#include <unistd.h>

#include "SlothECoreML.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

static NSString *const kLatencyLogHeader =
    @"# McBopomofoLM latency log v4 (Core ML / ANE build). No typed text is recorded; buf_len = number of readings.\n"
    @"# K = one key event handled by the input method:\n"
    @"#   K time buf_len handler_ms walk_ms inwalk_ms prov_ms fwd rerank_ms settle settle_ms lm\n"
    @"#   walk_ms = stock walk; inwalk_ms = exact re-pick during the key; prov_ms = provisional re-pick with the\n"
    @"#   previous pass (anti-flicker); fwd = queued / none; settle = none / ready / waited / partial / timeout\n"
    @"#   (commit or candidate window waiting up to 30 ms for encoder+decoder); lm = on / off / loading / none\n"
    @"#   (no usable model: see the R lines) / plain\n"
    @"# A = one background pass (SlothE-T encoder, then the decoder) finished:\n"
    @"#   A time buf_len queue_ms fwd_ms dec_ms dec_calls pins inwalk_ms e2e_ms result decoder\n"
    @"#   e2e_ms = key event to result applied; result = applied / unchanged / stale / skipped / deferred / consumed;\n"
    @"#   decoder = off / none (no node to check) / scored / stale (abandoned, buffer changed) / failed\n"
    @"# R = model runtime after a load attempt (encoder, then decoder):\n"
    @"#   R time model runtime reason load_ms plan_ane_pct probe_ms plan_ms fn_ane_pct\n"
    @"#   runtime = ane / none; reason = ok / integrity / missing / load_error / not_on_ane / probe_slow / plan_failed /\n"
    @"#   macos / vocab / probe_failed / no_encoder; load_ms includes the ANE compile when its cache is cold;\n"
    @"#   plan_ane_pct = lowest share of estimated cost on the Neural Engine over the model's functions (MLComputePlan);\n"
    @"#   gate: encoder >= 99, decoder >= 80 per function; fn_ane_pct = each function's share, e.g. L8:100.0,L16:100.0\n"
    @"#   model decoder.t32 / .t64 / .t96 = a decoder function loaded on first need (v2.1);\n"
    @"#   decoder.n16 ... n96 = a next-token function for the correction window (v3.1b, on the first illegal reading)\n"
    @"# P = prewarm: first key of a new buffer after the models idled > 1 s; one dummy call on each model:\n"
    @"#   P time idle_ms enc_ms dec_ms\n"
    @"# C = a new buffer started (v3.1a carried context; no text): C time ctx_len reset sr client pass\n"
    @"#   ctx_len = chars carried from earlier commits; reset = what cleared them since the previous buffer\n"
    @"#   (none / activate / deactivate / commit_composition / key / idle / moved / no_selection);\n"
    @"#   sr = client selection vs where our last insert left it (ok / mismatch / na; slow = the client took\n"
    @"#   > 20 ms and is not asked again); client = bundle id\n"
    @"# X = a correction window (illegal reading, v3.1b) closed; no text: X time n_cands chosen ready_ms source\n"
    @"#   chosen = rank picked (1-based) or esc; ready_ms = illegal reading to model scores (empty = not ready);\n"
    @"#   source = the list shown: dn (decoder next-token + encoder) / t1 (T1 rule: no left text, no next-token\n"
    @"#   function, or not ready within 50 ms; an open window is never refreshed)\n"
    @"# G = the typing-pause decoder gate ran (v3.2a; a full-sentence pass ~150 ms after the last key); no text:\n"
    @"#   G time buf_len requests dec_calls dec_cached no_decision pins changed_chars fwd_ms dec_ms e2e_ms result\n"
    @"#   requests = nodes checked; dec_calls / dec_cached = model calls / answered from the score cache; no_decision =\n"
    @"#   requests the decoder could not answer (text too long for its longest function, function not loaded, call\n"
    @"#   failed); pins = decisions of this pass; changed_chars = chars of the shown text that differ from before the\n"
    @"#   result was applied; e2e_ms = pause start to applied (empty = not applied); result = applied / stale (a newer\n"
    @"#   key came) / skipped (already stale at start) / deferred (a window or another state was up) / failed\n"
    @"# Y = the correction entry for a legal-but-wrong syllable (v3.2b; page 1's last slot); no text:\n"
    @"#   Y time event slot n_cands ready_ms trigger\n"
    @"#   event = offered (the entry was in a candidate window that opened) / chosen (it was picked) / not_ready (a\n"
    @"#   window opened on a syllable whose entry was not computed yet, so it was left out); slot = its 1-based place\n"
    @"#   in the list (empty for not_ready); n_cands = list length; ready_ms = trigger to scores (empty = not ready);\n"
    @"#   trigger = pause (150 ms idle, syllable at the cursor) / arrow (cursor moved) / open (computed for the next window)\n"
    @"# M = the input mode changed (v3.2c; a single Shift tap toggles Chinese / English); no text: M time mode source\n"
    @"#   mode = zh / en (the mode now); source = tap / tap_reading (the tap also turned a Zhuyin reading in progress into\n"
    @"#   letters) / reset (English mode was on and the input method changed its input mode, or the Shift-toggle switch\n"
    @"#   was turned off, or (v3.2i) a client was activated while another window's last Shift tap had left Chinese mode;\n"
    @"#   since v3.2i English mode is global for the process and activate / deactivate alone no longer reset it)\n"
    @"# C line, v3.2g: it has one more field after client (added in v3.2e): pass = the number of printable keys (Space,\n"
    @"#   keypad characters) the input method left to the app in the Empty state since the previous C line. They are\n"
    @"#   counted always, but carried into ctx_len only when the client's selection can confirm the insertion (v3.2f; a\n"
    @"#   key the app ignored then shows as sr=mismatch / reset=moved). A count only: no key, no text. Older C lines\n"
    @"#   have no pass. v3.2g: reset also has unverified_backspace = Backspace with an empty buffer while the context\n"
    @"#   had gaps (a key or line break the app inserted that the context does not hold): the whole context is dropped.\n"
    @"# H = the Shift-tap mode HUD was placed (v3.4j); geometry only, no text; AppKit screen points, origin bottom-left;\n"
    @"#   a rectangle is x,y,w,h: H time client idx raw verdict place pointer frame visible hud\n"
    @"#   client = bundle id; idx = index/tries of the caret query (the character index last passed to the client's\n"
    @"#   attributes(forCharacterIndex:lineHeightRectangle:), and how many indexes were tried: an answer at the origin\n"
    @"#   means 'none' and the next lower index is asked), or - when the client has no text-input interface;\n"
    @"#   raw = the rectangle the client answered at that index; verdict = ok (the rectangle was used) or why not:\n"
    @"#   nil / origin / not_finite / negative_size / empty / off_screen (on no screen's full frame);\n"
    @"#   place = caret / pointer (no usable caret: next to the mouse pointer) / bottom (neither); pointer = the mouse\n"
    @"#   pointer x,y or -; frame / visible = the full and the visible frame of the screen it was placed on;\n"
    @"#   hud = the panel's frame.\n";
static NSString *const kLatencyLogGMarker = @"# G = the typing-pause decoder gate";  // the v3.2 line docs start here ...
static NSString *const kLatencyLogMDoc = @"# M = the input mode changed";           // ... this one ends v3.2c's ...
static NSString *const kLatencyLogNewestDoc = @"# C line, v3.2g:";                  // ... and this one (v3.2g; v3.2e / v3.2f wrote older texts) is the newest
static NSString *const kLatencyLogHDoc = @"# H = the Shift-tap mode HUD";            // v3.4j: the last block of the header
static NSString *const kLatencyLogVersionPrefix = @"# McBopomofoLM latency log v4";
NSNotificationName const SlothEPreferencesDidChangeNotification = @"SlothEPreferencesDidChange";
NSNotificationName const SlothEDiagnosticLogPreferenceDidChangeNotification = @"SlothEDiagnosticLogPreferenceDidChange";
static NSString *const kDiagnosticLogDefaultsKey = @"SlothEDiagnosticLogEnabled";  // Preferences.swift: kSlothEDiagnosticLogEnabledKey
/// v3.4: hidden switches (no menu item, read at every decoder pass): "SlothEDecoderFullBuffer" = NO goes back to the
/// v3.0-v3.3 per-key gate (candidate span only, lambda 2); "SlothEDecoderSuffixCapChars" = the right-text cap (24).
static NSString *const kDecoderFullBufferDefaultsKey = @"SlothEDecoderFullBuffer";
static NSString *const kDecoderSuffixCapDefaultsKey = @"SlothEDecoderSuffixCapChars";
static const char *const kLogQueueSpecificKey = "org.openvanilla.McBopomofoLM.slothe.log.queue";

static const unsigned long long kMaxLatencyLogBytes = 4ull * 1024ull * 1024ull;
static const size_t kMaxEndToEndSamples = 4096;
// v3.4n: has any runtime of this process started loading models? (a destructive test must not run after that)
static std::atomic<bool> gAnyModelLoadStarted { false };

@interface SlothEModelStoreResult ()
@property (readwrite, nonatomic) BOOL ok;
@property (readwrite, nonatomic) BOOL created;
@property (readwrite, nonatomic, nullable) NSString *setID;
@property (readwrite, nonatomic, nullable) NSString *directory;
@property (readwrite, nonatomic, nullable) NSString *error;
@end

@implementation SlothEModelStoreResult
@synthesize ok = _ok;
@synthesize created = _created;
@synthesize setID = _setID;
@synthesize directory = _directory;
@synthesize error = _error;
@end

@implementation SlothERuntime {
    std::shared_ptr<McBopomofoSlothE::Engine> _engine;
    std::shared_ptr<McBopomofoSlothE::Decoder> _decoder;
    McBopomofoSlothE::DecoderScoreCache _scoreCache;
    std::atomic<bool> _decoderLoadedFlag;
    std::atomic<bool> _decoderFailedFlag;
    double _decoderLoadMs;
    NSMutableArray<NSNumber *> *_e2e;
    std::atomic<bool> _loadedFlag;
    std::atomic<bool> _loadingFlag;
    std::atomic<bool> _failedFlag;
    NSString *_loadErrorString;
    NSString *_decoderLoadErrorString;
    NSString *_loadReason;
    NSString *_decoderLoadReason;
    double _loadMs;
    double _verifyMs;
    McBopomofoSlothE::CoreMLLoadInfo _encInfo;
    McBopomofoSlothE::CoreMLLoadInfo _decInfo;
    std::atomic<uint64_t> _lastPrewarmNs;
    NSUInteger _prewarmCount;
    std::atomic<NSUInteger> _prewarmDone;
    BOOL _aneCacheCleared;
    BOOL _loadAllDecoderFunctions;
    // v3.4p: the model store. `_modelDir` is where the models are loaded from (the store copy, or the bundle).
    BOOL _useModelStore;           // application runtimes; others only with modelStoreRootForTesting
    BOOL _modelStorePrepared;      // load queue
    BOOL _modelStoreCollected;     // load queue: the clean-up after the first successful load ran
    NSString *_modelDir;           // under @synchronized(self)
    NSString *_modelStoreStatus;   // under @synchronized(self)
    NSString *_modelStoreSetID;    // load queue
    NSString *_modelStoreActiveRoot;  // load queue
    std::atomic<NSUInteger> _storeSetsCollected;
    std::atomic<NSUInteger> _storeTemporariesCollected;
    std::atomic<NSUInteger> _storeSetsKeptForLeases;
    // Sol r49: the manifest as it was when this runtime started (every later check uses it, never the file again), and the
    // lease on the model set this runtime loads from, held for the runtime's whole life
    std::shared_ptr<const McBopomofoSlothE::ModelManifest> _manifest;
    std::shared_ptr<McBopomofoSlothE::ModelSetLease> _lease;
    // v3.4n: probe retry ("still loading"), eager compile pass
    BOOL _encoderProbeRetried;
    BOOL _decoderProbeRetried;
    // v3.4n: the master switch. suspendBackgroundWork bumps the generation and sets _suspended; the eager steps and
    // the delayed probe retries carry the generation they were scheduled under and do nothing under another one while
    // suspended, recording what is left; startLoading / resumeBackgroundWork continue it.
    std::atomic<uint64_t> _workGeneration;
    std::atomic<bool> _suspended;
    BOOL _encoderRetryPending;     // load queue
    BOOL _decoderRetryPending;     // load queue
    NSInteger _eagerResumeIndex;   // load queue; -1 = nothing left over
    std::atomic<NSUInteger> _probeRetryCount;
    std::atomic<NSUInteger> _loadAttempts;  // encoder + decoder loads that actually started (tests)
    std::atomic<bool> _eagerDone;
    std::atomic<uint64_t> _eagerMicroseconds;
    NSMutableArray<NSDictionary<NSString *, id> *> *_eagerLog;  // under _lazyMutex
    std::atomic<NSUInteger> _lazyLoadsRequested;
    std::atomic<NSUInteger> _lazyLoadsDone;
    std::atomic<NSUInteger> _lazyLoadsFailed;
    std::mutex _lazyMutex;
    std::map<size_t, McBopomofoSlothE::CoreMLLoadInfo> _lazyInfo;
    dispatch_queue_t _loadQueue;
    dispatch_queue_t _logQueue;
    NSFileHandle *_logHandle;              // log queue only
    // v3.3h: bit 0 = the log is ON; the upper bits count the OFF / ON transitions (a logging-session id).
    // A transition runs ON the log queue, so it is ordered with every write; a record carries the state it
    // was made under and is written only while the state still equals it (one switch of the log, in either
    // direction, drops everything made before it that has not been written yet).
    std::atomic<uint64_t> _logState;
    id _diagnosticLogObserver;
    NSISO8601DateFormatter *_logFormatter; // log queue only

    dispatch_queue_t _computeQueue;
    // coalescing job mailbox (any thread, under _jobMutex)
    std::mutex _jobMutex;
    dispatch_block_t _pendingJob;
    BOOL _drainScheduled;
    NSUInteger _peakPendingJobs;
    NSUInteger _coalescedJobs;
    NSUInteger _startedJobs;
    NSUInteger _submittedJobs;

    // current key event (main thread only)
    double _ksWalkMs;
    double _ksInWalkMs;
    BOOL _ksInWalkRan;
    double _ksProvMs;
    BOOL _ksProvRan;
    NSString *_ksForward;
    double _ksRerankMs;
    BOOL _ksRerankRan;
    NSString *_ksSettle;
    double _ksSettleMs;
}

@synthesize resourcePath = _resourcePath;
@synthesize logPath = _logPath;
@synthesize computeQueue = _computeQueue;
@synthesize debugComputeDelayMilliseconds = _debugComputeDelayMilliseconds;
@synthesize commitWaitMilliseconds = _commitWaitMilliseconds;
@synthesize prewarmIdleSeconds = _prewarmIdleSeconds;
@synthesize computeUnitsCPUOnlyForTesting = _computeUnitsCPUOnlyForTesting;
@synthesize decoderCPUOnlyForTesting = _decoderCPUOnlyForTesting;
@synthesize installProgress = _installProgress;
@synthesize keepANECacheForTesting = _keepANECacheForTesting;
@synthesize simulatePlacementFailureOnceForTesting = _simulatePlacementFailureOnceForTesting;
@synthesize simulateProbeSlowOnceForTesting = _simulateProbeSlowOnceForTesting;
@synthesize simulateDecoderProbeSlowOnceForTesting = _simulateDecoderProbeSlowOnceForTesting;
@synthesize probeRetryDelaySeconds = _probeRetryDelaySeconds;
@synthesize eagerCompileAfterLoad = _eagerCompileAfterLoad;
@synthesize bundleVersionForTesting = _bundleVersionForTesting;
@synthesize aneCacheClearMarkerPathForTesting = _aneCacheClearMarkerPathForTesting;
@synthesize aneCacheInstallMarkerPathForTesting = _aneCacheInstallMarkerPathForTesting;
@synthesize markerClaimedHookForTesting = _markerClaimedHookForTesting;
@synthesize clearActionForTesting = _clearActionForTesting;
@synthesize modelStoreRootForTesting = _modelStoreRootForTesting;
@synthesize fallbackPublishHookForTesting = _fallbackPublishHookForTesting;
@synthesize eagerStepHookForTesting = _eagerStepHookForTesting;
@synthesize lazyCPUOnlyForTesting = _lazyCPUOnlyForTesting;
@synthesize minEncoderANECostPercent = _minEncoderANECostPercent;
@synthesize minDecoderANECostPercent = _minDecoderANECostPercent;
@synthesize appliedCount = _appliedCount;
@synthesize unchangedCount = _unchangedCount;
@synthesize staleCount = _staleCount;
@synthesize skippedCount = _skippedCount;
@synthesize deferredCount = _deferredCount;
@synthesize consumedCount = _consumedCount;
@synthesize commitWaitCount = _commitWaitCount;
@synthesize commitTimeoutCount = _commitTimeoutCount;
@synthesize commitPartialCount = _commitPartialCount;
@synthesize decoderRunCount = _decoderRunCount;
@synthesize decoderPinCount = _decoderPinCount;
@synthesize decoderStaleCount = _decoderStaleCount;
@synthesize pauseGateRunCount = _pauseGateRunCount;
@synthesize pauseGateAppliedCount = _pauseGateAppliedCount;
@synthesize pauseGateChangedCount = _pauseGateChangedCount;
@synthesize legalEntryOfferedCount = _legalEntryOfferedCount;
@synthesize legalEntryChosenCount = _legalEntryChosenCount;
@synthesize legalEntryNotReadyCount = _legalEntryNotReadyCount;
@synthesize englishModeCount = _englishModeCount;
@synthesize logEnqueueRaceHookForTesting = _logEnqueueRaceHookForTesting;

+ (BOOL)runningUnderXCTest
{
    NSDictionary *env = NSProcessInfo.processInfo.environment;
    return env[@"XCTestConfigurationFilePath"] != nil || env[@"XCTestBundlePath"] != nil || env[@"XCTestSessionIdentifier"] != nil || NSClassFromString(@"XCTestCase") != nil;
}

+ (NSString *)defaultLatencyLogPath
{
    NSString *appSupport = NSSearchPathForDirectoriesInDomains(NSApplicationSupportDirectory, NSUserDomainMask, YES).firstObject;
    return [[appSupport stringByAppendingPathComponent:@"McBopomofoLM"] stringByAppendingPathComponent:@"latency.log"];
}

+ (SlothERuntime *)sharedRuntime
{
    static SlothERuntime *shared;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        NSString *resources = [NSBundle.mainBundle.resourcePath stringByAppendingPathComponent:@"SlothE"];
        NSString *log = [SlothERuntime runningUnderXCTest] ? nil : [SlothERuntime defaultLatencyLogPath];
        shared = [[SlothERuntime alloc] initForApplicationWithResourcePath:resources logPath:log];
    });
    return shared;
}

+ (BOOL)diagnosticLogPreference
{
    return [NSUserDefaults.standardUserDefaults boolForKey:kDiagnosticLogDefaultsKey];
}

- (instancetype)initForApplicationWithResourcePath:(NSString *)resourcePath logPath:(nullable NSString *)logPath
{
    self = [self initWithResourcePath:resourcePath logPath:logPath];
    if (self) {
        _useModelStore = YES;  // v3.4p: the application loads its models from the model store
        _logState.store(logPath != nil && [SlothERuntime diagnosticLogPreference] ? 1 : 0);
        __weak SlothERuntime *weakSelf = self;
        _diagnosticLogObserver = [NSNotificationCenter.defaultCenter addObserverForName:SlothEDiagnosticLogPreferenceDidChangeNotification object:nil queue:nil usingBlock:^(NSNotification *note) {
            weakSelf.diagnosticLogEnabled = [SlothERuntime diagnosticLogPreference];
        }];
    }
    return self;
}

- (BOOL)diagnosticLogEnabled
{
    return _logPath != nil && (_logState.load(std::memory_order_acquire) & 1) != 0;
}

/// v3.3h: the transition runs on the log queue and the caller waits for it. So: a write that is already
/// running finishes first; then the state changes and (OFF) the file is closed before this returns;
/// every write that was queued earlier, or that passed its guard just before, finds another state and
/// does nothing. After `diagnosticLogEnabled = NO` returns, no directory, file or line is made until
/// the switch is turned on again. Turning it ON opens the log at once and writes the model placement
/// that is already known (the models usually finished loading while the log was OFF).
- (void)setDiagnosticLogEnabled:(BOOL)enabled
{
    if (_logPath == nil) {
        return;
    }
    void (^transition)(void) = ^{
        uint64_t state = self->_logState.load(std::memory_order_acquire);
        if (((state & 1) != 0) == (enabled != NO)) {
            return;
        }
        self->_logState.store((((state >> 1) + 1) << 1) | (enabled ? 1 : 0), std::memory_order_release);
        if (enabled) {
            if ([self _openLog]) {
                [self _writeModelPlacementSummaryOnLogQueue];
            }
        } else if (self->_logHandle != nil) {
            [self->_logHandle closeAndReturnError:nil];
            self->_logHandle = nil;
        }
    };
    if (dispatch_get_specific(kLogQueueSpecificKey) != NULL) {
        transition();
    } else {
        dispatch_sync(_logQueue, transition);
    }
}

- (BOOL)_logging
{
    return _logPath != nil && (_logState.load(std::memory_order_acquire) & 1) != 0;
}

- (instancetype)initWithResourcePath:(NSString *)resourcePath logPath:(nullable NSString *)logPath
{
    self = [super init];
    if (self) {
        _resourcePath = [resourcePath copy];
        _logPath = [logPath copy];
        _logState.store(logPath != nil ? 1 : 0);
        _loadedFlag.store(false);
        _loadingFlag.store(false);
        _failedFlag.store(false);
        _decoderLoadedFlag.store(false);
        _decoderFailedFlag.store(false);
        _e2e = [NSMutableArray array];
        _loadQueue = dispatch_queue_create("org.openvanilla.McBopomofoLM.slothe.load", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));
        _logQueue = dispatch_queue_create("org.openvanilla.McBopomofoLM.slothe.log", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));
        dispatch_queue_set_specific(_logQueue, kLogQueueSpecificKey, (void *)kLogQueueSpecificKey, NULL);
        _computeQueue = dispatch_queue_create("org.openvanilla.McBopomofoLM.slothe.compute", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));
        _commitWaitMilliseconds = 30;
        _prewarmIdleSeconds = 1.0;
        _minEncoderANECostPercent = 99.0;
        _minDecoderANECostPercent = 80.0;
        _probeRetryDelaySeconds = 30.0;
        _workGeneration.store(0);
        _suspended.store(false);
        _eagerResumeIndex = -1;
        _probeRetryCount.store(0);
        _loadAttempts.store(0);
        _storeSetsCollected.store(0);
        _storeTemporariesCollected.store(0);
        _storeSetsKeptForLeases.store(0);
        _eagerDone.store(false);
        _eagerMicroseconds.store(0);
        _eagerLog = [NSMutableArray array];
        _lastPrewarmNs.store(0);
        _prewarmDone.store(0);
        _lazyLoadsRequested.store(0);
        _lazyLoadsDone.store(0);
        _lazyLoadsFailed.store(0);
        _computeUnitsCPUOnlyForTesting = NO;
        _ksForward = @"none";
        _ksSettle = @"none";
    }
    return self;
}

+ (SlothERuntime *)sharedLoadedRuntimeForTesting
{
    static SlothERuntime *shared;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        NSString *resources = [NSBundle.mainBundle.resourcePath stringByAppendingPathComponent:@"SlothE"];
        shared = [[SlothERuntime alloc] initWithResourcePath:resources logPath:nil];
        shared.keepANECacheForTesting = YES;
        [shared loadForInstallWithProgress:nil];  // every decoder function loaded (the lazy path has its own tests)
    });
    return shared;
}

- (instancetype)initSharingModelsOf:(SlothERuntime *)other logPath:(nullable NSString *)logPath
{
    self = [self initWithResourcePath:other.resourcePath logPath:logPath];
    if (self) {
        _modelDir = other.modelDirectory;
        _modelStoreStatus = other.modelStoreStatus;
        _modelStorePrepared = YES;
        _manifest = other->_manifest;
        _lease = other->_lease;
        _engine = other->_engine;
        _decoder = other->_decoder;
        _loadMs = other->_loadMs;
        _decoderLoadMs = other->_decoderLoadMs;
        _encInfo = other->_encInfo;
        _decInfo = other->_decInfo;
        _loadReason = other->_loadReason;
        _decoderLoadReason = other->_decoderLoadReason;
        _loadedFlag.store(other->_loadedFlag.load());
        _failedFlag.store(other->_failedFlag.load());
        _decoderLoadedFlag.store(other->_decoderLoadedFlag.load());
        _decoderFailedFlag.store(other->_decoderFailedFlag.load());
    }
    return self;
}

- (void)dealloc
{
    if (_diagnosticLogObserver != nil) {
        [NSNotificationCenter.defaultCenter removeObserver:_diagnosticLogObserver];
    }
    [_logHandle closeAndReturnError:nil];
}

- (BOOL)loaded
{
    return _loadedFlag.load(std::memory_order_acquire);
}

- (BOOL)loading
{
    return _loadingFlag.load(std::memory_order_acquire);
}

- (BOOL)loadFailed
{
    return _failedFlag.load(std::memory_order_acquire);
}

- (nullable NSString *)loadError
{
    return _failedFlag.load(std::memory_order_acquire) ? _loadErrorString : nil;
}

- (double)loadMilliseconds
{
    return _loadedFlag.load(std::memory_order_acquire) ? _loadMs : 0;
}

- (nullable McBopomofoSlothE::Engine *)engine
{
    return _loadedFlag.load(std::memory_order_acquire) ? _engine.get() : nullptr;
}

- (BOOL)decoderLoaded
{
    return _decoderLoadedFlag.load(std::memory_order_acquire);
}

- (BOOL)decoderLoadFailed
{
    return _decoderFailedFlag.load(std::memory_order_acquire);
}

- (nullable NSString *)decoderLoadError
{
    return _decoderFailedFlag.load(std::memory_order_acquire) ? _decoderLoadErrorString : nil;
}

- (double)decoderLoadMilliseconds
{
    return _decoderLoadedFlag.load(std::memory_order_acquire) ? _decoderLoadMs : 0;
}

- (nullable McBopomofoSlothE::Decoder *)decoder
{
    return _decoderLoadedFlag.load(std::memory_order_acquire) ? _decoder.get() : nullptr;
}

- (McBopomofoSlothE::DecoderParams)decoderParams
{
    // v3.4 (SlothE/eval-2026-10/): the decoder scores the whole buffer around the candidate, lambda 1, tau 0.5,
    // top-3, the right text capped at 24 chars. Config A' of v3.0-v3.3 (lambda 2, candidate span only) is
    // DecoderParams::V2(); v1 was lambda 3, tau 0.9.
    McBopomofoSlothE::DecoderParams params;
    NSUserDefaults *defaults = NSUserDefaults.standardUserDefaults;
    if ([defaults objectForKey:kDecoderFullBufferDefaultsKey] != nil && ![defaults boolForKey:kDecoderFullBufferDefaultsKey]) {
        params = McBopomofoSlothE::DecoderParams::V2();
    }
    id cap = [defaults objectForKey:kDecoderSuffixCapDefaultsKey];
    if ([cap isKindOfClass:NSNumber.class] && [cap integerValue] >= 0) {
        params.suffixCapChars = static_cast<size_t>([cap integerValue]);
    }
    return params;
}

- (McBopomofoSlothE::DecoderScoreCache *)decoderScoreCache
{
    return &_scoreCache;
}

- (NSArray<NSNumber *> *)recentEndToEndMilliseconds
{
    return [_e2e copy];
}

- (McBopomofoSlothE::InWalkParams)inWalkParams
{
    McBopomofoSlothE::InWalkParams params;
    params.beta = 0.3;
    params.gamma = 0.0;
    params.penalty = -15.0;
    params.variantGuard = true;
    return params;
}

#pragma mark - Compute jobs

- (void)submitComputeJob:(dispatch_block_t)job
{
    dispatch_block_t replaced = nil;
    BOOL scheduleDrain = NO;
    {
        std::lock_guard<std::mutex> lock(_jobMutex);
        replaced = _pendingJob;
        _pendingJob = [job copy];
        ++_submittedJobs;
        if (replaced != nil) {
            ++_coalescedJobs;
        }
        _peakPendingJobs = std::max<NSUInteger>(_peakPendingJobs, 1);
        if (!_drainScheduled) {
            _drainScheduled = YES;
            scheduleDrain = YES;
        }
    }
    replaced = nil;  // released here, outside the lock
    if (scheduleDrain) {
        dispatch_async(_computeQueue, ^{
            dispatch_block_t next = nil;
            {
                std::lock_guard<std::mutex> lock(self->_jobMutex);
                next = self->_pendingJob;
                self->_pendingJob = nil;
                self->_drainScheduled = NO;
                if (next != nil) {
                    ++self->_startedJobs;
                }
            }
            if (next != nil) {
                next();
            }
        });
    }
}

- (NSUInteger)pendingComputeJobs
{
    std::lock_guard<std::mutex> lock(_jobMutex);
    return _pendingJob != nil ? 1 : 0;
}

- (NSUInteger)peakPendingComputeJobs
{
    std::lock_guard<std::mutex> lock(_jobMutex);
    return _peakPendingJobs;
}

- (NSUInteger)coalescedComputeJobs
{
    std::lock_guard<std::mutex> lock(_jobMutex);
    return _coalescedJobs;
}

- (NSUInteger)submittedComputeJobs
{
    std::lock_guard<std::mutex> lock(_jobMutex);
    return _submittedJobs;
}

- (NSUInteger)startedComputeJobs
{
    std::lock_guard<std::mutex> lock(_jobMutex);
    return _startedJobs;
}

+ (int64_t)liveGridSnapshots
{
    return McBopomofoSlothE::SlothEGrid::LiveSnapshots();
}

+ (uint64_t)physFootprintBytes
{
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
        return 0;
    }
    return info.phys_footprint;
}

- (void)resetComputeJobStatistics
{
    std::lock_guard<std::mutex> lock(_jobMutex);
    _peakPendingJobs = _pendingJob != nil ? 1 : 0;
    _coalescedJobs = 0;
    _startedJobs = 0;
    _submittedJobs = _pendingJob != nil ? 1 : 0;
}

- (void)startLoading
{
    [self resumeBackgroundWork];  // v3.4n: the master switch is on again: the work it interrupted goes on
    if (_loadedFlag.load() || _failedFlag.load() || _loadingFlag.exchange(true)) {
        return;
    }
    _eagerCompileAfterLoad = YES;  // v3.4n: the input method compiles every decoder function in the background
    dispatch_async(_loadQueue, ^{
        [self _loadOnLoadQueue];
    });
}

- (BOOL)loadSynchronously
{
    _loadingFlag.store(true);
    dispatch_sync(_loadQueue, ^{
        [self _loadOnLoadQueue];
    });
    return _loadedFlag.load();
}

- (McBopomofoSlothE::CoreMLLoadOptions)_loadOptionsForDecoder:(BOOL)decoder
{
    McBopomofoSlothE::CoreMLLoadOptions options;
    BOOL cpuOnly = _computeUnitsCPUOnlyForTesting || (decoder && _decoderCPUOnlyForTesting);
    options.minEncoderANECostPercent = _minEncoderANECostPercent;
    options.minDecoderANECostPercent = _minDecoderANECostPercent;
    options.units = cpuOnly ? McBopomofoSlothE::ComputeUnits::kCPUOnly : McBopomofoSlothE::ComputeUnits::kCPUAndNeuralEngine;
    void (^progress)(NSString *) = _installProgress;
    if (progress != nil) {
        options.progress = [progress](const std::string &line) { progress(@(line.c_str())); };
    }
    if (decoder) {
        // v2.1: t16 at start-up, the longer functions on first need (install: all of them).
        if (!_loadAllDecoderFunctions) {
            options.decoderLengthsAtLoad = { 16 };
        }
        options.lazyUnits = _lazyCPUOnlyForTesting ? McBopomofoSlothE::ComputeUnits::kCPUOnly : McBopomofoSlothE::ComputeUnits::kCPUAndNeuralEngine;
        // The backend keeps these, so they hold the runtime weakly (no retain cycle);
        // a queued load holds it strongly until it has run.
        __weak SlothERuntime *weakSelf = self;
        dispatch_queue_t queue = _loadQueue;
        options.runLater = [weakSelf, queue](std::function<void()> work) {
            SlothERuntime *strongSelf = weakSelf;
            if (strongSelf == nil) {
                return;
            }
            ++strongSelf->_lazyLoadsRequested;
            dispatch_async(queue, ^{
                (void)strongSelf;
                work();
            });
        };
        options.onLazyLoad = [weakSelf](size_t length, const McBopomofoSlothE::CoreMLLoadInfo &info) {
            [weakSelf _noteLazyLoad:length info:info];
        };
        options.onLazyNextLoad = [weakSelf](size_t length, const McBopomofoSlothE::CoreMLLoadInfo &info) {
            [weakSelf _logRuntime:[NSString stringWithFormat:@"decoder.n%zu", length] info:info reason:@(info.reason.c_str())];
            if (!info.ok) {
                NSLog(@"McBopomofoLM: SlothE next-token function n%zu not used until the input method restarts (%s: %s); the correction window uses the T1 rule", length, info.reason.c_str(), info.detail.c_str());
            }
        };
        std::string dir([self _modelDirectoryForLoad].fileSystemRepresentation);
        std::string manifestPath = [self _bundleManifestPath];
        std::shared_ptr<const McBopomofoSlothE::ModelManifest> snapshot = _manifest;
        options.verifyBeforeLazyLoad = [dir, manifestPath, snapshot](std::string *error) {
            // v3.4p: a store copy is checked against the BUNDLE's manifest as this runtime read it at start-up (Sol r49:
            // an update may have replaced the file at the bundle path since; it is never read again), at every lazy load
            if (snapshot != nullptr) {
                return McBopomofoSlothE::VerifyRuntimeFiles(dir, McBopomofoSlothE::DecoderRuntimeFiles(), error, *snapshot);
            }
            return McBopomofoSlothE::VerifyRuntimeFiles(dir, McBopomofoSlothE::DecoderRuntimeFiles(), error, manifestPath);
        };
    }
    return options;
}

// A decoder function loaded on first need (background load queue).
- (void)_noteLazyLoad:(size_t)length info:(const McBopomofoSlothE::CoreMLLoadInfo &)info
{
    NSString *reason = @(info.reason.c_str());
    if (info.ok) {
        _lazyLoadsDone.fetch_add(1);
    } else {
        _lazyLoadsFailed.fetch_add(1);
        NSLog(@"McBopomofoLM: SlothE decoder t%zu not used until the input method restarts (%@: %s); those requests stay encoder-only", length, reason, info.detail.c_str());
        // A failed plan or a collapsed placement points at a damaged compile
        // cache: ask the NEXT start-up to clear it and recompile (once).
        if (!_lazyCPUOnlyForTesting && [self _looksLikeABadANECache:info]) {
            NSString *marker = [self _markerPath];
            if (marker != nil) {
                [NSFileManager.defaultManager createDirectoryAtPath:marker.stringByDeletingLastPathComponent withIntermediateDirectories:YES attributes:nil error:nil];
                [@"t" writeToFile:marker atomically:YES encoding:NSUTF8StringEncoding error:nil];
            }
        }
    }
    {
        std::lock_guard<std::mutex> lock(_lazyMutex);
        _lazyInfo[length] = info;
    }
    [self _logRuntime:[NSString stringWithFormat:@"decoder.t%zu", length] info:info reason:reason];
}

- (NSArray<NSNumber *> *)decoderLoadedLengths
{
    NSMutableArray *out = [NSMutableArray array];
    McBopomofoSlothE::Decoder *decoder = [self decoder];
    if (decoder != nullptr) {
        for (size_t T : decoder->loadedLengths()) {
            [out addObject:@(T)];
        }
    }
    return out;
}

- (NSUInteger)decoderUnavailableRequests
{
    McBopomofoSlothE::Decoder *decoder = [self decoder];
    return decoder != nullptr ? static_cast<NSUInteger>(decoder->unavailableCount()) : 0;
}

- (NSUInteger)decoderLazyLoadsRequested
{
    return _lazyLoadsRequested.load();
}

- (NSUInteger)decoderLazyLoadsDone
{
    return _lazyLoadsDone.load();
}

- (NSUInteger)decoderLazyLoadsFailed
{
    return _lazyLoadsFailed.load();
}

- (NSDictionary<NSString *, NSNumber *> *)lazyLoadSummaryForLength:(NSInteger)length
{
    std::lock_guard<std::mutex> lock(_lazyMutex);
    auto it = _lazyInfo.find(static_cast<size_t>(length));
    if (it == _lazyInfo.end()) {
        return @{};
    }
    const auto &i = it->second;
    return @{ @"ok" : @(i.ok), @"loadMs" : @(i.loadMs), @"planMs" : @(i.planMs), @"probeMs" : @(i.probeMs), @"aneCostPercent" : @(i.aneCostPercent) };
}

- (void)_progress:(NSString *)line
{
    if (_installProgress != nil) {
        _installProgress(line);
    }
}

- (NSString *)_runtimeRecordTail:(NSString *)model info:(const McBopomofoSlothE::CoreMLLoadInfo &)info reason:(NSString *)reason
{
    NSString *pct = info.aneCostPercent >= 0 ? [NSString stringWithFormat:@"%.1f", info.aneCostPercent] : @"";
    NSString *probe = info.probeMs >= 0 ? [NSString stringWithFormat:@"%.2f", info.probeMs] : @"";
    NSString *plan = info.planChecked ? [NSString stringWithFormat:@"%.0f", info.planMs] : @"";
    NSMutableArray *fns = [NSMutableArray array];
    for (const auto &[fn, p] : info.functionANEPercent) {
        [fns addObject:[NSString stringWithFormat:@"%s:%.1f", fn.c_str(), p]];
    }
    return [NSString stringWithFormat:@"\t%@\t%@\t%@\t%.0f\t%@\t%@\t%@\t%@\n", model, [reason isEqualToString:@"ok"] ? @"ane" : @"none", reason, info.loadMs, pct, probe, plan, [fns componentsJoinedByString:@","]];
}

- (void)_logRuntime:(NSString *)model info:(const McBopomofoSlothE::CoreMLLoadInfo &)info reason:(NSString *)reason
{
    [self _appendTimestampedLine:@"R" tail:[self _runtimeRecordTail:model info:info reason:reason]];
}

/// Log queue only: the R records of the models that finished loading before the log was switched on, in
/// the order the load made them (encoder, decoder, the lazily loaded decoder functions), under a comment
/// line that says why they are late.
- (void)_writeModelPlacementSummaryOnLogQueue
{
    NSMutableArray<NSString *> *records = [NSMutableArray array];
    if (_loadedFlag.load(std::memory_order_acquire) || _failedFlag.load(std::memory_order_acquire)) {
        [records addObject:[self _runtimeRecordTail:@"encoder" info:_encInfo reason:(_loadReason != nil ? _loadReason : @"unknown")]];
        if (_decoderLoadedFlag.load(std::memory_order_acquire) || _decoderFailedFlag.load(std::memory_order_acquire)) {
            [records addObject:[self _runtimeRecordTail:@"decoder" info:_decInfo reason:(_decoderLoadReason != nil ? _decoderLoadReason : @"unknown")]];
        }
    }
    {
        std::lock_guard<std::mutex> lock(_lazyMutex);
        for (const auto &[length, info] : _lazyInfo) {
            [records addObject:[self _runtimeRecordTail:[NSString stringWithFormat:@"decoder.t%zu", length] info:info reason:@(info.reason.c_str())]];
        }
    }
    if (records.count == 0) {
        return;
    }
    NSDate *now = [NSDate date];
    NSString *timestamp = [self _timestamp:now];
    [_logHandle writeData:[[NSString stringWithFormat:@"# model placement, loaded before the log was switched on (%@)\n", timestamp] dataUsingEncoding:NSUTF8StringEncoding] error:nil];
    for (NSString *tail in records) {
        [_logHandle writeData:[[NSString stringWithFormat:@"R\t%@%@", timestamp, tail] dataUsingEncoding:NSUTF8StringEncoding] error:nil];
    }
}

- (dispatch_queue_t)logQueueForTesting
{
    return _logQueue;
}

- (NSArray<NSString *> *)modelPlacementRecordsForTesting
{
    __block NSArray<NSString *> *out = nil;
    dispatch_sync(_logQueue, ^{
        NSMutableArray<NSString *> *records = [NSMutableArray array];
        if (self->_loadedFlag.load() || self->_failedFlag.load()) {
            [records addObject:[self _runtimeRecordTail:@"encoder" info:self->_encInfo reason:(self->_loadReason != nil ? self->_loadReason : @"unknown")]];
            if (self->_decoderLoadedFlag.load() || self->_decoderFailedFlag.load()) {
                [records addObject:[self _runtimeRecordTail:@"decoder" info:self->_decInfo reason:(self->_decoderLoadReason != nil ? self->_decoderLoadReason : @"unknown")]];
            }
        }
        out = records;
    });
    return out;
}

// Encoder first (its result alone already drives the in-walk), then the
// decoder. Core ML only, CPU_AND_NE; a model that fails integrity, does not
// load, or is not placed on the ANE is not used -- no CPU fallback.
- (void)_loadOnLoadQueue
{
    if (_loadedFlag.load() || _failedFlag.load()) {
        // v3.4n: the start-up operation is more than the encoder: while the decoder's delayed retry is pending (a
        // second entry such as loadSynchronously must not end "still loading", or the ready / partial notice is lost)
        if (!_encoderRetryPending && !_decoderRetryPending) {
            _loadingFlag.store(false);
        }
        return;
    }
    gAnyModelLoadStarted.store(true);
    ++_loadAttempts;
    [self _prepareModelStore];  // v3.4p: before the clock starts, so the load timings stay comparable
    auto t0 = std::chrono::steady_clock::now();
    std::string dir([self _modelDirectoryForLoad].fileSystemRepresentation);
    std::string error;
    NSString *reason = @"ok";
    std::unique_ptr<McBopomofoSlothE::Engine> engine;
    NSString *clearWhy = nil;
    // v3.4n: claim, clear, cap and delete are one sequence under one lock (see _consumeANECacheClearMarker)
    if ([self _consumeANECacheClearMarker:&clearWhy clear:nil] && clearWhy != nil) {
        _aneCacheCleared = NO;  // this load may still self-heal once
        [self _progress:[NSString stringWithFormat:@"  compile cache cleared (%@): compiling from scratch", clearWhy]];
    }
    [self _progress:@"SlothE-T 25M encoder (enc25m.mlmodelc):"];
    bool ok = [self _verifyFilesInDirectory:dir names:McBopomofoSlothE::EncoderRuntimeFiles() error:&error];
    _verifyMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    _encInfo = McBopomofoSlothE::CoreMLLoadInfo();
    if (!ok) {
        reason = @"integrity";
        [self _progress:[NSString stringWithFormat:@"  integrity check failed: %s", error.c_str()]];
    } else {
        [self _progress:[NSString stringWithFormat:@"  files verified (size + sha256) in %.0f ms", _verifyMs]];
        auto backend = McBopomofoSlothE::LoadCoreMLEncoder(dir, [self _loadOptionsForDecoder:NO], &_encInfo);
        if (backend != nullptr && _simulatePlacementFailureOnceForTesting.length > 0) {
            backend.reset();
            _encInfo.ok = false;
            _encInfo.reason = _simulatePlacementFailureOnceForTesting.UTF8String;
            _encInfo.aneCostPercent = 0;
            _encInfo.detail = "simulated for a test";
            _simulatePlacementFailureOnceForTesting = nil;
        }
        if (backend == nullptr && [self _looksLikeABadANECache:_encInfo] && [self _clearANECacheOnce:@(("encoder " + _encInfo.reason).c_str())]) {
            [self _progress:@"  placement check failed or collapsed to the CPU (damaged or stale Neural Engine cache?): cache cleared, compiling again"];
            backend = McBopomofoSlothE::LoadCoreMLEncoder(dir, [self _loadOptionsForDecoder:NO], &_encInfo);
        }
        if (backend != nullptr && _simulateProbeSlowOnceForTesting) {
            backend.reset();
            _encInfo.ok = false;
            _encInfo.reason = "probe_slow";
            _encInfo.probeMs = 99;
            _encInfo.detail = "simulated for a test";
            _simulateProbeSlowOnceForTesting = NO;
        }
        if (backend == nullptr && _encInfo.reason == "probe_slow" && !_encoderProbeRetried && !_loadAllDecoderFunctions) {
            // v3.4n: a slow probe may be contention (the user typing during the compile), not the model: the
            // runtime stays "loading" (not failed) and loads the encoder once more after a pause.
            _encoderProbeRetried = YES;
            ++_probeRetryCount;
            _encoderRetryPending = YES;
            [self _appendLogLine:[NSString stringWithFormat:@"# %@ encoder latency probe slow (%.2f ms): loading again in %.0f s\n", [NSDate date], _encInfo.probeMs, _probeRetryDelaySeconds]];
            [self _progress:[NSString stringWithFormat:@"  encoder latency probe slow (%.2f ms): loading again in %.0f s", _encInfo.probeMs, _probeRetryDelaySeconds]];
            const uint64_t generation = _workGeneration.load();
            dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(_probeRetryDelaySeconds * NSEC_PER_SEC)), _loadQueue, ^{
                // Sol r48: the switch is checked ON ITS OWN, before the generation (a retry scheduled while the switch
                // is already off captures the already-current generation): the retry stays pending and the next
                // resumeBackgroundWork runs it. A generation that moved on means a suspend -> resume cycle happened
                // meanwhile, whose resume block runs the pending retry itself.
                if (self->_suspended.load() || self->_workGeneration.load() != generation) {
                    return;
                }
                if (!self->_encoderRetryPending) {
                    return;  // already run by a resume
                }
                self->_encoderRetryPending = NO;
                [self _loadOnLoadQueue];
            });
            return;  // _loadingFlag stays set
        }
        if (backend == nullptr) {
            ok = false;
            reason = @(_encInfo.reason.c_str());
            error = _encInfo.detail;
        } else {
            engine = std::make_unique<McBopomofoSlothE::Engine>();
            if (!engine->load(dir, std::move(backend), &error)) {
                ok = false;
                reason = [@(error.c_str()) hasPrefix:@"model probe"] ? @"probe_failed" : @"vocab";
                engine.reset();
            } else if (engine->vocabulary().unkShareCount() != McBopomofoSlothE::kExpectedUnkShareCount) {
                NSLog(@"McBopomofoLM: char2id maps %zu chars to the UNK id (expected %zu); rare-char scores use log(%zu)", engine->vocabulary().unkShareCount(), McBopomofoSlothE::kExpectedUnkShareCount, engine->vocabulary().unkShareCount());
            }
        }
    }
    double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    [self _logRuntime:@"encoder" info:_encInfo reason:reason];
    if (ok) {
        _loadMs = elapsed;
        _engine = std::move(engine);
        _loadReason = @"ok";
        _loadedFlag.store(true, std::memory_order_release);
        [self _progress:[NSString stringWithFormat:@"  encoder ON the Neural Engine: loaded in %.1f s (ANE compile included when not cached)", elapsed / 1000.0]];
        [self _collectModelStoreGarbageAfterLoad];  // v3.4p: only here, after this runtime's own successful load
        if ([self _loadDecoderOnLoadQueue]) {
            return;  // the decoder's latency probe was slow: it loads again after a pause, still "loading"
        }
    } else {
        _loadReason = reason;
        _loadErrorString = [NSString stringWithFormat:@"%@: %s", reason, error.c_str()];
        _failedFlag.store(true, std::memory_order_release);
        NSLog(@"McBopomofoLM: SlothE-T encoder not used (%@); stock McBopomofo behaviour", _loadErrorString);
        [self _progress:[NSString stringWithFormat:@"  encoder NOT used (%@): the input method behaves like stock McBopomofo", _loadErrorString]];
        _decoderLoadReason = @"no_encoder";
        _decoderFailedFlag.store(true, std::memory_order_release);
    }
    _loadingFlag.store(false, std::memory_order_release);
}

// The decoder, after the encoder; until it is loaded (or if it cannot be)
// the key handler runs the encoder-only in-walk path.
// Returns YES when a slow latency probe made it schedule one more load of the decoder (v3.4n): the caller then
// leaves `loading` set. NO otherwise (the decoder is loaded, or failed for good).
- (BOOL)_loadDecoderOnLoadQueue
{
    ++_loadAttempts;
    auto t0 = std::chrono::steady_clock::now();
    std::string dir([self _modelDirectoryForLoad].fileSystemRepresentation);
    std::string error;
    NSString *reason = @"ok";
    [self _progress:@"SlothE decoder (dec60m.mlmodelc):"];
    _decInfo = McBopomofoSlothE::CoreMLLoadInfo();
    auto decoder = std::make_unique<McBopomofoSlothE::Decoder>();
    bool ok = [self _verifyFilesInDirectory:dir names:McBopomofoSlothE::DecoderRuntimeFiles() error:&error];
    if (!ok) {
        reason = @"integrity";
    } else {
        [self _progress:@"  files verified (size + sha256)"];
        int32_t bos = 0;
        int32_t pad = 0;
        auto tokenizer = McBopomofoSlothE::LoadDecoderTokenizer(dir, &bos, &pad, &error);
        auto backend = tokenizer != nullptr ? McBopomofoSlothE::LoadCoreMLDecoder(dir, [self _loadOptionsForDecoder:YES], &_decInfo) : nullptr;
        if (tokenizer != nullptr && backend == nullptr && [self _looksLikeABadANECache:_decInfo] && [self _clearANECacheOnce:@(("decoder " + _decInfo.reason).c_str())]) {
            [self _progress:@"  placement check failed or collapsed to the CPU (damaged or stale Neural Engine cache?): cache cleared, compiling again"];
            backend = McBopomofoSlothE::LoadCoreMLDecoder(dir, [self _loadOptionsForDecoder:YES], &_decInfo);
        }
        if (backend != nullptr && _simulateDecoderProbeSlowOnceForTesting) {
            backend.reset();
            _decInfo.ok = false;
            _decInfo.reason = "probe_slow";
            _decInfo.probeMs = 99;
            _decInfo.detail = "simulated for a test";
            _simulateDecoderProbeSlowOnceForTesting = NO;
        }
        if (tokenizer != nullptr && backend == nullptr && _decInfo.reason == "probe_slow" && !_decoderProbeRetried && !_loadAllDecoderFunctions) {
            _decoderProbeRetried = YES;
            ++_probeRetryCount;
            _decoderRetryPending = YES;
            [self _appendLogLine:[NSString stringWithFormat:@"# %@ decoder latency probe slow (%.2f ms): loading again in %.0f s\n", [NSDate date], _decInfo.probeMs, _probeRetryDelaySeconds]];
            [self _progress:[NSString stringWithFormat:@"  decoder latency probe slow (%.2f ms): loading again in %.0f s", _decInfo.probeMs, _probeRetryDelaySeconds]];
            const uint64_t generation = _workGeneration.load();
            dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(_probeRetryDelaySeconds * NSEC_PER_SEC)), _loadQueue, ^{
                if (self->_suspended.load() || self->_workGeneration.load() != generation) {
                    return;  // as the encoder's retry: stays pending, resumeBackgroundWork runs it
                }
                if (!self->_decoderRetryPending) {
                    return;
                }
                [self _runDecoderRetry];
            });
            return YES;
        }
        if (tokenizer == nullptr) {
            ok = false;
            reason = @"missing";
        } else if (backend == nullptr) {
            ok = false;
            reason = @(_decInfo.reason.c_str());
            error = _decInfo.detail;
        } else if (!decoder->load(std::move(backend), tokenizer, bos, pad, &error)) {
            ok = false;
            reason = @"load_error";
        } else {
            std::string firstError;  // v3.1b: without it the correction window keeps the T1 rule
            if (!decoder->loadFirstChars(dir + "/" + McBopomofoSlothE::kDecoderFirstCharsName, &firstError)) {
                NSLog(@"McBopomofoLM: next-token first-char table not used (%s)", firstError.c_str());
            }
            std::vector<double> scores;
            McBopomofoSlothE::DecoderCallStats stats;
            ok = decoder->score("", std::vector<std::string> { "今天", "天氣" }, &scores, &stats);
            for (double v : scores) {
                ok = ok && std::isfinite(v) && v < 0.0;
            }
            if (!ok) {
                reason = @"probe_failed";
                error = "decoder probe gave invalid scores";
            }
        }
    }
    double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    [self _logRuntime:@"decoder" info:_decInfo reason:reason];
    if (ok) {
        _decoderLoadMs = elapsed;
        _decoder = std::move(decoder);
        _decoderLoadReason = @"ok";
        _decoderLoadedFlag.store(true, std::memory_order_release);
        [self _progress:[NSString stringWithFormat:@"  decoder ON the Neural Engine: loaded in %.1f s", elapsed / 1000.0]];
        [self _startEagerCompilePass];
    } else {
        _decoderLoadReason = reason;
        _decoderLoadErrorString = [NSString stringWithFormat:@"%@: %s", reason, error.c_str()];
        _decoderFailedFlag.store(true, std::memory_order_release);
        NSLog(@"McBopomofoLM: SlothE decoder not used (%@); encoder-only in-walk", _decoderLoadErrorString);
        [self _progress:[NSString stringWithFormat:@"  decoder NOT used (%@): encoder-only choices", _decoderLoadErrorString]];
    }
    return NO;
}

// ---------------------------------------------------------------- v3.4n: the eager compile pass

// Need order: the functions the typing reaches first. Every step is its own block on the load queue, queued only
// after the previous one finished, so a lazy load requested by the user's typing in between waits for at most
// the one function being compiled.
struct EagerStep {
    bool next;
    size_t length;
};
static const EagerStep kEagerOrder[] = { { false, 32 }, { true, 16 }, { false, 64 }, { true, 32 }, { false, 96 }, { true, 64 }, { true, 96 } };

- (void)_startEagerCompilePass
{
    if (!_eagerCompileAfterLoad || _loadAllDecoderFunctions || _decoder == nullptr) {
        return;
    }
    _eagerMicroseconds.store(0);
    _eagerDone.store(false);
    if (_suspended.load()) {
        _eagerResumeIndex = 0;  // AI selection is off: the pass waits for the switch
        return;
    }
    [self _eagerCompileStep:0 decoder:_decoder];
}

// A step that finds the switch off records where the pass stopped (`_eagerResumeIndex`) and does nothing; when the
// switch is on again (resumeBackgroundWork) the pass goes on from there.
- (void)_eagerCompileStep:(NSUInteger)index decoder:(std::shared_ptr<McBopomofoSlothE::Decoder>)decoder
{
    if (index >= sizeof(kEagerOrder) / sizeof(kEagerOrder[0])) {
        _eagerDone.store(true, std::memory_order_release);
        [self _appendLogLine:[NSString stringWithFormat:@"# %@ eager compile pass finished in %.1f s\n", [NSDate date], _eagerMicroseconds.load() / 1e6]];
        return;
    }
    __weak SlothERuntime *weakSelf = self;
    dispatch_async(_loadQueue, ^{
        SlothERuntime *strongSelf = weakSelf;
        if (strongSelf == nil) {
            return;
        }
        // Sol r48: suspension is checked on its own, first, whatever the generation says (a pass that started while the
        // switch was already off carries the already-current generation).
        if (strongSelf->_suspended.load()) {
            strongSelf->_eagerResumeIndex = static_cast<NSInteger>(index);
            return;
        }
        // The pass is a chain with one live block at a time. A block that finds the switch off parks the pass (the
        // index) and ends; the resume block restarts it from there. A block queued before a suspend -> resume cycle
        // that finds the switch on again is the pass's only live block (the resume block ran first and found nothing
        // parked, or runs after it and will find nothing): ending it would lose the pass, so it goes on. The
        // generation therefore needs no check here; the retry and resume blocks, which can be doubled, have one.
        if (strongSelf->_eagerResumeIndex == static_cast<NSInteger>(index)) {
            strongSelf->_eagerResumeIndex = -1;
        }
        dispatch_block_t hook = strongSelf->_eagerStepHookForTesting;
        if (hook != nil) {
            hook();
        }
        const EagerStep step = kEagerOrder[index];
        NSString *name = [NSString stringWithFormat:@"%c%zu", step.next ? 'n' : 't', step.length];
        BOOL skipped = decoder->isFunctionLoaded(step.next, step.length);  // a lazy load got there first
        BOOL ok = YES;
        double ms = 0;
        std::string detail;
        if (!skipped) {
            auto t0 = std::chrono::steady_clock::now();
            ok = decoder->precompileFunction(step.next, step.length, &detail);
            ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            strongSelf->_eagerMicroseconds.fetch_add(static_cast<uint64_t>(ms * 1000.0));
        }
        {
            std::lock_guard<std::mutex> lock(strongSelf->_lazyMutex);
            [strongSelf->_eagerLog addObject:@{ @"name" : name, @"ms" : @(ms), @"ok" : @(ok), @"skipped" : @(skipped) }];
        }
        if (!skipped) {
            [strongSelf _appendLogLine:[NSString stringWithFormat:@"# %@ eager compile %@ %.0f ms %@\n", [NSDate date], name, ms, ok ? @"ok" : [NSString stringWithFormat:@"failed (%s)", detail.c_str()]]];
            if (!ok) {
                NSLog(@"McBopomofoLM: SlothE eager compile of decoder %@ failed (%s); it loads on first need", name, detail.c_str());
            }
        }
        [strongSelf _eagerCompileStep:index + 1 decoder:decoder];
    });
}

// ---- the master switch (v3.4n)

- (void)suspendBackgroundWork
{
    _suspended.store(true);
    ++_workGeneration;
}

- (BOOL)backgroundWorkSuspended
{
    return _suspended.load();
}

- (void)resumeBackgroundWork
{
    if (!_suspended.exchange(false)) {
        return;
    }
    // Sol r48: the resume block checks the switch first, and its generation: a suspend -> resume pair that comes after
    // this resume queued its own block, so this one is stale and does nothing.
    const uint64_t generation = _workGeneration.load();
    __weak SlothERuntime *weakSelf = self;
    dispatch_async(_loadQueue, ^{
        SlothERuntime *strongSelf = weakSelf;
        if (strongSelf == nil || strongSelf->_suspended.load() || strongSelf->_workGeneration.load() != generation) {
            return;
        }
        if (strongSelf->_encoderRetryPending) {
            strongSelf->_encoderRetryPending = NO;
            [strongSelf _loadOnLoadQueue];
        } else if (strongSelf->_decoderRetryPending) {
            [strongSelf _runDecoderRetry];
        } else if (strongSelf->_eagerResumeIndex >= 0 && strongSelf->_decoder != nullptr) {
            NSUInteger index = static_cast<NSUInteger>(strongSelf->_eagerResumeIndex);
            strongSelf->_eagerResumeIndex = -1;
            [strongSelf _eagerCompileStep:index decoder:strongSelf->_decoder];
        }
    });
}

// The decoder's delayed second load (load queue).
- (void)_runDecoderRetry
{
    _decoderRetryPending = NO;
    [self _loadDecoderOnLoadQueue];
    if (!_decoderRetryPending) {
        _loadingFlag.store(false, std::memory_order_release);
    }
}

- (void)startEagerCompilePassForTesting
{
    _eagerCompileAfterLoad = YES;
    [self _startEagerCompilePass];
}

- (BOOL)eagerCompileDone
{
    return _eagerDone.load(std::memory_order_acquire);
}

- (double)eagerCompileMilliseconds
{
    return _eagerMicroseconds.load() / 1000.0;
}

- (NSArray<NSDictionary<NSString *, id> *> *)eagerCompileLog
{
    std::lock_guard<std::mutex> lock(_lazyMutex);
    return [_eagerLog copy];
}

- (NSUInteger)loadAttemptsForTesting
{
    return _loadAttempts.load();
}

- (dispatch_queue_t)loadQueueForTesting
{
    return _loadQueue;
}

- (NSUInteger)probeRetryCount
{
    return _probeRetryCount.load();
}

// Core ML keeps its Neural Engine compile cache for this app in
// ~/Library/Caches/<bundle id>/com.apple.e5rt.e5bundlecache, keyed by model
// path. After the app is replaced at the same path, stale entries made
// MLComputePlan fail ("internal failure") until the cache was cleared
// (observed 2026-09-23). Only this app's own cache directory is removed.
+ (nullable NSString *)aneCachePath
{
    NSString *bundleID = NSBundle.mainBundle.bundleIdentifier;
    NSString *caches = NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES).firstObject;
    if (bundleID.length == 0 || caches == nil) {
        return nil;
    }
    return [[caches stringByAppendingPathComponent:bundleID] stringByAppendingPathComponent:@"com.apple.e5rt.e5bundlecache"];
}

// A damaged or stale compile cache (seen: an interrupted compile left
// "Invalid sentinel in blob_metadata" and the plan put 0% on the ANE; files
// replaced at the same path made MLComputePlan fail) is worth one clean
// recompile. A real CPU placement comes back the same after the retry.
- (void)noteLazyLoadFailureForTesting:(NSInteger)length reason:(NSString *)reason anePercent:(double)anePercent
{
    McBopomofoSlothE::CoreMLLoadInfo info;
    info.reason = reason.UTF8String;
    info.aneCostPercent = anePercent;
    info.detail = "simulated for a test";
    [self _noteLazyLoad:static_cast<size_t>(length) info:info];
}

// Set when a lazily loaded decoder function failed like a damaged cache;
// consumed by the next start-up (clear the cache once, then load).
+ (nullable NSString *)aneCacheClearMarkerPath
{
    NSString *cache = [SlothERuntime aneCachePath];
    return cache == nil ? nil : [cache.stringByDeletingLastPathComponent stringByAppendingPathComponent:@"slothe-clear-ane-cache-at-next-start"];
}

+ (void)setPlanFailuresForTesting:(NSInteger)count
{
    McBopomofoSlothE::SetPlanFailuresForTesting(static_cast<int>(count));
}

- (void)requestDecoderFunctionForTesting:(NSInteger)length next:(BOOL)next
{
    if (_decoder != nullptr) {
        _decoder->requestFunction(next, static_cast<size_t>(length));
    }
}

- (nullable NSString *)_markerPath
{
    return _aneCacheClearMarkerPathForTesting != nil ? _aneCacheClearMarkerPathForTesting : [SlothERuntime aneCacheClearMarkerPath];
}

- (nullable NSString *)_installMarkerPath
{
    return _aneCacheInstallMarkerPathForTesting != nil ? _aneCacheInstallMarkerPathForTesting : [SlothERuntime aneCacheInstallMarkerPath];
}

// v3.4n: two markers that ask the NEXT start-up to clear the compile cache, kept apart so that neither writer can
// touch the other's request:
//  - the INSTALL marker (`aneCacheInstallMarkerPath`), written only by `install`: always honoured, no cap. A new
//    executable at the same path must not meet the previous one's entries; install writes a marker instead of deleting
//    the cache because on an upgrade a new input method process may already be compiling between the file copy and
//    the install command.
//  - the SELF-HEAL marker (`aneCacheClearMarkerPath`, the name older builds used), written only by the runtime when a
//    lazily loaded decoder function failed its check even after its retry: honoured at most once per bundle version
//    (`slothe-ane-cache-cleared-for-version` next to it, written only by the consumer), so a Mac that keeps flaking
//    cannot loop through a 3-minute recompile at every start.
// Consuming is ONE sequence under ONE lock (Sol r48): flock(LOCK_EX) on `slothe-ane-cache.lock` next to the markers,
// held across claim -> clear -> write the cap -> delete the claims. Everything that consumes takes that lock, so
//  - a `.claimed-*` file found when the lock is taken is ABANDONED (its consumer crashed or was killed after the claim
//    and before the clear completed): it counts as a pending request of its own kind (an install claim stays an install
//    request, a self-heal claim a self-heal request, cap and all);
//  - the claim is deleted only after the clear has completed, so no crash can lose a request;
//  - the cap is read and written only here, under the lock: two consumers cannot both pass the check.
// Writers that do not take the lock (the runtime's self-heal writer) stay safe by the claim itself: a marker is CLAIMED
// by rename(2) to a unique name in the same directory (atomic), and a writer that comes after the claim creates a new
// file under the marker's name, which is not the claimed one and so is not deleted with it: at worst one extra compile,
// never a lost request. `install` takes the same lock only around writing its marker (it waits for at most one consume).
static NSString *const kCacheLockName = @"slothe-ane-cache.lock";

// -1 when the lock file cannot be opened (no cache folder yet: no marker either) or locked.
static int AcquireCacheLock(NSString *folder)
{
    int fd = open([folder stringByAppendingPathComponent:kCacheLockName].fileSystemRepresentation, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) {
        return -1;
    }
    while (flock(fd, LOCK_EX) != 0) {
        if (errno != EINTR) {
            close(fd);
            return -1;
        }
    }
    return fd;
}

static void ReleaseCacheLock(int fd)
{
    if (fd >= 0) {
        flock(fd, LOCK_UN);
        close(fd);
    }
}

// The claimed file's path, or nil (nothing there, or another process claimed it).
- (nullable NSString *)_claimMarkerAtPath:(NSString *)path
{
    NSString *claimed = [path stringByAppendingFormat:@".claimed-%d-%08x", getpid(), arc4random()];
    if (rename(path.fileSystemRepresentation, claimed.fileSystemRepresentation) != 0) {
        return nil;
    }
    dispatch_block_t hook = _markerClaimedHookForTesting;
    if (hook != nil) {
        hook();
    }
    return claimed;
}

// The `.claimed-*` files of one marker found in its folder (the caller holds the lock: they are abandoned).
static NSArray<NSString *> *AbandonedClaims(NSString *marker)
{
    NSString *folder = marker.stringByDeletingLastPathComponent;
    NSString *prefix = [marker.lastPathComponent stringByAppendingString:@".claimed-"];
    NSMutableArray<NSString *> *found = [NSMutableArray array];
    for (NSString *name in [NSFileManager.defaultManager contentsOfDirectoryAtPath:folder error:nil]) {
        if ([name hasPrefix:prefix]) {
            [found addObject:[folder stringByAppendingPathComponent:name]];
        }
    }
    return found;
}

// The real clear: remove this app's compile cache. YES when it is gone (or never was).
- (BOOL)_clearANECacheForMarker:(NSString *)why
{
    NSString *path = [SlothERuntime aneCachePath];
    if (path == nil) {
        return NO;
    }
    NSError *error = nil;
    BOOL removed = [NSFileManager.defaultManager removeItemAtPath:path error:&error];
    BOOL wasNotThere = error != nil && [error.domain isEqualToString:NSCocoaErrorDomain] && (error.code == NSFileNoSuchFileError || error.code == NSFileReadNoSuchFileError);
    if (!removed && !wasNotThere) {
        [self _appendLogLine:[NSString stringWithFormat:@"# %@ Neural Engine compile cache could NOT be cleared (%@): %@\n", [NSDate date], why, error.localizedDescription]];
        return NO;
    }
    _aneCacheCleared = YES;
    [self _appendLogLine:[NSString stringWithFormat:@"# %@ Neural Engine compile cache cleared (%@)\n", [NSDate date], why]];
    return YES;
}

// YES, with *why set: the compile cache was cleared by this call. `clear` replaces the real clearing (tests).
- (BOOL)_consumeANECacheClearMarker:(NSString *_Nullable *_Nonnull)why clear:(nullable BOOL (^)(NSString *))clear
{
    *why = nil;
    NSString *installMarker = [self _installMarkerPath];
    NSString *healMarker = [self _markerPath];
    NSString *folder = (installMarker != nil ? installMarker : healMarker).stringByDeletingLastPathComponent;
    if (folder.length == 0) {
        return NO;
    }
    int lock = AcquireCacheLock(folder);
    if (lock < 0) {
        return NO;  // no cache folder yet (so no marker), or the lock cannot be taken: the requests stay where they are
    }
    NSFileManager *fm = NSFileManager.defaultManager;
    NSMutableArray<NSString *> *installClaims = [NSMutableArray array];
    NSMutableArray<NSString *> *healClaims = [NSMutableArray array];
    if (installMarker != nil) {
        [installClaims addObjectsFromArray:AbandonedClaims(installMarker)];
    }
    if (healMarker != nil) {
        [healClaims addObjectsFromArray:AbandonedClaims(healMarker)];
    }
    if (installClaims.count + healClaims.count > 0) {
        [self _appendLogLine:[NSString stringWithFormat:@"# %@ found %lu claimed compile-cache request(s) a previous start-up did not finish: treated as pending\n", [NSDate date], (unsigned long)(installClaims.count + healClaims.count)]];
    }
    NSString *claimed = installMarker != nil ? [self _claimMarkerAtPath:installMarker] : nil;
    if (claimed != nil) {
        [installClaims addObject:claimed];
    }
    claimed = healMarker != nil ? [self _claimMarkerAtPath:healMarker] : nil;
    if (claimed != nil) {
        [healClaims addObject:claimed];
    }

    NSString *reason = nil;
    NSString *capPath = healMarker != nil ? [healMarker.stringByDeletingLastPathComponent stringByAppendingPathComponent:@"slothe-ane-cache-cleared-for-version"] : nil;
    NSString *version = _bundleVersionForTesting != nil ? _bundleVersionForTesting : [NSBundle.mainBundle objectForInfoDictionaryKey:@"CFBundleVersion"];
    if (version.length == 0) {
        version = @"?";
    }
    BOOL writeCap = NO;
    if (installClaims.count > 0) {
        // One clear serves both; the self-heal request, if any, is satisfied by it and does not use up the version's
        // one self-heal clear.
        reason = @"the install asked for a clean compile";
    } else if (healClaims.count > 0 && capPath != nil) {
        NSString *done = [NSString stringWithContentsOfFile:capPath encoding:NSUTF8StringEncoding error:nil];
        if ([done isEqualToString:version]) {
            [self _appendLogLine:[NSString stringWithFormat:@"# %@ a decoder function failed its check last run, but the compile cache was already cleared once for version %@: not clearing again\n", [NSDate date], version]];
            NSLog(@"McBopomofoLM: not clearing the Neural Engine compile cache again for version %@", version);
        } else {
            reason = @"a decoder function failed its check last run";
            writeCap = YES;
        }
    }
    BOOL cleared = NO;
    BOOL finished = YES;
    if (reason != nil) {
        BOOL (^action)(NSString *) = clear != nil ? clear : _clearActionForTesting;
        cleared = action != nil ? action(reason) : [self _clearANECacheForMarker:reason];
        if (cleared) {
            if (writeCap) {
                [version writeToFile:capPath atomically:YES encoding:NSUTF8StringEncoding error:nil];
            }
        } else {
            finished = NO;  // the clear did not complete: the claims stay, and the next start-up finds them
        }
    }
    if (finished) {
        for (NSString *path in installClaims) {
            [fm removeItemAtPath:path error:nil];
        }
        for (NSString *path in healClaims) {
            [fm removeItemAtPath:path error:nil];
        }
    }
    ReleaseCacheLock(lock);
    if (cleared) {
        *why = reason;
    }
    return cleared;
}

+ (BOOL)writeANECacheClearMarkerForInstall
{
    NSString *marker = [SlothERuntime aneCacheInstallMarkerPath];
    return marker != nil && [self writeANECacheClearMarkerForInstallAtPath:marker];
}

// The marker itself, with the cache lock ALREADY held by the caller (or deliberately not taken).
static BOOL WriteInstallMarkerWithoutLock(NSString *marker)
{
    return [@"install" writeToFile:marker atomically:YES encoding:NSUTF8StringEncoding error:nil];
}

+ (BOOL)writeANECacheClearMarkerForInstallAtPath:(NSString *)marker
{
    NSString *folder = marker.stringByDeletingLastPathComponent;
    [NSFileManager.defaultManager createDirectoryAtPath:folder withIntermediateDirectories:YES attributes:nil error:nil];
    // The cache lock, only for the write: a consume in progress finishes first (at most one; it takes milliseconds).
    // Without the lock (it cannot be opened) the marker is still written: the request matters more than the order.
    int lock = AcquireCacheLock(folder);
    BOOL ok = WriteInstallMarkerWithoutLock(marker);
    ReleaseCacheLock(lock);
    return ok;
}

+ (nullable NSString *)aneCacheInstallMarkerPath
{
    NSString *cache = [SlothERuntime aneCachePath];
    return cache == nil ? nil : [cache.stringByDeletingLastPathComponent stringByAppendingPathComponent:@"slothe-clear-ane-cache-after-install"];
}

+ (BOOL)anyModelLoadStartedInThisProcess
{
    return gAnyModelLoadStarted.load();
}

- (BOOL)consumeANECacheClearMarkerForTesting:(NSString *_Nullable *_Nullable)why
{
    NSString *reason = nil;
    // clearActionForTesting replaces the clearing; without one nothing is cleared (the cache would be the real one)
    BOOL clear = [self _consumeANECacheClearMarker:&reason clear:_clearActionForTesting != nil ? _clearActionForTesting : ^BOOL(NSString *) { return YES; }];
    if (why != nullptr) {
        *why = reason;
    }
    return clear;
}

- (BOOL)verifyModelFilesWithDetail:(NSString *_Nullable *_Nullable)detail milliseconds:(nullable double *)milliseconds
{
    auto t0 = std::chrono::steady_clock::now();
    std::string dir(_resourcePath.fileSystemRepresentation);
    std::string error;
    BOOL ok = McBopomofoSlothE::VerifyRuntimeFiles(dir, McBopomofoSlothE::EncoderRuntimeFiles(), &error)
        && McBopomofoSlothE::VerifyRuntimeFiles(dir, McBopomofoSlothE::DecoderRuntimeFiles(), &error);
    if (milliseconds != nullptr) {
        *milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    if (!ok && detail != nullptr) {
        *detail = @(error.c_str());
    }
    return ok;
}

- (BOOL)_looksLikeABadANECache:(const McBopomofoSlothE::CoreMLLoadInfo &)info
{
    if (_computeUnitsCPUOnlyForTesting) {
        return NO;
    }
    return info.reason == "plan_failed" || (info.reason == "not_on_ane" && info.aneCostPercent >= 0 && info.aneCostPercent < 1.0);
}

- (BOOL)_clearANECacheOnce:(NSString *)why
{
    if (_aneCacheCleared) {
        return NO;
    }
    _aneCacheCleared = YES;
    NSString *path = [SlothERuntime aneCachePath];
    if (path == nil) {
        return NO;
    }
    [NSFileManager.defaultManager removeItemAtPath:path error:nil];
    [self _appendLogLine:[NSString stringWithFormat:@"# %@ Neural Engine compile cache cleared (%@)\n", [NSDate date], why]];
    return YES;
}

#pragma mark - The model store (v3.4p)

static NSString *const kBundleIdentifierPrefix = @"org.openvanilla.inputmethod.";

+ (NSString *)modelStoreRootForBundleIdentifier:(NSString *)bundleID
{
    NSString *appSupport = NSSearchPathForDirectoriesInDomains(NSApplicationSupportDirectory, NSUserDomainMask, YES).firstObject;
    if (appSupport == nil) {
        appSupport = [NSHomeDirectory() stringByAppendingPathComponent:@"Library/Application Support"];
    }
    NSString *folder;
    if (bundleID.length == 0) {
        folder = @"McBopomofoLM.unknown";
    } else if ([bundleID hasPrefix:kBundleIdentifierPrefix] && bundleID.length > kBundleIdentifierPrefix.length) {
        folder = [bundleID substringFromIndex:kBundleIdentifierPrefix.length];  // McBopomofoLM, McBopomofoLM.dev
    } else {
        folder = bundleID;
    }
    return [[appSupport stringByAppendingPathComponent:folder] stringByAppendingPathComponent:@"Models"];
}

+ (NSString *)modelStoreRoot
{
    if ([SlothERuntime runningUnderXCTest]) {
        // a unit-test host never touches Application Support: a temp folder, unless a test names its own root
        return [NSTemporaryDirectory() stringByAppendingPathComponent:@"McBopomofoLM-test-models"];
    }
    return [SlothERuntime modelStoreRootForBundleIdentifier:NSBundle.mainBundle.bundleIdentifier];
}

- (NSString *)_resolvedModelStoreRoot
{
    return _modelStoreRootForTesting.length > 0 ? _modelStoreRootForTesting : [SlothERuntime modelStoreRoot];
}

// The bundle's manifest: the one the app's signature seals. Every copy of the models (the store) is checked against it.
- (std::string)_bundleManifestPath
{
    return std::string(_resourcePath.fileSystemRepresentation) + "/" + McBopomofoSlothE::kRuntimeManifestName;
}

- (NSString *)_modelDirectoryForLoad
{
    @synchronized(self) {
        return _modelDir != nil ? _modelDir : _resourcePath;
    }
}

- (nullable NSString *)modelDirectory
{
    @synchronized(self) {
        return _modelDir;
    }
}

- (nullable NSString *)modelStoreStatus
{
    @synchronized(self) {
        return _modelStoreStatus;
    }
}

- (void)_setModelDirectory:(NSString *)directory status:(NSString *)status
{
    @synchronized(self) {
        _modelDir = [directory copy];
        _modelStoreStatus = [status copy];
    }
}

- (NSUInteger)modelStoreSetsCollected
{
    return _storeSetsCollected.load();
}

- (NSUInteger)modelStoreTemporariesCollected
{
    return _storeTemporariesCollected.load();
}

static SlothEModelStoreResult *WrapStoreResult(const McBopomofoSlothE::ModelStoreResult &r, std::shared_ptr<McBopomofoSlothE::ModelSetLease> *lease)
{
    SlothEModelStoreResult *result = [[SlothEModelStoreResult alloc] init];
    result.ok = r.ok;
    result.created = r.ok && r.created;
    result.setID = r.setId.empty() ? nil : @(r.setId.c_str());
    result.directory = r.ok ? @(r.directory.c_str()) : nil;
    result.error = r.ok ? nil : @(r.error.c_str());
    if (lease != nullptr) {
        *lease = r.lease;
    }
    return result;
}

- (SlothEModelStoreResult *)ensureModelStoreAtRoot:(nullable NSString *)root
{
    NSString *useRoot = root.length > 0 ? root : [self _resolvedModelStoreRoot];
    McBopomofoSlothE::ModelStoreResult r = McBopomofoSlothE::EnsureModelStore(std::string(_resourcePath.fileSystemRepresentation), std::string(useRoot.fileSystemRepresentation));
    return WrapStoreResult(r, nullptr);
}

// Verify against the start-up snapshot of the bundle's manifest; the file only when there is no snapshot (it could not be
// read at start-up: then the check says so, as it always did).
- (bool)_verifyFilesInDirectory:(const std::string &)dir names:(const std::vector<std::string> &)names error:(std::string *)error
{
    if (_manifest != nullptr) {
        return McBopomofoSlothE::VerifyRuntimeFiles(dir, names, error, *_manifest);
    }
    return McBopomofoSlothE::VerifyRuntimeFiles(dir, names, error, [self _bundleManifestPath]);
}

// Load queue, once, before the first load: read the bundle's manifest (the snapshot every later check uses), make sure the
// store is there (this covers an install done by `ditto` without `install`, and a store the user deleted), take the lease
// on its set and point the loads at it. If that fails (disk full, permissions, a redirected or foreign folder, a damaged
// set another process is still reading), the models are loaded from the bundle as before.
- (void)_prepareModelStore
{
    if (_modelStorePrepared) {
        return;
    }
    _modelStorePrepared = YES;
    auto manifest = std::make_shared<McBopomofoSlothE::ModelManifest>();
    std::string manifestError;
    if (McBopomofoSlothE::LoadModelManifest(std::string(_resourcePath.fileSystemRepresentation), manifest.get(), &manifestError)) {
        _manifest = manifest;
    }
    if (!_useModelStore && _modelStoreRootForTesting.length == 0) {
        [self _setModelDirectory:_resourcePath status:@"off"];
        return;
    }
    NSString *root = [self _resolvedModelStoreRoot];
    auto t0 = std::chrono::steady_clock::now();
    McBopomofoSlothE::ModelStoreResult r;
    if (_manifest != nullptr) {
        r = McBopomofoSlothE::EnsureModelStore(*_manifest, std::string(_resourcePath.fileSystemRepresentation), std::string(root.fileSystemRepresentation));
    } else {
        r.error = manifestError;
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::shared_ptr<McBopomofoSlothE::ModelSetLease> lease;
    SlothEModelStoreResult *result = WrapStoreResult(r, &lease);
    if (result.ok) {
        _lease = lease;
        _modelStoreSetID = result.setID;
        _modelStoreActiveRoot = root;
        [self _setModelDirectory:result.directory status:result.created ? @"created" : @"reused"];
        [self _appendLogLine:[NSString stringWithFormat:@"# %@ model store %@ in %.0f ms: %@\n", [NSDate date], result.created ? @"created" : @"reused", ms, result.directory]];
        [self _progress:[NSString stringWithFormat:@"  model store %@ (%.0f ms)", result.created ? @"created" : @"reused", ms]];
        return;
    }
    [self _setModelDirectory:_resourcePath status:@"fallback"];
    [self _appendLogLine:[NSString stringWithFormat:@"# %@ model store unavailable (%@): loading the models from the app\n", [NSDate date], result.error]];
    NSLog(@"McBopomofoLM: model store unavailable (%@); the models are loaded from the app", result.error);
    [self _progress:[NSString stringWithFormat:@"  model store unavailable (%@): loading from the app", result.error]];
    [self _noteModelStoreFallback];
}

- (void)prepareModelStoreForTesting
{
    dispatch_sync(_loadQueue, ^{
        [self _prepareModelStore];
    });
}

- (void)collectModelStoreGarbageForTesting
{
    dispatch_sync(_loadQueue, ^{
        [self _collectModelStoreGarbageAfterLoad];
    });
}

- (void)noteModelStoreFallbackForTesting
{
    [self _noteModelStoreFallback];
}

- (NSUInteger)modelStoreSetsKeptForLeases
{
    return _storeSetsKeptForLeases.load();
}

// The store failed and the models load from the bundle path: its compile cache may hold entries of an older copy of
// the files, so the install marker is written (the load that follows honours it at once). Once per CFBundleVersion
// (`slothe-model-store-fallback-for-version` next to the markers): a Mac whose disk stays full must not recompile at
// every start. Sol r49: the stamp is checked, the marker published and the stamp written in ONE transaction under the
// cache lock (the same flock as the marker consume and the install writer; no nested acquisition), so two runtimes that
// fall back together publish one marker for the version.
- (void)_noteModelStoreFallback
{
    NSString *marker = [self _installMarkerPath];
    if (marker == nil) {
        return;
    }
    NSString *folder = marker.stringByDeletingLastPathComponent;
    [NSFileManager.defaultManager createDirectoryAtPath:folder withIntermediateDirectories:YES attributes:nil error:nil];
    int lock = AcquireCacheLock(folder);
    if (lock < 0) {
        return;  // the lock cannot be taken: no request this time (the next start tries again)
    }
    NSString *stamp = [folder stringByAppendingPathComponent:@"slothe-model-store-fallback-for-version"];
    NSString *version = _bundleVersionForTesting != nil ? _bundleVersionForTesting : [NSBundle.mainBundle objectForInfoDictionaryKey:@"CFBundleVersion"];
    if (version.length == 0) {
        version = @"?";
    }
    NSString *done = [NSString stringWithContentsOfFile:stamp encoding:NSUTF8StringEncoding error:nil];
    if (![done isEqualToString:version]) {
        dispatch_block_t hook = _fallbackPublishHookForTesting;
        if (hook != nil) {
            hook();
        }
        if (WriteInstallMarkerWithoutLock(marker)) {
            [version writeToFile:stamp atomically:YES encoding:NSUTF8StringEncoding error:nil];
        }
    }
    ReleaseCacheLock(lock);
}

// Only here, never in `install`: other sets and leftover temporary folders go after this runtime's own successful load,
// except the set a runtime still holds a lease on and the one the bundle on disk names now (Sol r49). A runtime that fell
// back to the bundle cleans nothing.
- (void)_collectModelStoreGarbageAfterLoad
{
    if (_modelStoreCollected || _modelStoreSetID == nil || _modelStoreActiveRoot == nil) {
        return;
    }
    _modelStoreCollected = YES;
    McBopomofoSlothE::ModelStoreCleanup cleanup = McBopomofoSlothE::CollectModelStoreGarbage(std::string(_modelStoreActiveRoot.fileSystemRepresentation), std::string(_modelStoreSetID.UTF8String), std::string(_resourcePath.fileSystemRepresentation));
    _storeSetsCollected.store(cleanup.sets);
    _storeTemporariesCollected.store(cleanup.temporaries);
    _storeSetsKeptForLeases.store(cleanup.leased);
    if (cleanup.sets + cleanup.temporaries + cleanup.leased > 0) {
        [self _appendLogLine:[NSString stringWithFormat:@"# %@ model store clean-up: %zu old set(s), %zu leftover folder(s) deleted, %zu set(s) kept (in use by another input method process)\n", [NSDate date], cleanup.sets, cleanup.temporaries, cleanup.leased]];
    }
}

- (BOOL)loadForInstallWithProgress:(void (^)(NSString *line))progress
{
    _installProgress = [progress copy];
    _loadAllDecoderFunctions = YES;  // compile every function, so later lazy loads are cache loads
    // A (re)install compiles from scratch: entries left by a previous copy of the
    // app at this path can make the placement check fail.
    if (!_keepANECacheForTesting) {
        [self _clearANECacheOnce:@"install"];
        _aneCacheCleared = NO;  // the load may still self-heal once
        if (progress != nil) {
            progress(@"Cleared this app's Neural Engine compile cache: the models compile from scratch.");
        }
    }
    _loadingFlag.store(true);
    auto t0 = std::chrono::steady_clock::now();
    dispatch_sync(_loadQueue, ^{
        [self _loadOnLoadQueue];
    });
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (progress != nil) {
        progress([NSString stringWithFormat:@"Models ready in %.1f s: encoder %@, decoder %@.", s,
                  _loadedFlag.load() ? @"on the Neural Engine" : [@"not used: " stringByAppendingString:(_loadReason != nil ? _loadReason : @"?")],
                  _decoderLoadedFlag.load() ? @"on the Neural Engine" : [@"not used: " stringByAppendingString:(_decoderLoadReason != nil ? _decoderLoadReason : @"?")]]);
    }
    _installProgress = nil;
    [self flushLog];
    return _loadedFlag.load() && _decoderLoadedFlag.load();
}

#pragma mark - Prewarm

- (NSUInteger)prewarmCount
{
    return _prewarmCount;
}

- (NSUInteger)prewarmsCompleted
{
    return _prewarmDone.load();
}

- (BOOL)prewarmIfIdle
{
    McBopomofoSlothE::Engine *engine = [self engine];
    if (engine == nullptr) {
        return NO;
    }
    uint64_t now = McBopomofoSlothE::MonotonicNanoseconds();
    uint64_t last = std::max(McBopomofoSlothE::LastModelUseNanoseconds(), _lastPrewarmNs.load());
    double idleMs = last == 0 ? -1 : static_cast<double>(now - last) / 1e6;
    if (last != 0 && idleMs <= _prewarmIdleSeconds * 1000.0) {
        return NO;
    }
    _lastPrewarmNs.store(now);
    ++_prewarmCount;
    McBopomofoSlothE::Decoder *decoder = [self decoder];
    dispatch_async(_computeQueue, ^{
        double encMs = 0;
        double decMs = -1;
        engine->prewarm(&encMs);
        if (decoder != nullptr) {
            decoder->prewarm(&decMs);
        }
        self->_prewarmDone.fetch_add(1);
        NSString *dec = decMs >= 0 ? [NSString stringWithFormat:@"%.2f", decMs] : @"";
        [self _appendTimestampedLine:@"P" tail:[NSString stringWithFormat:@"\t%.0f\t%.2f\t%@\n", idleMs, encMs, dec]];
    });
    return YES;
}

- (nullable NSString *)loadReason
{
    return _loadReason;
}

- (nullable NSString *)decoderLoadReason
{
    return _decoderLoadReason;
}

- (NSDictionary<NSString *, NSNumber *> *)placementSummary
{
    NSMutableDictionary *d = [NSMutableDictionary dictionary];
    d[@"encoderANECostPercent"] = @(_encInfo.aneCostPercent);
    d[@"encoderProbeMs"] = @(_encInfo.probeMs);
    d[@"encoderLoadMs"] = @(_encInfo.loadMs);
    d[@"encoderPlanMs"] = @(_encInfo.planMs);
    d[@"decoderANECostPercent"] = @(_decInfo.aneCostPercent);
    d[@"decoderProbeMs"] = @(_decInfo.probeMs);
    d[@"decoderLoadMs"] = @(_decInfo.loadMs);
    d[@"decoderPlanMs"] = @(_decInfo.planMs);
    d[@"verifyMs"] = @(_verifyMs);
    for (const auto &[fn, ms] : _encInfo.functionLoadMs) {
        d[[@"encoderLoadMs." stringByAppendingString:@(fn.c_str())]] = @(ms);
    }
    for (const auto &[fn, ms] : _decInfo.functionLoadMs) {
        d[[@"decoderLoadMs." stringByAppendingString:@(fn.c_str())]] = @(ms);
    }
    return d;
}

#pragma mark - Key event timing

- (void)beginKeystroke
{
    _ksWalkMs = 0;
    _ksInWalkMs = 0;
    _ksInWalkRan = NO;
    _ksProvMs = 0;
    _ksProvRan = NO;
    _ksForward = @"none";
    _ksRerankMs = 0;
    _ksRerankRan = NO;
    _ksSettle = @"none";
    _ksSettleMs = 0;
}

- (void)noteStockWalkMilliseconds:(double)milliseconds
{
    _ksWalkMs += milliseconds;
}

- (void)noteInWalkMilliseconds:(double)milliseconds
{
    _ksInWalkRan = YES;
    _ksInWalkMs += milliseconds;
}

- (void)noteProvisionalMilliseconds:(double)milliseconds
{
    _ksProvRan = YES;
    _ksProvMs += milliseconds;
}

- (void)noteForwardQueued
{
    _ksForward = @"queued";
}

- (void)noteForwardFromCache
{
    if (![_ksForward isEqualToString:@"queued"]) {
        _ksForward = @"cached";
    }
}

- (void)noteRerankMilliseconds:(double)milliseconds
{
    _ksRerankRan = YES;
    _ksRerankMs += milliseconds;
}

- (void)noteSettle:(NSString *)kind milliseconds:(double)milliseconds
{
    _ksSettle = [kind copy];
    _ksSettleMs += milliseconds;
    if ([kind isEqualToString:@"waited"]) {
        ++_commitWaitCount;
    } else if ([kind isEqualToString:@"timeout"]) {
        ++_commitTimeoutCount;
    } else if ([kind isEqualToString:@"partial"]) {
        ++_commitPartialCount;
    }
}

- (void)endKeystrokeWithBufferLength:(NSUInteger)bufferLength handlerMilliseconds:(double)milliseconds state:(NSString *)state
{
    if (![self _logging]) {
        return;
    }
    NSString *inWalk = _ksInWalkRan ? [NSString stringWithFormat:@"%.3f", _ksInWalkMs] : @"";
    NSString *prov = _ksProvRan ? [NSString stringWithFormat:@"%.3f", _ksProvMs] : @"";
    NSString *rerank = _ksRerankRan ? [NSString stringWithFormat:@"%.3f", _ksRerankMs] : @"";
    NSString *settleMs = [_ksSettle isEqualToString:@"none"] ? @"" : [NSString stringWithFormat:@"%.3f", _ksSettleMs];
    NSString *tail = [NSString stringWithFormat:@"\t%lu\t%.3f\t%.3f\t%@\t%@\t%@\t%@\t%@\t%@\t%@\n", (unsigned long)bufferLength, milliseconds, _ksWalkMs, inWalk, prov, _ksForward, rerank, _ksSettle, settleMs, state];
    [self _appendTimestampedLine:@"K" tail:tail];
}

- (void)noteAsyncResult:(NSString *)result bufferLength:(NSUInteger)bufferLength queueMilliseconds:(double)queueMilliseconds forwardMilliseconds:(double)forwardMilliseconds decoderMilliseconds:(double)decoderMilliseconds decoderCalls:(NSInteger)decoderCalls pins:(NSInteger)pins decoder:(NSString *)decoder inWalkMilliseconds:(double)inWalkMilliseconds endToEndMilliseconds:(double)endToEndMilliseconds
{
    if ([result isEqualToString:@"applied"]) {
        ++_appliedCount;
    } else if ([result isEqualToString:@"unchanged"]) {
        ++_unchangedCount;
    } else if ([result isEqualToString:@"stale"]) {
        ++_staleCount;
    } else if ([result isEqualToString:@"skipped"]) {
        ++_skippedCount;
    } else if ([result isEqualToString:@"deferred"]) {
        ++_deferredCount;
    } else if ([result isEqualToString:@"consumed"]) {
        ++_consumedCount;
    }
    if ([decoder isEqualToString:@"scored"]) {
        ++_decoderRunCount;
    } else if ([decoder isEqualToString:@"stale"]) {
        ++_decoderStaleCount;
    }
    if ([result isEqualToString:@"applied"] || [result isEqualToString:@"unchanged"]) {
        _decoderPinCount += static_cast<NSUInteger>(std::max<NSInteger>(pins, 0));
    }
    if (endToEndMilliseconds >= 0) {
        [_e2e addObject:@(endToEndMilliseconds)];
        if (_e2e.count > kMaxEndToEndSamples) {
            [_e2e removeObjectAtIndex:0];
        }
    }
    if (![self _logging]) {
        return;
    }
    NSString *dec = [decoder isEqualToString:@"off"] ? @"" : [NSString stringWithFormat:@"%.3f", decoderMilliseconds];
    NSString *inWalk = inWalkMilliseconds > 0 ? [NSString stringWithFormat:@"%.3f", inWalkMilliseconds] : @"";
    NSString *e2e = endToEndMilliseconds >= 0 ? [NSString stringWithFormat:@"%.3f", endToEndMilliseconds] : @"";
    NSString *tail = [NSString stringWithFormat:@"\t%lu\t%.3f\t%.3f\t%@\t%ld\t%ld\t%@\t%@\t%@\t%@\n", (unsigned long)bufferLength, queueMilliseconds, forwardMilliseconds, dec, (long)decoderCalls, (long)pins, inWalk, e2e, result, decoder];
    [self _appendTimestampedLine:@"A" tail:tail];
}

- (void)notePauseGateWithBufferLength:(NSUInteger)bufferLength requests:(NSInteger)requests calls:(NSInteger)calls cached:(NSInteger)cached noDecision:(NSInteger)noDecision pins:(NSInteger)pins changedChars:(NSInteger)changedChars forwardMilliseconds:(double)forwardMilliseconds decoderMilliseconds:(double)decoderMilliseconds totalMilliseconds:(double)totalMilliseconds result:(NSString *)result
{
    ++_pauseGateRunCount;
    if ([result isEqualToString:@"applied"]) {
        ++_pauseGateAppliedCount;
        if (changedChars > 0) {
            ++_pauseGateChangedCount;
        }
    }
    if (![self _logging]) {
        return;
    }
    NSString *total = totalMilliseconds >= 0 ? [NSString stringWithFormat:@"%.3f", totalMilliseconds] : @"";
    NSString *tail = [NSString stringWithFormat:@"\t%lu\t%ld\t%ld\t%ld\t%ld\t%ld\t%ld\t%.3f\t%.3f\t%@\t%@\n", (unsigned long)bufferLength, (long)requests, (long)calls, (long)cached, (long)noDecision, (long)pins, (long)changedChars, forwardMilliseconds, decoderMilliseconds, total, result];
    [self _appendTimestampedLine:@"G" tail:tail];
}

- (void)noteLegalEntryEvent:(NSString *)event slot:(NSInteger)slot candidates:(NSUInteger)candidates readyMilliseconds:(double)readyMilliseconds trigger:(NSString *)trigger
{
    if ([event isEqualToString:@"offered"]) {
        ++_legalEntryOfferedCount;
    } else if ([event isEqualToString:@"chosen"]) {
        ++_legalEntryChosenCount;
    } else if ([event isEqualToString:@"not_ready"]) {
        ++_legalEntryNotReadyCount;
    }
    if (![self _logging]) {
        return;
    }
    NSString *slotText = slot > 0 ? [NSString stringWithFormat:@"%ld", (long)slot] : @"";
    NSString *ready = readyMilliseconds >= 0 ? [NSString stringWithFormat:@"%.3f", readyMilliseconds] : @"";
    [self _appendTimestampedLine:@"Y" tail:[NSString stringWithFormat:@"\t%@\t%@\t%lu\t%@\t%@\n", event, slotText, (unsigned long)candidates, ready, trigger]];
}

- (void)noteInputModeChange:(NSString *)mode source:(NSString *)source
{
    if ([mode isEqualToString:@"en"]) {
        ++_englishModeCount;
    }
    if (![self _logging]) {
        return;
    }
    [self _appendTimestampedLine:@"M" tail:[NSString stringWithFormat:@"\t%@\t%@\n", mode, source]];
}

- (void)noteModeHUDGeometry:(NSString *)geometry client:(NSString *)client
{
    if (![self _logging]) {
        return;
    }
    // bundle identifiers are ASCII; anything else is dropped so the log never holds text
    NSMutableString *app = [NSMutableString string];
    for (NSUInteger i = 0; i < client.length && i < 128; ++i) {
        unichar c = [client characterAtIndex:i];
        if (c > 0x20 && c < 0x7f && c != '\t') {
            [app appendFormat:@"%C", c];
        }
    }
    // the geometry is numbers, '/', ',', '-' and the words in the H doc; a tab separates its fields, never a newline
    NSString *fields = [[geometry stringByReplacingOccurrencesOfString:@"\n" withString:@" "] stringByReplacingOccurrencesOfString:@"\r" withString:@" "];
    [self _appendTimestampedLine:@"H" tail:[NSString stringWithFormat:@"\t%@\t%@\n", app.length ? app : @"-", fields]];
}

- (void)noteTypoWindowWithCandidates:(NSUInteger)count chosen:(NSString *)chosen readyMilliseconds:(double)readyMilliseconds source:(NSString *)source
{
    if (![self _logging]) {
        return;
    }
    NSString *ready = readyMilliseconds >= 0 ? [NSString stringWithFormat:@"%.3f", readyMilliseconds] : @"";
    [self _appendTimestampedLine:@"X" tail:[NSString stringWithFormat:@"\t%lu\t%@\t%@\t%@\n", (unsigned long)count, chosen, ready, source]];
}

- (void)submitSideComputeJob:(dispatch_block_t)job
{
    dispatch_async(_computeQueue, job);
}

- (BOOL)loadNextTokenFunctionsForTesting:(NSTimeInterval)timeout
{
    McBopomofoSlothE::Decoder *decoder = [self decoder];
    if (decoder == nullptr) {
        return NO;
    }
    BOOL all = YES;
    NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
    for (NSUInteger chars : { 1, 20, 40, 64 }) {  // one left text per function length (n16 / n32 / n64 / n96)
        std::string text;
        for (NSUInteger k = 0; k < chars; ++k) {
            text += "一";
        }
        std::vector<float> lp;
        McBopomofoSlothE::DecoderCallStats stats;
        BOOL ok = NO;
        while (!(ok = decoder->nextTokenLogProbs(text, &lp, &stats)) && deadline.timeIntervalSinceNow > 0) {
            [NSThread sleepForTimeInterval:0.2];
        }
        all = all && ok;
    }
    return all;
}

- (void)noteBufferStartWithContextLength:(NSUInteger)contextLength reset:(NSString *)reset selection:(NSString *)selection client:(NSString *)client passThrough:(NSUInteger)passThrough
{
    if (![self _logging]) {
        return;
    }
    // bundle identifiers are ASCII; anything else is dropped so the log never holds text
    NSMutableString *app = [NSMutableString string];
    for (NSUInteger i = 0; i < client.length && i < 128; ++i) {
        unichar c = [client characterAtIndex:i];
        if (c > 0x20 && c < 0x7f && c != '\t') {
            [app appendFormat:@"%C", c];
        }
    }
    [self _appendTimestampedLine:@"C" tail:[NSString stringWithFormat:@"\t%lu\t%@\t%@\t%@\t%lu\n", (unsigned long)contextLength, reset, selection, app.length ? app : @"-", (unsigned long)passThrough]];
}

- (void)_appendTimestampedLine:(NSString *)kind tail:(NSString *)tail
{
    uint64_t token = _logState.load(std::memory_order_acquire);  // v3.3h: guards every record, R and P included
    if (_logPath == nil || (token & 1) == 0) {
        return;
    }
    NSDate *now = [NSDate date];
    dispatch_block_t hook = _logEnqueueRaceHookForTesting;
    if (hook != nil) {
        hook();
    }
    dispatch_async(_logQueue, ^{
        if (self->_logState.load(std::memory_order_acquire) != token) {
            return;
        }
        [self _writeLogLine:[NSString stringWithFormat:@"%@\t%@%@", kind, [self _timestamp:now], tail]];
    });
}

- (void)flushLog
{
    dispatch_sync(_logQueue, ^{
        [self->_logHandle synchronizeAndReturnError:nil];
    });
}

#pragma mark - Log file (log queue)

- (void)_appendLogLine:(NSString *)line
{
    uint64_t token = _logState.load(std::memory_order_acquire);
    if (_logPath == nil || (token & 1) == 0) {
        return;
    }
    dispatch_block_t hook = _logEnqueueRaceHookForTesting;
    if (hook != nil) {
        hook();
    }
    dispatch_async(_logQueue, ^{
        if (self->_logState.load(std::memory_order_acquire) != token) {
            return;
        }
        [self _writeLogLine:line];
    });
}

- (NSString *)_timestamp:(NSDate *)date
{
    if (_logFormatter == nil) {
        _logFormatter = [[NSISO8601DateFormatter alloc] init];
        _logFormatter.formatOptions = NSISO8601DateFormatWithInternetDateTime | NSISO8601DateFormatWithFractionalSeconds;
        _logFormatter.timeZone = NSTimeZone.localTimeZone;
    }
    return [_logFormatter stringFromDate:date];
}

- (void)_writeLogLine:(NSString *)line
{
    // log queue only, and only for a record whose token still matches (checked by the caller)
    if (_logHandle == nil && ![self _openLog]) {
        return;
    }
    NSError *error = nil;
    if (![_logHandle writeData:[line dataUsingEncoding:NSUTF8StringEncoding] error:&error]) {
        [_logHandle closeAndReturnError:nil];
        _logHandle = nil;
    }
}

- (BOOL)_openLog
{
    NSFileManager *fm = NSFileManager.defaultManager;
    NSString *dir = _logPath.stringByDeletingLastPathComponent;
    if (![fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil]) {
        return NO;
    }
    NSDictionary *attrs = [fm attributesOfItemAtPath:_logPath error:nil];
    if (attrs != nil) {  // a log from another format version (v1-v3 build) is kept aside, not appended to
        NSFileHandle *h = [NSFileHandle fileHandleForReadingAtPath:_logPath];
        NSData *head = [h readDataOfLength:kLatencyLogVersionPrefix.length];
        [h closeFile];
        if (![[[NSString alloc] initWithData:head encoding:NSUTF8StringEncoding] isEqualToString:kLatencyLogVersionPrefix]) {
            NSString *old = [_logPath stringByAppendingPathExtension:@"pre-v4"];
            [fm removeItemAtPath:old error:nil];
            [fm moveItemAtPath:_logPath toPath:old error:nil];
            attrs = nil;
        }
    }
    if (attrs != nil && attrs.fileSize > kMaxLatencyLogBytes) {
        NSString *rotated = [_logPath stringByAppendingPathExtension:@"1"];
        [fm removeItemAtPath:rotated error:nil];
        [fm moveItemAtPath:_logPath toPath:rotated error:nil];
        attrs = nil;
    }
    if (attrs == nil && ![fm createFileAtPath:_logPath contents:[kLatencyLogHeader dataUsingEncoding:NSUTF8StringEncoding] attributes:nil]) {
        return NO;
    }
    // a log made before v3.2 has no doc for the new line types: append it once (the G ... doc block); a log
    // made by v3.2c / v3.2d has them but not the v3.2e C-line note: append only that note, once
    BOOL needsV32Doc = NO;
    BOOL needsV32eDoc = NO;
    BOOL needsHDoc = NO;
    if (attrs != nil) {
        // (the log is at most 4 MB and this runs once per process start)
        NSData *all = [NSData dataWithContentsOfFile:_logPath options:NSDataReadingMappedIfSafe error:nil];
        auto has = ^BOOL(NSString *needle) {
            return all != nil && [all rangeOfData:[needle dataUsingEncoding:NSUTF8StringEncoding] options:0 range:NSMakeRange(0, all.length)].location != NSNotFound;
        };
        needsV32Doc = !has(kLatencyLogMDoc);
        needsV32eDoc = !needsV32Doc && !has(kLatencyLogNewestDoc);
        needsHDoc = !needsV32Doc && !needsV32eDoc && !has(kLatencyLogHDoc);  // (the two blocks above end with it)
    }
    _logHandle = [NSFileHandle fileHandleForWritingAtPath:_logPath];
    if (_logHandle == nil) {
        return NO;
    }
    [_logHandle seekToEndReturningOffset:nil error:nil];
    if (needsV32Doc) {
        NSRange g = [kLatencyLogHeader rangeOfString:kLatencyLogGMarker];
        [_logHandle writeData:[[kLatencyLogHeader substringFromIndex:g.location] dataUsingEncoding:NSUTF8StringEncoding] error:nil];
    } else if (needsV32eDoc) {
        NSRange e = [kLatencyLogHeader rangeOfString:kLatencyLogNewestDoc];
        [_logHandle writeData:[[kLatencyLogHeader substringFromIndex:e.location] dataUsingEncoding:NSUTF8StringEncoding] error:nil];
    } else if (needsHDoc) {
        NSRange h = [kLatencyLogHeader rangeOfString:kLatencyLogHDoc];
        [_logHandle writeData:[[kLatencyLogHeader substringFromIndex:h.location] dataUsingEncoding:NSUTF8StringEncoding] error:nil];
    }
    // Build marker, once per log open (= once per process start or rotation).
    NSString *marker = [NSString stringWithFormat:@"# McBopomofoLM v3.3h (candidate window jump-ahead C, delta %.1f; UNK split; 1-syllable LM order; carried context; correction window; typing-pause gate; correction entry; Shift toggle; keypad literals; pass-through caret fix; English direct output; pass-through carried only when confirmable; Backspace drops a context with gaps; English mode global for the process; diagnostic log written only while switched on; log switch serialized with writes) log opened %@\n", McBopomofoSlothE::kJumpAheadDelta, [self _timestamp:[NSDate date]]];
    [_logHandle writeData:[marker dataUsingEncoding:NSUTF8StringEncoding] error:nil];
    return YES;
}

@end
