// SlothE-T runtime for the McBopomofoLM side-by-side test build.

#import "SlothERuntime.h"

#include <mach/mach.h>

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
    @"#   model decoder.t32 / .t64 / .t96 = a decoder function loaded on first need (v2.1)\n"
    @"# P = prewarm: first key of a new buffer after the models idled > 1 s; one dummy call on each model:\n"
    @"#   P time idle_ms enc_ms dec_ms\n";
static NSString *const kLatencyLogVersionPrefix = @"# McBopomofoLM latency log v4";
NSNotificationName const SlothEPreferencesDidChangeNotification = @"SlothEPreferencesDidChange";

static const unsigned long long kMaxLatencyLogBytes = 4ull * 1024ull * 1024ull;
static const size_t kMaxEndToEndSamples = 4096;

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
    std::atomic<NSUInteger> _lazyLoadsRequested;
    std::atomic<NSUInteger> _lazyLoadsDone;
    std::atomic<NSUInteger> _lazyLoadsFailed;
    std::mutex _lazyMutex;
    std::map<size_t, McBopomofoSlothE::CoreMLLoadInfo> _lazyInfo;
    dispatch_queue_t _loadQueue;
    dispatch_queue_t _logQueue;
    NSFileHandle *_logHandle;              // log queue only
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
        shared = [[SlothERuntime alloc] initWithResourcePath:resources logPath:log];
    });
    return shared;
}

- (instancetype)initWithResourcePath:(NSString *)resourcePath logPath:(nullable NSString *)logPath
{
    self = [super init];
    if (self) {
        _resourcePath = [resourcePath copy];
        _logPath = [logPath copy];
        _loadedFlag.store(false);
        _loadingFlag.store(false);
        _failedFlag.store(false);
        _decoderLoadedFlag.store(false);
        _decoderFailedFlag.store(false);
        _e2e = [NSMutableArray array];
        _loadQueue = dispatch_queue_create("org.openvanilla.McBopomofoLM.slothe.load", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));
        _logQueue = dispatch_queue_create("org.openvanilla.McBopomofoLM.slothe.log", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));
        _computeQueue = dispatch_queue_create("org.openvanilla.McBopomofoLM.slothe.compute", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0));
        _commitWaitMilliseconds = 30;
        _prewarmIdleSeconds = 1.0;
        _minEncoderANECostPercent = 99.0;
        _minDecoderANECostPercent = 80.0;
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
    McBopomofoSlothE::DecoderParams params;
    params.lambda = 2.0;  // config A' (walk2/frozen_final2.json); v1 was lambda 3, tau 0.9
    params.tau = 0.5;
    params.topK = 3;
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
    if (_loadedFlag.load() || _failedFlag.load() || _loadingFlag.exchange(true)) {
        return;
    }
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
        std::string dir(_resourcePath.fileSystemRepresentation);
        options.verifyBeforeLazyLoad = [dir](std::string *error) {
            return McBopomofoSlothE::VerifyRuntimeFiles(dir, McBopomofoSlothE::DecoderRuntimeFiles(), error);
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
            NSString *marker = [SlothERuntime aneCacheClearMarkerPath];
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

- (void)_logRuntime:(NSString *)model info:(const McBopomofoSlothE::CoreMLLoadInfo &)info reason:(NSString *)reason
{
    NSString *pct = info.aneCostPercent >= 0 ? [NSString stringWithFormat:@"%.1f", info.aneCostPercent] : @"";
    NSString *probe = info.probeMs >= 0 ? [NSString stringWithFormat:@"%.2f", info.probeMs] : @"";
    NSString *plan = info.planChecked ? [NSString stringWithFormat:@"%.0f", info.planMs] : @"";
    NSMutableArray *fns = [NSMutableArray array];
    for (const auto &[fn, p] : info.functionANEPercent) {
        [fns addObject:[NSString stringWithFormat:@"%s:%.1f", fn.c_str(), p]];
    }
    NSString *tail = [NSString stringWithFormat:@"\t%@\t%@\t%@\t%.0f\t%@\t%@\t%@\t%@\n", model, [reason isEqualToString:@"ok"] ? @"ane" : @"none", reason, info.loadMs, pct, probe, plan, [fns componentsJoinedByString:@","]];
    [self _appendTimestampedLine:@"R" tail:tail];
}

// Encoder first (its result alone already drives the in-walk), then the
// decoder. Core ML only, CPU_AND_NE; a model that fails integrity, does not
// load, or is not placed on the ANE is not used -- no CPU fallback.
- (void)_loadOnLoadQueue
{
    if (_loadedFlag.load() || _failedFlag.load()) {
        _loadingFlag.store(false);
        return;
    }
    auto t0 = std::chrono::steady_clock::now();
    std::string dir(_resourcePath.fileSystemRepresentation);
    std::string error;
    NSString *reason = @"ok";
    std::unique_ptr<McBopomofoSlothE::Engine> engine;
    NSString *marker = [SlothERuntime aneCacheClearMarkerPath];
    if (marker != nil && [NSFileManager.defaultManager fileExistsAtPath:marker]) {
        [NSFileManager.defaultManager removeItemAtPath:marker error:nil];
        [self _clearANECacheOnce:@"a decoder function failed its check last run"];
        _aneCacheCleared = NO;  // this load may still self-heal once
        [self _progress:@"  a decoder function failed its Neural Engine check last run: compile cache cleared, compiling again"];
    }
    [self _progress:@"SlothE-T 25M encoder (enc25m.mlmodelc):"];
    bool ok = McBopomofoSlothE::VerifyRuntimeFiles(dir, McBopomofoSlothE::EncoderRuntimeFiles(), &error);
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
        [self _loadDecoderOnLoadQueue];
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
- (void)_loadDecoderOnLoadQueue
{
    auto t0 = std::chrono::steady_clock::now();
    std::string dir(_resourcePath.fileSystemRepresentation);
    std::string error;
    NSString *reason = @"ok";
    [self _progress:@"SlothE decoder (dec60m.mlmodelc):"];
    _decInfo = McBopomofoSlothE::CoreMLLoadInfo();
    auto decoder = std::make_unique<McBopomofoSlothE::Decoder>();
    bool ok = McBopomofoSlothE::VerifyRuntimeFiles(dir, McBopomofoSlothE::DecoderRuntimeFiles(), &error);
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
    } else {
        _decoderLoadReason = reason;
        _decoderLoadErrorString = [NSString stringWithFormat:@"%@: %s", reason, error.c_str()];
        _decoderFailedFlag.store(true, std::memory_order_release);
        NSLog(@"McBopomofoLM: SlothE decoder not used (%@); encoder-only in-walk", _decoderLoadErrorString);
        [self _progress:[NSString stringWithFormat:@"  decoder NOT used (%@): encoder-only choices", _decoderLoadErrorString]];
    }
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
    if (_logPath == nil) {
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
    if (_logPath == nil) {
        return;
    }
    NSString *dec = [decoder isEqualToString:@"off"] ? @"" : [NSString stringWithFormat:@"%.3f", decoderMilliseconds];
    NSString *inWalk = inWalkMilliseconds > 0 ? [NSString stringWithFormat:@"%.3f", inWalkMilliseconds] : @"";
    NSString *e2e = endToEndMilliseconds >= 0 ? [NSString stringWithFormat:@"%.3f", endToEndMilliseconds] : @"";
    NSString *tail = [NSString stringWithFormat:@"\t%lu\t%.3f\t%.3f\t%@\t%ld\t%ld\t%@\t%@\t%@\t%@\n", (unsigned long)bufferLength, queueMilliseconds, forwardMilliseconds, dec, (long)decoderCalls, (long)pins, inWalk, e2e, result, decoder];
    [self _appendTimestampedLine:@"A" tail:tail];
}

- (void)_appendTimestampedLine:(NSString *)kind tail:(NSString *)tail
{
    NSDate *now = [NSDate date];
    dispatch_async(_logQueue, ^{
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
    if (_logPath == nil) {
        return;
    }
    dispatch_async(_logQueue, ^{
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
    _logHandle = [NSFileHandle fileHandleForWritingAtPath:_logPath];
    if (_logHandle == nil) {
        return NO;
    }
    [_logHandle seekToEndReturningOffset:nil error:nil];
    return YES;
}

@end
