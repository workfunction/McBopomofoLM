// SlothE-T runtime for the McBopomofoLM side-by-side test build.

#import "SlothERuntime.h"

#include <mach/mach.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

static NSString *const kLatencyLogHeader =
    @"# McBopomofoLM latency log v3. No typed text is recorded; buf_len = number of readings in the buffer.\n"
    @"# K = one key event handled by the input method:\n"
    @"#   K time buf_len handler_ms walk_ms inwalk_ms prov_ms fwd rerank_ms settle settle_ms lm\n"
    @"#   walk_ms = stock walk; inwalk_ms = exact re-pick during the key; prov_ms = provisional re-pick with the\n"
    @"#   previous pass (anti-flicker); fwd = queued / none; settle = none / ready / waited / partial / timeout\n"
    @"#   (commit or candidate window waiting up to 30 ms for encoder+decoder); lm = on/off/loading/failed/plain\n"
    @"# A = one background pass (SlothE-T encoder, then the decoder) finished:\n"
    @"#   A time buf_len queue_ms fwd_ms dec_ms dec_calls pins inwalk_ms e2e_ms result decoder\n"
    @"#   e2e_ms = key event to result applied; result = applied / unchanged / stale / skipped / deferred / consumed;\n"
    @"#   decoder = off / none (no node to check) / scored / stale (abandoned, buffer changed) / failed\n";
NSNotificationName const SlothEPreferencesDidChangeNotification = @"SlothEPreferencesDidChange";

static const unsigned long long kMaxLatencyLogBytes = 4ull * 1024ull * 1024ull;
static const int kWarmGraphLengths = 8;
static const int kGraphCacheCapacity = 8;
static const size_t kMaxEndToEndSamples = 4096;

@implementation SlothERuntime {
    std::unique_ptr<McBopomofoSlothE::Engine> _engine;
    std::unique_ptr<McBopomofoSlothE::Decoder> _decoder;
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
    double _loadMs;
    double _verifyMs;
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
        _ksForward = @"none";
        _ksSettle = @"none";
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
    params.lambda = 3.0;  // final config A (walk2/frozen_final.json); v1 was 1.5
    params.tau = 0.9;
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

- (void)_loadOnLoadQueue
{
    if (_loadedFlag.load() || _failedFlag.load()) {
        _loadingFlag.store(false);
        return;
    }
    auto t0 = std::chrono::steady_clock::now();
    auto engine = std::make_unique<McBopomofoSlothE::Engine>();
    std::string error;
    std::string dir(_resourcePath.fileSystemRepresentation);
    // Size + sha256 of every encoder input against runtime-manifest.txt first:
    // a damaged or swapped file never reaches the loader. Failure = stock.
    bool ok = McBopomofoSlothE::VerifyRuntimeFiles(dir, McBopomofoSlothE::EncoderRuntimeFiles(), &error);
    _verifyMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ok = ok && engine->load(dir, &error);
    if (ok) {
        // Warm up the compute threads once; the result stays in the cache.
        double ms = 0;
        bool hit = false;
        engine->forward(std::vector<std::string> { "ㄅㄚ" }, &ms, &hit);
    }
    double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (ok) {
        _loadMs = elapsed;
        _engine = std::move(engine);
        _loadedFlag.store(true, std::memory_order_release);
        [self _appendLogLine:[NSString stringWithFormat:@"# %@ SlothE-T 12M loaded in %.1f ms (files verified in %.1f ms)\n", [NSDate date], elapsed, _verifyMs]];
        // Pre-build the per-length compute graphs for typical buffer lengths so
        // the first keystrokes do not pay for graph construction. Runs after
        // `loaded` is set; the engine serializes these with real passes.
        _engine->setGraphCacheCapacity(kGraphCacheCapacity);
        auto warm0 = std::chrono::steady_clock::now();
        std::vector<std::string> warm;
        for (int t = 2; t <= kWarmGraphLengths; ++t) {
            warm.assign(static_cast<size_t>(t), "ㄅㄚ");
            double ms = 0;
            bool hit = false;
            _engine->forward(warm, &ms, &hit);
        }
        int graphs = 0;
        size_t bytes = 0;
        uint64_t builds = 0;
        _engine->graphCacheStats(&graphs, &bytes, &builds);
        double warmMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - warm0).count();
        [self _appendLogLine:[NSString stringWithFormat:@"# %@ warmed %d compute graphs (%.1f MB) in %.1f ms\n", [NSDate date], graphs, static_cast<double>(bytes) / 1048576.0, warmMs]];
        [self _loadDecoderOnLoadQueue];
    } else {
        _loadErrorString = @(error.c_str());
        _failedFlag.store(true, std::memory_order_release);
        NSLog(@"McBopomofoLM: SlothE-T load failed: %@", _loadErrorString);
        [self _appendLogLine:[NSString stringWithFormat:@"# %@ SlothE-T load failed; stock McBopomofo behaviour\n", [NSDate date]]];
    }
    _loadingFlag.store(false, std::memory_order_release);
}

// The SlothE decoder (phase 3), after the encoder; until it is loaded the
// key handler runs the encoder-only (phase 2) path.
- (void)_loadDecoderOnLoadQueue
{
    NSString *path = [_resourcePath stringByAppendingPathComponent:@(McBopomofoSlothE::kDecoderModelFileName)];
    auto t0 = std::chrono::steady_clock::now();
    std::string error;
    // Same integrity check as the encoder; any failure leaves the encoder-only
    // in-walk path running (decoder missing/failed -> encoder-only by design).
    bool ok = McBopomofoSlothE::VerifyRuntimeFiles(std::string(_resourcePath.fileSystemRepresentation),
        std::vector<std::string> { McBopomofoSlothE::kDecoderModelFileName }, &error);
    double verifyMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    auto decoder = std::make_unique<McBopomofoSlothE::Decoder>();
    ok = ok && decoder->load(std::string(path.fileSystemRepresentation), 4, 128, 4, &error);
    if (ok) {
        std::vector<double> scores;
        McBopomofoSlothE::DecoderCallStats stats;
        ok = decoder->score("", std::vector<std::string> { "今天", "天氣" }, &scores, &stats);  // warm-up + probe
        for (double v : scores) {
            ok = ok && std::isfinite(v) && v < 0.0;
        }
        if (!ok) {
            error = "decoder probe gave invalid scores";
        }
        decoder->reset();
    }
    double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (ok) {
        _decoderLoadMs = elapsed;
        _decoder = std::move(decoder);
        _decoderLoadedFlag.store(true, std::memory_order_release);
        [self _appendLogLine:[NSString stringWithFormat:@"# %@ SlothE decoder loaded in %.1f ms (file verified in %.1f ms)\n", [NSDate date], elapsed, verifyMs]];
    } else {
        _decoderLoadErrorString = @(error.c_str());
        _decoderFailedFlag.store(true, std::memory_order_release);
        NSLog(@"McBopomofoLM: SlothE decoder load failed: %s", error.c_str());
        [self _appendLogLine:[NSString stringWithFormat:@"# %@ SlothE decoder load failed; encoder-only in-walk stays on\n", [NSDate date]]];
    }
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
