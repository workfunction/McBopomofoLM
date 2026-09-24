// SlothE-T runtime for the McBopomofoLM side-by-side test build: owns the
// model (loaded on a background queue) and the serial compute queue the key
// handler submits forward passes to, keeps event counters, and appends
// per-event timing to ~/Library/Application Support/McBopomofoLM/latency.log
// (lengths, timings and outcomes only, never typed text).

#import <Foundation/Foundation.h>

#ifdef __cplusplus
#include "SlothEDecoder.h"
#include "SlothEEngine.h"
#include "SlothEPipeline.h"
#include "SlothEWalk.h"
#endif

NS_ASSUME_NONNULL_BEGIN

/// Posted (main thread) when the SlothE-T or decoder switch is toggled; every
/// key handler drops model-derived state and re-walks at once.
FOUNDATION_EXPORT NSNotificationName const SlothEPreferencesDidChangeNotification NS_SWIFT_NAME(slothEPreferencesDidChange);

@interface SlothERuntime : NSObject

/// The process-wide runtime (bundle resources, real latency log). Under
/// XCTest it has no log path and is never loaded automatically.
@property (class, readonly, nonatomic) SlothERuntime *sharedRuntime NS_SWIFT_NAME(shared);
@property (class, readonly, nonatomic) BOOL runningUnderXCTest;
/// ~/Library/Application Support/McBopomofoLM/latency.log
@property (class, readonly, nonatomic) NSString *defaultLatencyLogPath;

- (instancetype)initWithResourcePath:(NSString *)resourcePath logPath:(nullable NSString *)logPath NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;
/// Tests: a runtime with its own queues, counters and log that uses the
/// already-loaded models of `other` (one ANE load per test process).
- (instancetype)initSharingModelsOf:(SlothERuntime *)other logPath:(nullable NSString *)logPath;
/// Tests: the bundle's models, loaded once per process (no log).
@property (class, readonly, nonatomic) SlothERuntime *sharedLoadedRuntimeForTesting;

@property (readonly, nonatomic) NSString *resourcePath;
@property (readonly, nonatomic, nullable) NSString *logPath;
/// YES once the model finished loading successfully.
@property (readonly, nonatomic) BOOL loaded;
@property (readonly, nonatomic) BOOL loading;
@property (readonly, nonatomic) BOOL loadFailed;
@property (readonly, nonatomic, nullable) NSString *loadError;
@property (readonly, nonatomic) double loadMilliseconds;
/// Decoder (phase 3): loaded on the same background queue after the encoder.
@property (readonly, nonatomic) BOOL decoderLoaded;
@property (readonly, nonatomic) BOOL decoderLoadFailed;
@property (readonly, nonatomic, nullable) NSString *decoderLoadError;
@property (readonly, nonatomic) double decoderLoadMilliseconds;

/// Serial queue for SlothE-T forward passes (never the main queue).
@property (readonly, nonatomic) dispatch_queue_t computeQueue;
/// Test hook: extra sleep before every asynchronous forward pass.
@property (assign, atomic) double debugComputeDelayMilliseconds;
/// How long a commit (or opening the candidate window) waits for a pending
/// in-walk result before using the stock walk. Default 30 ms.
@property (assign, nonatomic) double commitWaitMilliseconds;

/// Coalescing submit to the compute queue: at most ONE job waits at any
/// time; a newer job replaces the waiting one, which is released without
/// running (with the grid snapshot it holds). The job that is already
/// running is not affected.
- (void)submitComputeJob:(dispatch_block_t)job;
@property (readonly) NSUInteger pendingComputeJobs;     // 0 or 1
@property (readonly) NSUInteger peakPendingComputeJobs; // highest pendingComputeJobs seen
@property (readonly) NSUInteger coalescedComputeJobs;   // waiting jobs replaced before they ran
@property (readonly) NSUInteger startedComputeJobs;
@property (readonly) NSUInteger submittedComputeJobs;
- (void)resetComputeJobStatistics;
/// Grid snapshots (decoder jobs' copies of the buffer) alive right now.
+ (int64_t)liveGridSnapshots;
/// This process's phys_footprint (task_vm_info), bytes.
+ (uint64_t)physFootprintBytes;

/// Prewarm (ANE wake-up): called on the first key of a new buffer. When the
/// models have not run for more than prewarmIdleSeconds (default 1 s), one
/// dummy prediction per model runs on the compute queue. Returns YES if fired.
- (BOOL)prewarmIfIdle;
@property (assign, nonatomic) double prewarmIdleSeconds;
@property (readonly, nonatomic) NSUInteger prewarmCount;      // fired (main thread)
@property (readonly, nonatomic) NSUInteger prewarmsCompleted; // finished on the compute queue

/// v2.1 lazy decoder functions: t16 loads at start-up; t32 / t64 / t96 load on
/// the background load queue the first time a request needs them (a request
/// for a function not loaded yet gets no decoder decision). Each lazily loaded
/// function gets its own placement check; one that fails stays unloaded and is
/// not retried in this process. `install` loads all of them.
@property (readonly, nonatomic) NSArray<NSNumber *> *decoderLoadedLengths;
@property (readonly, nonatomic) NSUInteger decoderUnavailableRequests;
@property (readonly, nonatomic) NSUInteger decoderLazyLoadsRequested;
@property (readonly, nonatomic) NSUInteger decoderLazyLoadsDone;
@property (readonly, nonatomic) NSUInteger decoderLazyLoadsFailed;
- (NSDictionary<NSString *, NSNumber *> *)lazyLoadSummaryForLength:(NSInteger)length;
/// Placement gate: minimum share (percent) of each function's estimated cost
/// that MLComputePlan puts on the Neural Engine. Encoder 99 (measures 100),
/// decoder 80 (measures 85.6-94.9; the rest is the token-id gather). Below it
/// the model is not used (reason not_on_ane; the R line has each function's %).
@property (assign, nonatomic) double minEncoderANECostPercent;
@property (assign, nonatomic) double minDecoderANECostPercent;
/// Test hook: lazily loaded decoder functions use Core ML CPU_ONLY (so their
/// placement check fails).
@property (assign, nonatomic) BOOL lazyCPUOnlyForTesting;

/// Why a model is not used ("ok" when it is): integrity, missing, load_error,
/// not_on_ane, probe_slow, plan_failed, macos, vocab, probe_failed, no_encoder.
@property (readonly, nonatomic, nullable) NSString *loadReason;
@property (readonly, nonatomic, nullable) NSString *decoderLoadReason;
/// Load / placement numbers of the last load (ms, ANE cost %, probe ms).
@property (readonly, nonatomic) NSDictionary<NSString *, NSNumber *> *placementSummary;
/// Test hook: load with Core ML CPU_ONLY instead of CPU_AND_NE; the placement
/// check must then reject both models ("not on the ANE" = no model).
@property (assign, nonatomic) BOOL computeUnitsCPUOnlyForTesting;
/// Test hook: the same for the decoder only (the encoder stays on the ANE).
@property (assign, nonatomic) BOOL decoderCPUOnlyForTesting;
/// `McBopomofoLM install`: load both models synchronously from their final
/// in-bundle paths (this compiles them for the ANE under this executable),
/// verify placement, and report progress lines. YES when both are on the ANE.
/// It first clears this app's Neural Engine compile cache (aneCachePath).
- (BOOL)loadForInstallWithProgress:(nullable void (^)(NSString *line))progress;
/// ~/Library/Caches/<bundle id>/com.apple.e5rt.e5bundlecache (Core ML's ANE
/// compile cache for this app). Cleared at install, and once per process if
/// MLComputePlan fails or puts < 1% of a model on the ANE (a damaged or stale
/// cache); the load is then retried.
@property (class, readonly, nonatomic, nullable) NSString *aneCachePath;
/// Marker file next to it: a lazily loaded decoder function failed with a
/// failed plan or < 1% on the ANE; the next start-up clears the cache once.
@property (class, readonly, nonatomic, nullable) NSString *aneCacheClearMarkerPath;
/// Test hook: report a lazy decoder load failure as if Core ML had returned it.
- (void)noteLazyLoadFailureForTesting:(NSInteger)length reason:(NSString *)reason anePercent:(double)anePercent;
/// Test hook: install without clearing the cache (keeps the test suite fast).
@property (assign, nonatomic) BOOL keepANECacheForTesting;
/// Test hook: the first encoder placement check reports this reason
/// ("plan_failed", or "not_on_ane" with 0% on the ANE = a collapsed placement).
@property (copy, nonatomic, nullable) NSString *simulatePlacementFailureOnceForTesting;
@property (copy, nonatomic, nullable) void (^installProgress)(NSString *line);

/// Loads the model on a background queue. Idempotent.
- (void)startLoading;
/// Loads the model on the calling thread (tests). Returns loaded.
- (BOOL)loadSynchronously;

/// Event counters (main thread).
@property (readonly, nonatomic) NSUInteger appliedCount;     // async in-walk result refreshed the buffer
@property (readonly, nonatomic) NSUInteger unchangedCount;   // async result arrived, same text as shown
@property (readonly, nonatomic) NSUInteger staleCount;       // result for an outdated buffer, dropped
@property (readonly, nonatomic) NSUInteger skippedCount;     // queued pass skipped, buffer already changed
@property (readonly, nonatomic) NSUInteger deferredCount;    // arrived while not in the plain composing state
@property (readonly, nonatomic) NSUInteger consumedCount;    // already applied by a commit / settle
@property (readonly, nonatomic) NSUInteger commitWaitCount;  // commit/candidate-window waited and got the result
@property (readonly, nonatomic) NSUInteger commitTimeoutCount; // waited, timed out, used the stock walk
@property (readonly, nonatomic) NSUInteger commitPartialCount; // waited, got the encoder stage only
@property (readonly, nonatomic) NSUInteger decoderRunCount;    // passes whose decoder stage scored nodes
@property (readonly, nonatomic) NSUInteger decoderPinCount;    // decoder decisions applied
@property (readonly, nonatomic) NSUInteger decoderStaleCount;  // decoder stage abandoned: buffer changed
/// End-to-end milliseconds (key event -> final result applied on the main
/// thread) of the most recent passes, oldest first (at most 4096 kept).
@property (readonly, nonatomic) NSArray<NSNumber *> *recentEndToEndMilliseconds;

/// Key event bookkeeping (main thread). The key handler brackets each key
/// event with begin/end and notes what happened in between.
- (void)beginKeystroke;
- (void)noteStockWalkMilliseconds:(double)milliseconds;
- (void)noteInWalkMilliseconds:(double)milliseconds;
- (void)noteProvisionalMilliseconds:(double)milliseconds;
- (void)noteForwardQueued;
- (void)noteForwardFromCache;
- (void)noteRerankMilliseconds:(double)milliseconds;
/// kind: "ready" (no wait), "waited", "partial" (encoder only), "timeout".
- (void)noteSettle:(NSString *)kind milliseconds:(double)milliseconds;
/// state: "on" / "off" / "loading" / "failed" / "plain".
- (void)endKeystrokeWithBufferLength:(NSUInteger)bufferLength handlerMilliseconds:(double)milliseconds state:(NSString *)state;
/// One asynchronous pass outcome: "applied", "unchanged", "stale",
/// "skipped", "deferred", "consumed". decoder: "off", "none", "scored",
/// "stale", "failed". endToEndMs < 0: not measured.
- (void)noteAsyncResult:(NSString *)result bufferLength:(NSUInteger)bufferLength queueMilliseconds:(double)queueMilliseconds forwardMilliseconds:(double)forwardMilliseconds decoderMilliseconds:(double)decoderMilliseconds decoderCalls:(NSInteger)decoderCalls pins:(NSInteger)pins decoder:(NSString *)decoder inWalkMilliseconds:(double)inWalkMilliseconds endToEndMilliseconds:(double)endToEndMilliseconds;
/// Blocks until pending log writes are on disk (tests).
- (void)flushLog;

#ifdef __cplusplus
/// Non-null only after a successful load.
- (nullable McBopomofoSlothE::Engine *)engine;
/// In-walk configuration, config A' (walk2/frozen_final2.json, 25M): beta 0.3, gamma 0,
/// illegal-char penalty -15, variant guard on.
- (McBopomofoSlothE::InWalkParams)inWalkParams;
/// Non-null only after the decoder loaded.
- (nullable McBopomofoSlothE::Decoder *)decoder;
/// Decoder gate, config A' (walk2/frozen_final2.json, 25M): lambda 2, tau 0.5, top-3.
- (McBopomofoSlothE::DecoderParams)decoderParams;
- (McBopomofoSlothE::DecoderScoreCache *)decoderScoreCache;
#endif

@end

NS_ASSUME_NONNULL_END
