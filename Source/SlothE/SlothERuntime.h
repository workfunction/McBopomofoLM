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
/// Frozen in-walk configuration (walk2/frozen.json, 12M): beta 0.3, gamma 0,
/// illegal-char penalty -15, variant guard on.
- (McBopomofoSlothE::InWalkParams)inWalkParams;
/// Non-null only after the decoder loaded.
- (nullable McBopomofoSlothE::Decoder *)decoder;
/// Decoder gate, final config A (walk2/frozen_final.json, 12M): lambda 3, tau 0.9, top-3.
- (McBopomofoSlothE::DecoderParams)decoderParams;
- (McBopomofoSlothE::DecoderScoreCache *)decoderScoreCache;
#endif

@end

NS_ASSUME_NONNULL_END
