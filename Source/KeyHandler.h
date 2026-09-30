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

#import <Foundation/Foundation.h>

@class KeyHandlerInput;
@class InputState;
@class SlothERuntime;

NS_ASSUME_NONNULL_BEGIN

typedef NSString *const InputMode NS_TYPED_ENUM;
extern InputMode InputModeBopomofo;
extern InputMode InputModePlainBopomofo;

@class KeyHandler;

@protocol KeyHandlerDelegate <NSObject>
- (id)candidateControllerForKeyHandler:(KeyHandler *)keyHandler;
- (void)keyHandler:(KeyHandler *)keyHandler didSelectCandidateAtIndex:(NSInteger)index candidateController:(id)controller;
- (BOOL)keyHandler:(KeyHandler *)keyHandler didRequestWriteUserPhraseWithState:(InputState *)state;
- (BOOL)keyHandler:(KeyHandler *)keyHandler didRequestBoostScoreForPhrase:(NSString *)phrase reading:(NSString *)reading;
- (BOOL)keyHandler:(KeyHandler *)keyHandler didRequestExcludePhrase:(NSString *)phrase reading:(NSString *)reading;
- (BOOL)keyHandlerDidRequestReloadLanguageModel:(KeyHandler *)keyHandler;
@optional
/// McBopomofoLM: an in-walk rescoring result arrived for the current buffer.
/// Return YES only when the plain composing state is showing, so the buffer
/// can be replaced without disturbing a candidate window or another state.
- (BOOL)keyHandlerCanRefreshComposingBuffer:(KeyHandler *)keyHandler;
/// McBopomofoLM: show this refreshed composing state (same buffer, rescored).
- (void)keyHandler:(KeyHandler *)keyHandler didRefreshComposingBufferWithState:(InputState *)state;
@end

@interface BuildAssociatedPhraseParams: NSObject
@property (strong, nonatomic) id previousState;
@property (assign, nonatomic) size_t prefixCursorIndex;
@property (strong, nonatomic) NSString *reading;
@property (strong, nonatomic) NSString *value;
@property (assign, nonatomic) NSInteger candidateIndex;
@property (assign, nonatomic) BOOL useVerticalMode;
@property (assign, nonatomic) BOOL autoTriggered;
@end

@interface KeyHandler : NSObject

- (BOOL)handleInput:(KeyHandlerInput *)input
              state:(InputState *)state
      stateCallback:(void (^)(InputState *))stateCallback
      errorCallback:(void (^)(void))errorCallback NS_SWIFT_NAME(handle(input:state:stateCallback:errorCallback:));
- (BOOL)handleAssociatedPhraseWithState:(InputState *)state
                        useVerticalMode:(BOOL)useVerticalMode
                          stateCallback:(void (^)(InputState *))stateCallback
                          errorCallback:(void (^)(void))errorCallback
                            autoTriggered:(BOOL)useShiftKey
                      maxCandidateCount:(size_t)maxCandidateCount;

- (void)syncWithPreferences;
- (void)fixNodeWithReading:(NSString *)reading
                                 value:(NSString *)value
                   originalCursorIndex:(size_t)originalCursorIndex
    useMoveCursorAfterSelectionSetting:(BOOL)flag NS_SWIFT_NAME(fixNode(reading:value:originalCursorIndex:useMoveCursorAfterSelectionSetting:));
- (void)fixNodeForAssociatedPhraseWithPrefixAt:(size_t)prefixCursorIndex
                                 prefixReading:(NSString *)pfxReading
                                   prefixValue:(NSString *)pfxValue
                       associatedPhraseReading:(NSString *)phraseReading
                         associatedPhraseValue:(NSString *)phraseValue;
- (void)clear;

- (void)handleForceCommitWithStateCallback:(void (^)(InputState *))stateCallback
    NS_SWIFT_NAME(handleForceCommit(stateCallback:));

- (InputState *)buildInputtingState;

- (nullable InputState *)buildAssociatedPhrasePlainStateWithReading:(NSString *)reading
                                                              value:(NSString *)value
                                                    useVerticalMode:(BOOL)useVerticalMode;
- (nullable InputState *)buildAssociatedPhraseStateWithParams:(BuildAssociatedPhraseParams *)params;

- (size_t)computeActualCursorIndex:(size_t)cursor;

- (NSArray<NSString *> *)collectUserFileIssues;

@property (strong, nonatomic) InputMode inputMode;
@property (weak, nonatomic) id<KeyHandlerDelegate> delegate;
@property (assign, nonatomic, readonly) NSInteger actualCandidateCursorIndex;
@property (assign, nonatomic, readonly) NSInteger cursorIndex;
/// McBopomofoLM: SlothE-T runtime used for candidate reranking and key timing.
/// Defaults to SlothERuntime.sharedRuntime; nil disables SlothE-T entirely.
@property (strong, nonatomic, nullable) SlothERuntime *slothERuntime;

/// McBopomofoLM v3.1a carried context (reports/w7.md Ce1): the chars committed
/// before the current buffer feed the encoder and decoder of the next buffer.
/// Memory only; never persisted or logged. Called by the input controller:
/// after every insertText: text = before any Simplified conversion (the
/// model context), insertedLength = UTF-16 length of what the client got ...
- (void)slothENoteCommittedText:(NSString *)text insertedLength:(NSInteger)insertedLength;
/// ... on activate / deactivate / an app-requested commitComposition ...
- (void)slothEResetCarriedContext:(NSString *)reason;
/// ... and before each key in the Empty state. selection returns the client's
/// selectedRange().location; it is called at most once, only for a key that
/// can start a buffer (the handler's own empty-state conditions), and never
/// again for a client whose answer took > 20 ms or was NSNotFound / bogus
/// (a client without a bundle id is never turned off: only that buffer falls
/// back). Clears the context when idle > 120 s, when the
/// selection is not where the last buffer's text ended, or (no selection)
/// when the last commit is > 15 s old.
- (void)slothEPrepareKey:(KeyHandlerInput *)input inEmptyStateWithSelection:(NSInteger (^)(void))selection;
/// YES once the current client is not asked for its selection any more.
@property (assign, nonatomic, readonly) BOOL slothESelectionOffForClient;
+ (void)slothEResetSelectionOffClientsForTesting;
/// Tests: carried chars, and the most recent reset that cleared a non-empty context.
@property (assign, nonatomic, readonly) NSUInteger slothECarriedLength;
@property (nonatomic, readonly, nullable) NSString *slothELastResetReason;
/// Seconds on a monotonic clock (tests inject their own).
@property (copy, nonatomic) NSTimeInterval (^slothEClock)(void);
/// v3.1b correction window pick (the controller, for InputStateChoosingTypoCorrection):
/// the illegal reading becomes `reading`, inserted into the grid with `value` pinned.
- (void)slothEApplyTypoCorrectionWithReading:(NSString *)reading value:(NSString *)value rank:(NSInteger)rank;
/// Tests: wait up to `milliseconds` for the correction scores of the current
/// illegal reading; YES when they are ready.
- (BOOL)slothEWaitForTypoScores:(double)milliseconds;
/// How long Space / Down waits for the scores before showing the T1 list (50 ms).
@property (assign, nonatomic) double slothETypoWaitMilliseconds;
/// Tests: correction-score jobs started (one per reading + context).
@property (assign, nonatomic, readonly) NSUInteger slothETypoJobsStarted;
/// v3.2a typing-pause gate (PLAN.md §18): this long after the last key the decoder
/// gate runs over the whole buffer (150 ms). Tests set it, or inject a scheduler.
@property (assign, nonatomic) double slothEPauseDelayMilliseconds;
/// Tests: called instead of dispatch_after (main queue) to arm the pause timer:
/// (delay in seconds, block to run when it fires). The block does nothing if a
/// key event came after it was armed.
@property (copy, nonatomic, nullable) void (^slothEPauseScheduler)(NSTimeInterval delay, dispatch_block_t fire);
/// v3.2b legal-but-wrong correction entry (the controller, for a candidate with
/// isLegalTypoEntry): deletes the reading at the syllable the window is on,
/// inserts `reading` there, puts the cursor back to originalCursorIndex and
/// re-walks. It is not a pick of a word: nothing goes to the user override model.
- (void)slothEApplyLegalTypoCorrectionWithReading:(NSString *)reading slot:(NSInteger)slot originalCursorIndex:(NSInteger)originalCursorIndex;
/// Tests: correction-entry jobs started (one per syllable + context), and the
/// entry of the finished job for the syllable at the cursor ((reading, char),
/// nil when there is none, or it is not finished / not for this buffer).
@property (assign, nonatomic, readonly) NSUInteger slothELegalJobsStarted;
- (nullable NSArray<NSString *> *)slothELegalEntryForTesting;
/// Tests: install a finished job that offers (reading, value) for the syllable at the
/// cursor, for this buffer and context, so the window placement and the pick can be
/// tested without the models' opinion.
- (void)slothEInstallLegalEntryForTestingWithReading:(NSString *)reading value:(NSString *)value;
/// Client bundle identifier for the latency log's C lines (set on activate).
@property (copy, nonatomic, nullable) NSString *slothEClientID;
@end

NS_ASSUME_NONNULL_END
