// McBopomofoLM v3.4 test support: the 18 "<prefix>再" twins that Source/Data/zai-twins.txt adds to data.txt (v3.4a),
// and a language model without them. The fixtures of the earlier versions (walk2 parity, v2_parity.jsonl, the W7 and
// pause-gate fixtures) were made on data.txt before the twins; a twin only changes the segmentation of a few
// sentences (開 + 會再 instead of 開會 + 再), so the tests that compare node by node load the data without them.

#ifndef MCBOPOMOFOTESTS_SLOTHEV34TESTSUPPORT_H_
#define MCBOPOMOFOTESTS_SLOTHEV34TESTSUPPORT_H_

#import <Foundation/Foundation.h>

#include <memory>
#include <set>
#include <string>

#include "McBopomofoLM.h"

namespace V34 {

// (reading, twin value), the twins of Source/Data/zai-twins.txt; the "在" entry is the twin with its last char replaced.
inline NSArray<NSArray<NSString *> *> *ZaiTwins()
{
    return @[
        @[ @"ㄅㄨˊ-ㄏㄨㄟˋ-ㄗㄞˋ", @"不會再" ], @[ @"ㄊㄚ-ㄗㄞˋ", @"他再" ], @[ @"ㄊㄚ-ㄗㄞˋ", @"她再" ], @[ @"ㄋㄧˇ-ㄗㄞˋ", @"你再" ],
        @[ @"ㄏㄞˊ-ㄋㄥˊ-ㄗㄞˋ", @"還能再" ], @[ @"ㄏㄞˊ-ㄏㄨㄟˋ-ㄗㄞˋ", @"還會再" ], @[ @"ㄏㄨㄟˋ-ㄗㄞˋ", @"會再" ],
        @[ @"ㄐㄧㄡˋ-ㄗㄞˋ", @"就再" ], @[ @"ㄐㄧㄤ-ㄗㄞˋ", @"將再" ], @[ @"ㄒㄧㄢ-ㄗㄞˋ", @"先再" ], @[ @"ㄓˇ-ㄋㄥˊ-ㄗㄞˋ", @"只能再" ],
        @[ @"ㄓˇ-ㄏㄠˇ-ㄗㄞˋ", @"只好再" ], @[ @"ㄓˇ-ㄏㄨㄟˋ-ㄗㄞˋ", @"只會再" ], @[ @"ㄗㄨㄟˋ-ㄏㄠˇ-ㄗㄞˋ", @"最好再" ],
        @[ @"ㄘㄞˊ-ㄏㄨㄟˋ-ㄗㄞˋ", @"才會再" ], @[ @"ㄧㄝˇ-ㄏㄨㄟˋ-ㄗㄞˋ", @"也會再" ], @[ @"ㄧㄝˇ-ㄒㄩˇ-ㄗㄞˋ", @"也許再" ],
        @[ @"ㄧㄥ-ㄍㄞ-ㄗㄞˋ", @"應該再" ],
    ];
}

// The bundled data.txt without the twins, loaded as a language model (written once to the temporary directory).
inline std::shared_ptr<McBopomofo::McBopomofoLM> LoadDataWithoutTwins()
{
    static std::shared_ptr<McBopomofo::McBopomofoLM> lm;
    if (lm != nullptr) {
        return lm;
    }
    NSString *source = [NSBundle.mainBundle pathForResource:@"data" ofType:@"txt"];
    NSString *text = [NSString stringWithContentsOfFile:source encoding:NSUTF8StringEncoding error:nil];
    std::set<std::string> drop;
    for (NSArray<NSString *> *twin in ZaiTwins()) {
        drop.insert(std::string(twin[0].UTF8String) + " " + twin[1].UTF8String + " ");
    }
    NSMutableString *kept = [NSMutableString stringWithCapacity:text.length];
    for (NSString *line in [text componentsSeparatedByString:@"\n"]) {
        BOOL isTwin = NO;
        if ([line containsString:@"再 "]) {
            for (const std::string &prefix : drop) {
                if ([line hasPrefix:@(prefix.c_str())]) {
                    isTwin = YES;
                    break;
                }
            }
        }
        if (!isTwin && line.length > 0) {
            [kept appendFormat:@"%@\n", line];
        }
    }
    NSString *path = [NSTemporaryDirectory() stringByAppendingPathComponent:@"mcbpmf-data-without-zai-twins.txt"];
    [kept writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:nil];
    lm = std::make_shared<McBopomofo::McBopomofoLM>();
    lm->loadLanguageModel(path.fileSystemRepresentation);
    return lm;
}

}  // namespace V34

#endif  // MCBOPOMOFOTESTS_SLOTHEV34TESTSUPPORT_H_
