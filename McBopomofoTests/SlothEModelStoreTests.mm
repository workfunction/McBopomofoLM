// McBopomofoLM v3.4p: the model store (SlothEEngine.cpp: EnsureModelStore, CollectModelStoreGarbage, the manifest-path
// form of VerifyRuntimeFiles). No Core ML and no real model: a small fake bundle (a runtime manifest, a few files, a
// nested ".mlmodelc"-like folder) in a temp folder, and temp store roots. The runtime's use of the store (loads from it,
// falls back, cleans up after its load) is in V34pTests.swift.

#import <XCTest/XCTest.h>
#import <CommonCrypto/CommonCrypto.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/xattr.h>
#include <unistd.h>

#include <atomic>
#include <string>
#include <vector>

#include "SlothEEngine.h"

using namespace McBopomofoSlothE;

static NSString *TempFolder(NSString *label)
{
    NSString *path = [NSTemporaryDirectory() stringByAppendingPathComponent:[NSString stringWithFormat:@"v34p-%@-%@", label, NSUUID.UUID.UUIDString]];
    [NSFileManager.defaultManager createDirectoryAtPath:path withIntermediateDirectories:YES attributes:nil error:nil];
    return path;
}

static NSString *Sha256Hex(NSData *data)
{
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    NSMutableString *hex = [NSMutableString string];
    for (unsigned char b : digest) {
        [hex appendFormat:@"%02x", b];
    }
    return hex;
}

/// A fake bundle `SlothE` folder: the files below and a runtime-manifest.txt listing all of them.
static NSDictionary<NSString *, NSData *> *FakeFiles(NSString *tag)
{
    NSMutableData *weights = [NSMutableData dataWithLength:300000];
    memset(weights.mutableBytes, tag.hash & 0xFF, weights.length);
    return @{
        @"syl_vocab.tsv" : [[@"a\t1\n" stringByAppendingString:tag] dataUsingEncoding:NSUTF8StringEncoding],
        @"enc.bin" : [NSData dataWithBytes:"\x01\x02\x03\x04" length:4],
        @"m.mlmodelc/coremldata.bin" : [@"coreml" dataUsingEncoding:NSUTF8StringEncoding],
        @"m.mlmodelc/analytics/a.bin" : [@"analytics" dataUsingEncoding:NSUTF8StringEncoding],
        @"m.mlmodelc/weights/weight.bin" : weights,
    };
}

static NSString *MakeFakeBundle(NSString *tag, NSDictionary<NSString *, NSString *> *extraManifestLines = nil)
{
    NSString *bundle = TempFolder(@"bundle");
    NSMutableString *manifest = [NSMutableString stringWithString:@"# fake runtime manifest\n"];
    NSDictionary<NSString *, NSData *> *files = FakeFiles(tag);
    for (NSString *name in [files.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
        NSString *path = [bundle stringByAppendingPathComponent:name];
        [NSFileManager.defaultManager createDirectoryAtPath:path.stringByDeletingLastPathComponent withIntermediateDirectories:YES attributes:nil error:nil];
        [files[name] writeToFile:path atomically:YES];
        [manifest appendFormat:@"%@ %lu %@\n", Sha256Hex(files[name]), (unsigned long)files[name].length, name];
    }
    for (NSString *line in extraManifestLines.allValues) {
        [manifest appendFormat:@"%@\n", line];
    }
    [manifest writeToFile:[bundle stringByAppendingPathComponent:@"runtime-manifest.txt"] atomically:YES encoding:NSUTF8StringEncoding error:nil];
    return bundle;
}

static std::vector<std::string> NamesOf(NSString *tag)
{
    std::vector<std::string> names;
    for (NSString *name in FakeFiles(tag).allKeys) {
        names.push_back(name.UTF8String);
    }
    return names;
}

static NSString *Std(const std::string &s) { return @(s.c_str()); }

struct Identity {
    ino_t inode = 0;
    struct timespec mtime = {};
    off_t size = 0;
};

static Identity IdentityOf(NSString *path)
{
    struct stat st;
    Identity id;
    if (lstat(path.fileSystemRepresentation, &st) == 0) {
        id.inode = st.st_ino;
        id.mtime = st.st_mtimespec;
        id.size = st.st_size;
    }
    return id;
}

static BOOL Same(const Identity &a, const Identity &b)
{
    return a.inode == b.inode && a.mtime.tv_sec == b.mtime.tv_sec && a.mtime.tv_nsec == b.mtime.tv_nsec && a.size == b.size;
}

static NSArray<NSString *> *Sorted(NSArray<NSString *> *names)
{
    return [names sortedArrayUsingSelector:@selector(compare:)];
}

static NSArray<NSString *> *Listing(NSString *folder)
{
    return [[NSFileManager.defaultManager contentsOfDirectoryAtPath:folder error:nil] sortedArrayUsingSelector:@selector(compare:)];
}

/// Every entry below `folder` with its kind and size, without following links: what must not change when something
/// outside a store is not to be touched.
static NSArray<NSString *> *Tree(NSString *folder)
{
    NSMutableArray<NSString *> *out = [NSMutableArray array];
    NSFileManager *fm = NSFileManager.defaultManager;
    for (NSString *relative in [fm enumeratorAtPath:folder]) {
        NSDictionary *attributes = [fm attributesOfItemAtPath:[folder stringByAppendingPathComponent:relative] error:nil];
        [out addObject:[NSString stringWithFormat:@"%@|%@|%@", relative, attributes[NSFileType], attributes[NSFileSize]]];
    }
    return [out sortedArrayUsingSelector:@selector(compare:)];
}

static void Link(NSString *from, NSString *to)
{
    [NSFileManager.defaultManager createDirectoryAtPath:from.stringByDeletingLastPathComponent withIntermediateDirectories:YES attributes:nil error:nil];
    symlink(to.fileSystemRepresentation, from.fileSystemRepresentation);
}

static BOOL IsLink(NSString *path)
{
    struct stat st;
    return lstat(path.fileSystemRepresentation, &st) == 0 && S_ISLNK(st.st_mode);
}

@interface SlothEModelStoreTests : XCTestCase
@end

@implementation SlothEModelStoreTests {
    NSMutableArray<NSString *> *_cleanup;
}

- (void)setUp
{
    _cleanup = [NSMutableArray array];
}

- (void)tearDown
{
    for (NSString *path in _cleanup) {
        [NSFileManager.defaultManager removeItemAtPath:path error:nil];
    }
}

- (NSString *)bundle:(NSString *)tag
{
    NSString *bundle = MakeFakeBundle(tag);
    [_cleanup addObject:bundle];
    return bundle;
}

- (NSString *)root
{
    NSString *root = [TempFolder(@"root") stringByAppendingPathComponent:@"Application Support/McBopomofoLM.test/Models"];  // intermediate folders do not exist yet
    [_cleanup addObject:root.stringByDeletingLastPathComponent.stringByDeletingLastPathComponent.stringByDeletingLastPathComponent];
    return root;
}

- (void)testSetIDIsTheFirst16HexCharactersOfTheManifestsSha256
{
    NSString *bundle = [self bundle:@"one"];
    std::string setId;
    std::string error;
    XCTAssertTrue(ModelSetId(bundle.fileSystemRepresentation, &setId, &error), @"%s", error.c_str());
    NSData *manifest = [NSData dataWithContentsOfFile:[bundle stringByAppendingPathComponent:@"runtime-manifest.txt"]];
    XCTAssertEqualObjects(Std(setId), [Sha256Hex(manifest) substringToIndex:16]);
    XCTAssertEqual(setId.size(), 16u);
    // other models, another manifest, another set id
    std::string other;
    XCTAssertTrue(ModelSetId([self bundle:@"two"].fileSystemRepresentation, &other, &error));
    XCTAssertTrue(setId != other);
    XCTAssertFalse(ModelSetId("/nonexistent", &other, &error));
}

- (void)testCreatesAStoreWithEveryManifestFileAndNothingElse
{
    NSString *bundle = [self bundle:@"create"];
    NSString *root = [self root];
    ModelStoreResult result = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertTrue(result.ok, @"%s", result.error.c_str());
    XCTAssertTrue(result.created);
    XCTAssertEqualObjects(Std(result.directory), [root stringByAppendingPathComponent:Std(result.setId)], @"<root>/<set id>");
    NSDictionary<NSString *, NSData *> *files = FakeFiles(@"create");
    for (NSString *name in files) {
        NSData *copy = [NSData dataWithContentsOfFile:[Std(result.directory) stringByAppendingPathComponent:name]];
        XCTAssertEqualObjects(copy, files[name], @"%@", name);
    }
    XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[Std(result.directory) stringByAppendingPathComponent:@"runtime-manifest.txt"]],
                   @"the store is checked against the bundle's manifest, it carries none of its own");
    XCTAssertEqualObjects(Listing(root), Sorted(@[ @".lock", @".leases", Std(result.setId) ]), @"no .tmp-* left, the lock file is there");
    // a real copy, not a hard link to the bundle's file
    struct stat copy, original;
    XCTAssertEqual(stat([Std(result.directory) stringByAppendingPathComponent:@"enc.bin"].fileSystemRepresentation, &copy), 0);
    XCTAssertEqual(stat([bundle stringByAppendingPathComponent:@"enc.bin"].fileSystemRepresentation, &original), 0);
    XCTAssertNotEqual(copy.st_ino, original.st_ino);
    XCTAssertEqual(copy.st_nlink, 1);
    // and it verifies against the bundle's manifest
    std::string error;
    XCTAssertTrue(VerifyRuntimeFiles(result.directory, NamesOf(@"create"), &error, std::string(bundle.fileSystemRepresentation) + "/runtime-manifest.txt"), @"%s", error.c_str());
}

- (void)testAnExistingIntactStoreIsReusedAndItsIdentityIsUntouched
{
    NSString *bundle = [self bundle:@"reuse"];
    NSString *root = [self root];
    ModelStoreResult first = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertTrue(first.ok && first.created);
    NSString *dir = Std(first.directory);
    std::vector<std::pair<NSString *, Identity>> before;
    for (NSString *name in FakeFiles(@"reuse")) {
        before.push_back({ name, IdentityOf([dir stringByAppendingPathComponent:name]) });
    }
    Identity dirBefore = IdentityOf(dir);
    usleep(20000);
    ModelStoreResult second = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertTrue(second.ok, @"%s", second.error.c_str());
    XCTAssertFalse(second.created, @"reused, not copied again");
    XCTAssertTrue(first.directory == second.directory);
    for (auto &entry : before) {
        XCTAssertTrue(Same(entry.second, IdentityOf([dir stringByAppendingPathComponent:entry.first])), @"%@: inode, mtime and size unchanged", entry.first);
    }
    XCTAssertTrue(Same(dirBefore, IdentityOf(dir)), @"the set directory itself is untouched");
    XCTAssertEqualObjects(Listing(root), Sorted(@[ @".lock", @".leases", Std(first.setId) ]));
}

- (void)testADamagedStoreIsReplaced
{
    NSString *bundle = [self bundle:@"damage"];
    NSString *root = [self root];
    ModelStoreResult first = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    NSString *dir = Std(first.directory);
    NSString *victim = [dir stringByAppendingPathComponent:@"m.mlmodelc/weights/weight.bin"];
    NSDictionary<NSString *, NSData *> *files = FakeFiles(@"damage");

    // one byte flipped, same size: the sha256 catches it
    NSMutableData *data = [NSMutableData dataWithContentsOfFile:victim];
    ((unsigned char *)data.mutableBytes)[1000] ^= 0x01;
    [data writeToFile:victim atomically:NO];
    first.lease.reset();  // nobody reads the set: a repair is allowed (a held lease would stop it, see the lease tests)
    ModelStoreResult flipped = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertTrue(flipped.ok, @"%s", flipped.error.c_str());
    XCTAssertTrue(flipped.created, @"replaced");
    XCTAssertEqualObjects([NSData dataWithContentsOfFile:victim], files[@"m.mlmodelc/weights/weight.bin"]);

    // a missing file
    flipped.lease.reset();
    [NSFileManager.defaultManager removeItemAtPath:[dir stringByAppendingPathComponent:@"enc.bin"] error:nil];
    ModelStoreResult missing = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertTrue(missing.ok && missing.created);
    XCTAssertEqualObjects([NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"enc.bin"]], files[@"enc.bin"]);

    // the set directory replaced by a plain file
    missing.lease.reset();
    [NSFileManager.defaultManager removeItemAtPath:dir error:nil];
    [@"not a directory" writeToFile:dir atomically:YES encoding:NSUTF8StringEncoding error:nil];
    ModelStoreResult file = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertTrue(file.ok && file.created, @"%s", file.error.c_str());
    BOOL isDirectory = NO;
    XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:dir isDirectory:&isDirectory] && isDirectory);
    XCTAssertEqualObjects(Listing(root), Sorted(@[ @".lock", @".leases", Std(first.setId) ]), @"no .corrupt-* or .tmp-* left behind");
}

- (void)testAStoreCarryingItsOwnRegeneratedManifestIsStillRejected
{
    NSString *bundle = [self bundle:@"forge"];
    NSString *root = [self root];
    ModelStoreResult first = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    NSString *dir = Std(first.directory);
    // somebody replaces a model file AND regenerates a manifest inside the store to match
    NSString *victim = [dir stringByAppendingPathComponent:@"enc.bin"];
    NSData *forged = [NSData dataWithBytes:"EVIL!!" length:6];
    [forged writeToFile:victim atomically:NO];
    NSString *storeManifest = [NSString stringWithFormat:@"%@ %lu enc.bin\n", Sha256Hex(forged), (unsigned long)forged.length];
    [storeManifest writeToFile:[dir stringByAppendingPathComponent:@"runtime-manifest.txt"] atomically:NO encoding:NSUTF8StringEncoding error:nil];

    std::string error;
    std::string bundleManifest = std::string(bundle.fileSystemRepresentation) + "/runtime-manifest.txt";
    XCTAssertTrue(VerifyRuntimeFiles(first.directory, { "enc.bin" }, &error), @"read against its OWN manifest the forgery would pass (%s)", error.c_str());
    XCTAssertFalse(VerifyRuntimeFiles(first.directory, { "enc.bin" }, &error, bundleManifest), @"against the bundle's it does not");
    XCTAssertNotEqual(error.find("enc.bin"), std::string::npos);

    first.lease.reset();
    ModelStoreResult again = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertTrue(again.ok && again.created, @"rejected and replaced: %s", again.error.c_str());
    XCTAssertEqualObjects([NSData dataWithContentsOfFile:victim], FakeFiles(@"forge")[@"enc.bin"]);
}

- (void)testTwoConcurrentCallersCopyOnce
{
    for (int round = 0; round < 3; ++round) {
        NSString *bundle = [self bundle:[NSString stringWithFormat:@"race%d", round]];
        NSString *root = [self root];
        std::atomic<int> copies { 0 };
        std::atomic<int> created { 0 };
        std::atomic<int> failed { 0 };
        std::atomic<int> *copiesPointer = &copies;
        std::atomic<int> *createdPointer = &created;
        std::atomic<int> *failedPointer = &failed;
        ModelStoreOptions options;
        options.beforePublish = [copiesPointer](const std::string &) { copiesPointer->fetch_add(1); usleep(60000); };
        dispatch_apply(6, dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^(size_t) {
            ModelStoreResult r = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation, options);
            if (!r.ok) {
                failedPointer->fetch_add(1);
            } else if (r.created) {
                createdPointer->fetch_add(1);
            }
        });
        XCTAssertEqual(failed.load(), 0, @"round %d", round);
        XCTAssertEqual(copies.load(), 1, @"round %d: only the caller that got the lock first copies", round);
        XCTAssertEqual(created.load(), 1, @"round %d: one created, five reused", round);
        XCTAssertEqual(Listing(root).count, 3u, @"round %d: .lock, .leases and one set", round);
    }
}

- (void)testAWinnerPublishedMeanwhileIsDetectedAndNeverClobbered
{
    NSString *bundle = [self bundle:@"winner"];
    NSString *root = [self root];
    std::string setId;
    std::string error;
    XCTAssertTrue(ModelSetId(bundle.fileSystemRepresentation, &setId, &error));
    NSString *final = [root stringByAppendingPathComponent:Std(setId)];
    ModelStoreOptions options;
    options.beforePublish = [&](const std::string &) {
        // another process (one that does not take the lock) publishes an intact copy first, with a marker file in it
        for (NSString *name in FakeFiles(@"winner")) {
            NSString *path = [final stringByAppendingPathComponent:name];
            [NSFileManager.defaultManager createDirectoryAtPath:path.stringByDeletingLastPathComponent withIntermediateDirectories:YES attributes:nil error:nil];
            [FakeFiles(@"winner")[name] writeToFile:path atomically:YES];
        }
        [@"winner" writeToFile:[final stringByAppendingPathComponent:@"WINNER"] atomically:YES encoding:NSUTF8StringEncoding error:nil];
    };
    ModelStoreResult result = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation, options);
    XCTAssertTrue(result.ok, @"%s", result.error.c_str());
    XCTAssertFalse(result.created, @"the winner's copy is used");
    XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:[final stringByAppendingPathComponent:@"WINNER"]], @"not clobbered");
    XCTAssertEqualObjects(Listing(root), Sorted(@[ @".lock", @".leases", Std(setId) ]), @"our temporary folder is gone");

    // a winner that is not intact is an error, and still not clobbered
    NSString *bundle2 = [self bundle:@"winner2"];
    NSString *root2 = [self root];
    std::string setId2;
    XCTAssertTrue(ModelSetId(bundle2.fileSystemRepresentation, &setId2, &error));
    NSString *final2 = [root2 stringByAppendingPathComponent:Std(setId2)];
    ModelStoreOptions broken;
    broken.beforePublish = [&](const std::string &) {
        [NSFileManager.defaultManager createDirectoryAtPath:final2 withIntermediateDirectories:YES attributes:nil error:nil];
        [@"x" writeToFile:[final2 stringByAppendingPathComponent:@"WINNER"] atomically:YES encoding:NSUTF8StringEncoding error:nil];
    };
    ModelStoreResult bad = EnsureModelStore(bundle2.fileSystemRepresentation, root2.fileSystemRepresentation, broken);
    XCTAssertFalse(bad.ok);
    XCTAssertFalse(bad.error.empty());
    XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:[final2 stringByAppendingPathComponent:@"WINNER"]]);
    XCTAssertEqualObjects(Listing(root2), Sorted(@[ @".lock", @".leases", Std(setId2) ]));
}

- (void)testEnsureNeverDeletesAnotherSetOrALeftoverTemporaryFolder
{
    // `install` calls only this: the clean-up is the runtime's, after its own successful load
    NSString *bundle = [self bundle:@"keep"];
    NSString *root = [self root];
    for (NSString *name in @[ @"aaaaaaaaaaaaaaaa", @".tmp-1-deadbeef", @".corrupt-2-cafe" ]) {
        [NSFileManager.defaultManager createDirectoryAtPath:[root stringByAppendingPathComponent:name] withIntermediateDirectories:YES attributes:nil error:nil];
    }
    ModelStoreResult result = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertTrue(result.ok && result.created);
    XCTAssertEqualObjects(Listing(root), Sorted(@[ @".corrupt-2-cafe", @".lock", @".leases", @".tmp-1-deadbeef", Std(result.setId), @"aaaaaaaaaaaaaaaa" ]));
}

- (void)testCollectGarbageKeepsTheCurrentSetTheLockAndWhatIsNotOurs
{
    NSString *bundle = [self bundle:@"gc"];
    NSString *root = [self root];
    ModelStoreResult current = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertTrue(current.ok);
    for (NSString *name in @[ @"aaaaaaaaaaaaaaaa", @"0123456789abcdef", @".tmp-1-deadbeef", @".corrupt-2-cafe" ]) {
        NSString *dir = [root stringByAppendingPathComponent:name];
        [NSFileManager.defaultManager createDirectoryAtPath:[dir stringByAppendingPathComponent:@"deep"] withIntermediateDirectories:YES attributes:nil error:nil];
        [@"x" writeToFile:[dir stringByAppendingPathComponent:@"deep/f"] atomically:YES encoding:NSUTF8StringEncoding error:nil];
    }
    // not ours: a file, a folder whose name is not a set id
    [@"mine" writeToFile:[root stringByAppendingPathComponent:@"notes.txt"] atomically:YES encoding:NSUTF8StringEncoding error:nil];
    [NSFileManager.defaultManager createDirectoryAtPath:[root stringByAppendingPathComponent:@"my-folder"] withIntermediateDirectories:YES attributes:nil error:nil];
    Identity before = IdentityOf(Std(current.directory));

    ModelStoreCleanup cleanup = CollectModelStoreGarbage(root.fileSystemRepresentation, current.setId, bundle.fileSystemRepresentation);
    XCTAssertEqual(cleanup.sets, 2u);
    XCTAssertEqual(cleanup.temporaries, 2u);
    XCTAssertEqual(cleanup.protectedSets, 1u, @"the current set");
    XCTAssertEqualObjects(Listing(root), Sorted(@[ @".lock", @".leases", @"my-folder", @"notes.txt", Std(current.setId) ]));
    XCTAssertTrue(Same(before, IdentityOf(Std(current.directory))), @"the current set is untouched");
    ModelStoreCleanup again = CollectModelStoreGarbage(root.fileSystemRepresentation, current.setId, bundle.fileSystemRepresentation);
    XCTAssertEqual(again.sets + again.temporaries, 0u);
    // a root that does not exist is nothing to clean
    ModelStoreCleanup none = CollectModelStoreGarbage("/nonexistent/v34p", current.setId, bundle.fileSystemRepresentation);
    XCTAssertEqual(none.sets + none.temporaries, 0u);
}

- (void)testAnUnsafeManifestNameIsRefusedBeforeAnythingIsWritten
{
    for (NSString *bad in @[ @"../evil.bin", @"/abs/evil.bin", @"a/../../evil.bin", @"a//b.bin" ]) {
        NSString *bundle = MakeFakeBundle(@"unsafe", @{ @"x" : [NSString stringWithFormat:@"%@ 1 %@", Sha256Hex([NSData data]), bad] });
        [_cleanup addObject:bundle];
        NSString *root = [self root];
        ModelStoreResult result = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
        XCTAssertFalse(result.ok, @"%@", bad);
        XCTAssertNotEqual(result.error.find("unsafe"), std::string::npos, @"%s", result.error.c_str());
        XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:root], @"%@: no root made", bad);
    }
}

- (void)testItFailsCleanlyWhenTheRootCannotBeMadeOrTheBundleIsDamaged
{
    NSString *bundle = [self bundle:@"fail"];
    NSString *folder = TempFolder(@"blocked");
    [_cleanup addObject:folder];
    NSString *file = [folder stringByAppendingPathComponent:@"plain-file"];
    [@"x" writeToFile:file atomically:YES encoding:NSUTF8StringEncoding error:nil];
    ModelStoreResult blocked = EnsureModelStore(bundle.fileSystemRepresentation, [file stringByAppendingPathComponent:@"Models"].fileSystemRepresentation);
    XCTAssertFalse(blocked.ok);
    XCTAssertFalse(blocked.error.empty());

    // a bundle file that no longer matches the bundle's manifest: the copy fails its check, nothing is published
    NSString *root = [self root];
    NSMutableData *data = [NSMutableData dataWithContentsOfFile:[bundle stringByAppendingPathComponent:@"enc.bin"]];
    ((unsigned char *)data.mutableBytes)[0] ^= 0x01;
    [data writeToFile:[bundle stringByAppendingPathComponent:@"enc.bin"] atomically:NO];
    ModelStoreResult damaged = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertFalse(damaged.ok);
    XCTAssertNotEqual(damaged.error.find("enc.bin"), std::string::npos, @"%s", damaged.error.c_str());
    XCTAssertEqualObjects(Listing(root), Sorted(@[ @".lock", @".leases" ]), @"no set, no temporary folder");

    ModelStoreResult missing = EnsureModelStore("/nonexistent", root.fileSystemRepresentation);
    XCTAssertFalse(missing.ok);
}

// MARK: - Sol r49: links, leases, snapshots

/// A folder outside any store that a redirected store would reach: it holds what a clean-up deletes (a set-named folder,
/// leftovers), and a file.
- (NSString *)outsideFolderIn:(NSString *)holder
{
    NSString *outside = [holder stringByAppendingPathComponent:@"outside"];
    for (NSString *name in @[ @".tmp-1-aaaa", @"aaaaaaaaaaaaaaaa", @".corrupt-2-bbbb", @"bbbbbbbbbbbbbbbb" ]) {
        NSString *dir = [outside stringByAppendingPathComponent:name];
        [NSFileManager.defaultManager createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
        [@"precious" writeToFile:[dir stringByAppendingPathComponent:@"file"] atomically:YES encoding:NSUTF8StringEncoding error:nil];
    }
    [@"sentinel" writeToFile:[outside stringByAppendingPathComponent:@"sentinel.txt"] atomically:YES encoding:NSUTF8StringEncoding error:nil];
    return outside;
}

// User decision 2026-10-14: a symlinked store folder is allowed (the user points `Models` somewhere else for testing);
// r49 HIGH accepted as by design. What stays careful is removal.
- (void)testASymlinkedStoreFolderIsFollowedOnPurpose
{
    NSString *bundle = [self bundle:@"linked-models"];
    NSString *holder = TempFolder(@"linked");
    [_cleanup addObject:holder];
    NSString *target = [holder stringByAppendingPathComponent:@"elsewhere/Models"];
    [NSFileManager.defaultManager createDirectoryAtPath:target withIntermediateDirectories:YES attributes:nil error:nil];
    NSString *root = [holder stringByAppendingPathComponent:@"App/McBopomofoLM.test/Models"];
    Link(root, target);  // Models -> elsewhere/Models
    ModelStoreResult result = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertTrue(result.ok && result.created, @"%s", result.error.c_str());
    XCTAssertTrue(IsLink(root), @"the link is still a link");
    XCTAssertEqualObjects(Listing(target), Sorted(@[ @".lock", @".leases", Std(result.setId) ]), @"the store was made where the link points");
    XCTAssertTrue(CollectModelStoreGarbage(root.fileSystemRepresentation, result.setId, bundle.fileSystemRepresentation).sets == 0);
    result.lease.reset();
    ModelStoreResult again = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertTrue(again.ok && !again.created, @"reused through the link");
}

- (void)testADamagedSymlinkedSetIsReplacedByARealCopyAndItsTargetIsNotTouched
{
    NSString *bundle = [self bundle:@"linked-set"];
    NSString *root = [self root];
    ModelStoreResult first = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    NSString *setDirectory = Std(first.directory);
    first.lease.reset();
    // the set directory becomes a link to a (damaged) folder somewhere else
    NSString *elsewhere = [self outsideFolderIn:root.stringByDeletingLastPathComponent.stringByDeletingLastPathComponent];
    [NSFileManager.defaultManager removeItemAtPath:setDirectory error:nil];
    Link(setDirectory, elsewhere);
    NSArray<NSString *> *before = Tree(elsewhere);
    ModelStoreResult repaired = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertTrue(repaired.ok && repaired.created, @"%s", repaired.error.c_str());
    XCTAssertFalse(IsLink(setDirectory), @"the link was replaced by a real directory");
    XCTAssertEqualObjects(Tree(elsewhere), before, @"what the link pointed to was neither entered nor deleted");
}

- (void)testNewFilesAreClonedOnApfsAndDistinctFromTheBundlesFiles
{
    NSString *bundle = [self bundle:@"clone"];
    NSString *root = [self root];
    struct statfs fs;
    XCTAssertEqual(statfs(NSTemporaryDirectory().fileSystemRepresentation, &fs), 0);
    if (strcmp(fs.f_fstypename, "apfs") != 0) {
        XCTSkip(@"clones need APFS");
        return;
    }
    // a download mark on a bundle file: a clone would carry it along
    setxattr([bundle stringByAppendingPathComponent:@"enc.bin"].fileSystemRepresentation, "com.apple.quarantine", "0081;00000000;Test;", 19, 0, 0);
    ModelStoreResult result = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertTrue(result.ok && result.created, @"%s", result.error.c_str());
    XCTAssertEqual(result.clonedFiles, NamesOf(@"clone").size(), @"every file cloned (one volume)");
    XCTAssertEqual(result.copiedFiles, 0u);
    for (NSString *name in FakeFiles(@"clone")) {
        struct stat copy, original;
        XCTAssertEqual(stat([Std(result.directory) stringByAppendingPathComponent:name].fileSystemRepresentation, &copy), 0);
        XCTAssertEqual(stat([bundle stringByAppendingPathComponent:name].fileSystemRepresentation, &original), 0);
        XCTAssertNotEqual(copy.st_ino, original.st_ino, @"%@: a clone is a new file with its own inode", name);
        XCTAssertEqual(copy.st_nlink, 1);
        XCTAssertEqualObjects([NSData dataWithContentsOfFile:[Std(result.directory) stringByAppendingPathComponent:name]], FakeFiles(@"clone")[name]);
    }
    XCTAssertEqual(getxattr([Std(result.directory) stringByAppendingPathComponent:@"enc.bin"].fileSystemRepresentation, "com.apple.quarantine", nullptr, 0, 0, 0), -1, @"no download mark on the store's file");
    // and the clone is verified against the bundle's manifest as before
    std::string error;
    XCTAssertTrue(VerifyRuntimeFiles(result.directory, NamesOf(@"clone"), &error, std::string(bundle.fileSystemRepresentation) + "/runtime-manifest.txt"), @"%s", error.c_str());
}

- (void)testWhenCloningIsNotPossibleTheFilesAreCopied
{
    NSString *bundle = [self bundle:@"noclone"];
    NSString *root = [self root];
    ModelStoreOptions options;
    options.disableCloneForTesting = true;  // as on another volume or a file system without clones
    ModelStoreResult result = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation, options);
    XCTAssertTrue(result.ok && result.created, @"%s", result.error.c_str());
    XCTAssertEqual(result.clonedFiles, 0u);
    XCTAssertEqual(result.copiedFiles, NamesOf(@"noclone").size());
    for (NSString *name in FakeFiles(@"noclone")) {
        XCTAssertEqualObjects([NSData dataWithContentsOfFile:[Std(result.directory) stringByAppendingPathComponent:name]], FakeFiles(@"noclone")[name]);
    }
}

- (void)testCleanUpAndRepairNeverFollowALinkInsideTheStore
{
    NSString *bundle = [self bundle:@"links-inside"];
    NSString *root = [self root];
    NSString *holder = root.stringByDeletingLastPathComponent.stringByDeletingLastPathComponent;
    NSString *outside = [self outsideFolderIn:holder];
    NSArray<NSString *> *before = Tree(outside);
    ModelStoreResult current = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertTrue(current.ok);
    // an old set that has a link to the outside inside it, links named like a leftover and like a set
    NSString *old = [root stringByAppendingPathComponent:@"bbbbbbbbbbbbbbbb"];
    [NSFileManager.defaultManager createDirectoryAtPath:old withIntermediateDirectories:YES attributes:nil error:nil];
    Link([old stringByAppendingPathComponent:@"evil"], outside);
    Link([root stringByAppendingPathComponent:@".tmp-9-link"], outside);
    Link([root stringByAppendingPathComponent:@"eeeeeeeeeeeeeeee"], outside);

    ModelStoreCleanup cleanup = CollectModelStoreGarbage(root.fileSystemRepresentation, current.setId, bundle.fileSystemRepresentation);
    XCTAssertEqual(cleanup.sets, 1u, @"the old set went, with its link (unlinked, not entered)");
    XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:old]);
    XCTAssertTrue(IsLink([root stringByAppendingPathComponent:@".tmp-9-link"]), @"a link is not a leftover folder");
    XCTAssertTrue(IsLink([root stringByAppendingPathComponent:@"eeeeeeeeeeeeeeee"]));
    XCTAssertEqualObjects(Tree(outside), before);

    // repair of a damaged set that contains a link
    NSString *dir = Std(current.directory);
    current.lease.reset();
    Link([dir stringByAppendingPathComponent:@"m.mlmodelc/weights/sneaky"], outside);
    NSString *victim = [dir stringByAppendingPathComponent:@"enc.bin"];
    [[NSData dataWithBytes:"XXXX" length:4] writeToFile:victim atomically:NO];
    ModelStoreResult repaired = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertTrue(repaired.ok && repaired.created, @"%s", repaired.error.c_str());
    XCTAssertFalse(IsLink([dir stringByAppendingPathComponent:@"m.mlmodelc/weights/sneaky"]));
    XCTAssertEqualObjects(Tree(outside), before, @"the link in the damaged set was unlinked, its target not entered");
}

- (void)testASetAnotherRuntimeStillHoldsALeaseOnIsNotCollected
{
    NSString *bundleA = [self bundle:@"lease-a"];
    NSString *bundleB = [self bundle:@"lease-b"];
    NSString *root = [self root];
    ModelStoreResult a = EnsureModelStore(bundleA.fileSystemRepresentation, root.fileSystemRepresentation);  // runtime A loads S_A
    ModelStoreResult b = EnsureModelStore(bundleB.fileSystemRepresentation, root.fileSystemRepresentation);  // runtime B loads S_B
    XCTAssertTrue(a.ok && b.ok && a.lease && b.lease);
    NSString *setA = Std(a.directory);
    XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:[root stringByAppendingPathComponent:@".leases"]]);

    // B's clean-up (B's bundle is the current one): A's set is leased, so it stays
    ModelStoreCleanup first = CollectModelStoreGarbage(root.fileSystemRepresentation, b.setId, bundleB.fileSystemRepresentation);
    XCTAssertEqual(first.sets, 0u);
    XCTAssertEqual(first.leased, 1u);
    XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:setA], @"A is still reading it");
    // A goes away: the next clean-up deletes the set and its lease file
    a.lease.reset();
    ModelStoreCleanup second = CollectModelStoreGarbage(root.fileSystemRepresentation, b.setId, bundleB.fileSystemRepresentation);
    XCTAssertEqual(second.sets, 1u);
    XCTAssertEqual(second.leased, 0u);
    XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:setA]);
    XCTAssertEqualObjects(Listing([root stringByAppendingPathComponent:@".leases"]), (@[ [Std(b.setId) stringByAppendingString:@".lock"] ]), @"A's lease file went with its set");
}

- (void)testAnOldRuntimesCleanUpNeverDeletesTheSetTheBundleOnDiskNamesNow
{
    NSString *bundleOld = [self bundle:@"gc-old"];
    NSString *bundleNew = [self bundle:@"gc-new"];
    NSString *root = [self root];
    ModelStoreResult old = EnsureModelStore(bundleOld.fileSystemRepresentation, root.fileSystemRepresentation);
    ModelStoreResult installed = EnsureModelStore(bundleNew.fileSystemRepresentation, root.fileSystemRepresentation);
    NSString *oldSet = Std(old.directory);
    NSString *newSet = Std(installed.directory);
    installed.lease.reset();  // `install` has exited: the new set has no reader yet
    // the old runtime (started on gc-old) finishes loading after the update: the bundle path now holds gc-new's files
    ModelStoreCleanup cleanup = CollectModelStoreGarbage(root.fileSystemRepresentation, old.setId, bundleNew.fileSystemRepresentation);
    XCTAssertEqual(cleanup.sets, 0u);
    XCTAssertEqual(cleanup.protectedSets, 2u, @"its own set and the installed one");
    XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:newSet], @"the installed set survives");
    XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:oldSet]);

    // a bundle whose manifest cannot be read (an update in the middle of replacing the app): no set is deleted
    NSString *stray = [root stringByAppendingPathComponent:@"cccccccccccccccc"];
    [NSFileManager.defaultManager createDirectoryAtPath:stray withIntermediateDirectories:YES attributes:nil error:nil];
    ModelStoreCleanup unknown = CollectModelStoreGarbage(root.fileSystemRepresentation, old.setId, "/nonexistent/app");
    XCTAssertEqual(unknown.sets, 0u);
    XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:stray]);
    // while with a readable manifest the stray, unleased, unnamed set does go
    ModelStoreCleanup known = CollectModelStoreGarbage(root.fileSystemRepresentation, old.setId, bundleNew.fileSystemRepresentation);
    XCTAssertEqual(known.sets, 1u);
    XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:stray]);
}

- (void)testADamagedSetSomebodyStillHoldsALeaseOnIsNotRepaired
{
    NSString *bundle = [self bundle:@"lease-repair"];
    NSString *root = [self root];
    ModelStoreResult reader = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);  // a runtime reading the set
    XCTAssertTrue(reader.ok);
    NSString *dir = Std(reader.directory);
    NSString *victim = [dir stringByAppendingPathComponent:@"enc.bin"];
    [[NSData dataWithBytes:"XXXX" length:4] writeToFile:victim atomically:NO];  // damaged underneath it

    ModelStoreResult other = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertFalse(other.ok, @"busy: not repaired");
    XCTAssertNotEqual(other.error.find("in use"), std::string::npos, @"%s", other.error.c_str());
    XCTAssertFalse(other.lease);
    XCTAssertEqualObjects([NSData dataWithContentsOfFile:victim], [NSData dataWithBytes:"XXXX" length:4], @"the files under the reader were not touched");
    XCTAssertEqualObjects(Listing(root), Sorted(@[ @".lock", @".leases", Std(reader.setId) ]), @"no .corrupt or .tmp either");

    reader.lease.reset();  // the reader is gone
    ModelStoreResult repaired = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertTrue(repaired.ok && repaired.created, @"%s", repaired.error.c_str());
    XCTAssertEqualObjects([NSData dataWithContentsOfFile:victim], FakeFiles(@"lease-repair")[@"enc.bin"]);
}

- (void)testALeaseFileOfASetThatIsGoneIsSwept
{
    NSString *bundle = [self bundle:@"sweep"];
    NSString *root = [self root];
    ModelStoreResult current = EnsureModelStore(bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    NSString *leases = [root stringByAppendingPathComponent:@".leases"];
    [@"" writeToFile:[leases stringByAppendingPathComponent:@"ffffffffffffffff.lock"] atomically:YES encoding:NSUTF8StringEncoding error:nil];
    CollectModelStoreGarbage(root.fileSystemRepresentation, current.setId, bundle.fileSystemRepresentation);
    XCTAssertEqualObjects(Listing(leases), (@[ [Std(current.setId) stringByAppendingString:@".lock"] ]));
}

- (void)testASnapshotOfTheManifestOutlivesTheFileBeingReplaced
{
    NSString *bundle = [self bundle:@"snapshot"];
    NSString *root = [self root];
    ModelManifest snapshot;
    std::string error;
    XCTAssertTrue(LoadModelManifest(bundle.fileSystemRepresentation, &snapshot, &error), @"%s", error.c_str());
    ModelStoreResult store = EnsureModelStore(snapshot, bundle.fileSystemRepresentation, root.fileSystemRepresentation);
    XCTAssertTrue(store.ok && store.setId == snapshot.setId);
    // an update replaces the bundle's manifest (new models: another sha for one of the files)
    NSString *manifestPath = [bundle stringByAppendingPathComponent:@"runtime-manifest.txt"];
    NSString *text = [NSString stringWithContentsOfFile:manifestPath encoding:NSUTF8StringEncoding error:nil];
    NSString *changed = [text stringByReplacingOccurrencesOfString:Sha256Hex(FakeFiles(@"snapshot")[@"enc.bin"]) withString:Sha256Hex([NSData dataWithBytes:"new" length:3])];
    XCTAssertNotEqualObjects(text, changed);
    [changed writeToFile:manifestPath atomically:YES encoding:NSUTF8StringEncoding error:nil];

    std::string manifestFile = std::string(manifestPath.fileSystemRepresentation);
    XCTAssertTrue(VerifyRuntimeFiles(store.directory, NamesOf(@"snapshot"), &error, snapshot), @"the snapshot still verifies the set this runtime loaded: %s", error.c_str());
    XCTAssertFalse(VerifyRuntimeFiles(store.directory, NamesOf(@"snapshot"), &error, manifestFile), @"the replaced file would fail it");
    XCTAssertNotEqual(error.find("enc.bin"), std::string::npos);
}

@end
