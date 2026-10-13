import os
import tempfile
import unittest
from curation.compilers.postprocess import *

SAMPLE_DATA = """ㄋㄧㄢˊ 年 -6.0865
ㄋㄧㄢˊ 粘 -11.2857
ㄋㄧㄢˊ 黏 -11.3369
ㄋㄧㄢˊ-ㄓㄨㄥ 年終 -11.3730
ㄋㄧㄢˊ-ㄓㄨㄥ 年中 -11.6689
ㄏㄠˇ 好 -2.78076369
ㄏㄠˇ-ㄒㄧㄢˇ 好險 -6.36398807
ㄏㄡ-ㄒㄧㄢˇ 好險 -6.36398807
ㄒㄧㄢˇ 顯 -3.57899621
ㄒㄧㄢˇ 險 -3.78145638
ㄓㄨㄥ 中 -5.8093
ㄓㄨㄥ 鍾 -9.6857
ㄓㄨㄥ 鐘 -9.8776
ㄓㄨㄥ 忠 -99.0000
ㄓㄨㄥ 盅 -99.0000
ㄓㄨㄥ 終 -99.0000
"""


class TestPostprocess(unittest.TestCase):
    def setUp(self):

        self.lm = {}

        for line in SAMPLE_DATA.splitlines():

            reading, value, score = line.split()
            if reading in self.lm:
                self.lm[reading].append((value, float(score)))
            else:
                self.lm[reading] = [(value, float(score))]

    def test_find_top_unigram_in_lm(self):
        value, score = find_top_unigram_in_lm(self.lm, "ㄋㄧㄢˊ")
        self.assertEqual(value, "年")
        self.assertLess(score, 0)

        value, score = find_top_unigram_in_lm(self.lm, "ㄋㄧㄢˊ-ㄓㄨㄥ")
        self.assertEqual(value, "年終")
        self.assertLess(score, 0)

        self.assertIsNone(find_top_unigram_in_lm(self.lm, "ㄅㄨˋ"))

    def test_find_top_unigram_in_lm_failure(self):
        self.assertIsNone(find_top_unigram_in_lm(self.lm, "ㄇㄟˊ"))

    def test_find_score_in_lm(self):
        s1 = find_score_in_lm(self.lm, "ㄋㄧㄢˊ-ㄓㄨㄥ", "年終")
        s2 = find_score_in_lm(self.lm, "ㄋㄧㄢˊ-ㄓㄨㄥ", "年中")
        self.assertLess(s2, s1)

    def test_find_score_in_lm_failure_not_found(self):
        self.assertIsNone(find_score_in_lm(self.lm, "ㄋㄧㄢˊ-ㄉㄧˇ", "年底"))
        self.assertIsNone(find_score_in_lm(self.lm, "ㄋㄧㄢˊ-ㄓㄨㄥ", "年鐘"))
        self.assertIsNone(find_score_in_lm(self.lm, "ㄋㄧㄢˊ-ㄓㄨㄥˋ", "年中"))

    def test_find_score_in_lm_failure_mismatching_reading_count(self):
        self.assertIsNone(find_score_in_lm(self.lm, "ㄋㄧㄢˊ-ㄓㄨㄥ", "年"))
        self.assertIsNone(find_score_in_lm(self.lm, "ㄋㄧㄢˊ", "年終"))

    def test_replace_score_in_lm(self):
        s = find_score_in_lm(self.lm, "ㄋㄧㄢˊ-ㄓㄨㄥ", "年中")
        self.assertNotEqual(s, -2.0)

        before = list(self.lm["ㄋㄧㄢˊ-ㄓㄨㄥ"])

        replace_score_in_lm(self.lm, "ㄋㄧㄢˊ-ㄓㄨㄥ", "年中", -2.0)

        after = list(self.lm["ㄋㄧㄢˊ-ㄓㄨㄥ"])
        self.assertNotEqual(before, after)

        s = find_score_in_lm(self.lm, "ㄋㄧㄢˊ-ㄓㄨㄥ", "年中")
        self.assertEqual(s, -2.0)

    def test_replace_score_in_lm_falure_no_reading(self):
        with self.assertRaises(ValueError):
            replace_score_in_lm(self.lm, "ㄋㄧˊ-ㄓㄨㄥ", "年中", -2.0)

    def test_replace_score_in_lm_falure_no_replacement(self):
        with self.assertRaises(ValueError):
            replace_score_in_lm(self.lm, "ㄋㄧㄢˊ-ㄓㄨㄥ", "年鐘", -2.0)

    def test_promote_over_single_syllables(self):
        v1, s1 = find_top_unigram_in_lm(self.lm, "ㄏㄠˇ")
        v2, s2 = find_top_unigram_in_lm(self.lm, "ㄒㄧㄢˇ")

        self.assertEqual(v1, "好")
        self.assertEqual(v2, "顯")

        before = find_score_in_lm(self.lm, "ㄏㄠˇ-ㄒㄧㄢˇ", "好險")
        self.assertLess(before, s1 + s2)

        promote_over_single_syllables(0, self.lm, "好險", "ㄏㄠˇ-ㄒㄧㄢˇ")
        after = find_score_in_lm(self.lm, "ㄏㄠˇ-ㄒㄧㄢˇ", "好險")
        self.assertGreater(after, s1 + s2)

    def test_promote_over_single_syllables_failures(self):
        # mismatching reading counts
        self.assertIsNone(
            promote_over_single_syllables(0, self.lm, "好", "ㄏㄠˇ-ㄒㄧㄢˇ")
        )
        self.assertIsNone(promote_over_single_syllables(0, self.lm, "好險", "ㄏㄠˇ"))

        # No such reading exists
        self.assertIsNone(
            promote_over_single_syllables(0, self.lm, "好險", "ㄏㄠˋ-ㄒㄧㄢˇ")
        )

        # Some unigrams not found
        self.assertIsNone(
            promote_over_single_syllables(0, self.lm, "好險", "ㄏㄡ-ㄒㄧㄢˇ")
        )

        # Promoting second time is an error
        v1, s1 = find_top_unigram_in_lm(self.lm, "ㄏㄠˇ")
        v2, s2 = find_top_unigram_in_lm(self.lm, "ㄒㄧㄢˇ")
        before = find_score_in_lm(self.lm, "ㄏㄠˇ-ㄒㄧㄢˇ", "好險")
        self.assertTrue(
            promote_over_single_syllables(0, self.lm, "好險", "ㄏㄠˇ-ㄒㄧㄢˇ")
        )
        find_score_in_lm(self.lm, "ㄏㄠˇ-ㄒㄧㄢˇ", "好險")
        self.assertIsNone(
            promote_over_single_syllables(0, self.lm, "好險", "ㄏㄠˇ-ㄒㄧㄢˇ")
        )

    def test_promote_over_peers(self):
        s1 = find_score_in_lm(self.lm, "ㄋㄧㄢˊ-ㄓㄨㄥ", "年終")
        s2 = find_score_in_lm(self.lm, "ㄋㄧㄢˊ-ㄓㄨㄥ", "年中")
        self.assertLess(s2, s1)

        promote_over_peers(0, self.lm, "年中", "ㄋㄧㄢˊ-ㄓㄨㄥ")
        s1 = find_score_in_lm(self.lm, "ㄋㄧㄢˊ-ㄓㄨㄥ", "年終")
        s2 = find_score_in_lm(self.lm, "ㄋㄧㄢˊ-ㄓㄨㄥ", "年中")
        self.assertGreater(s2, s1)

    def test_promote_over_peers_failures(self):
        # Not found
        self.assertIsNone(find_score_in_lm(self.lm, "ㄋㄧˊ-ㄓㄨㄥ", "年終"))

        # Promoting an already top unigram is an error
        self.assertIsNone(promote_over_peers(0, self.lm, "年終", "ㄋㄧㄢˊ-ㄓㄨㄥ"))


TWIN_DATA = """ㄗㄞˋ 在 -3.0
ㄗㄞˋ 再 -3.5
ㄏㄨㄟˋ-ㄗㄞˋ 會在 -4.22800493
ㄏㄨㄟˋ-ㄗㄞˋ 惠在 -9.0
ㄊㄚ-ㄗㄞˋ 他在 -4.41638718
ㄊㄚ-ㄗㄞˋ 她在 -5.02848465
ㄊㄚ-ㄗㄞˋ 牠在 -5.12000000
ㄋㄧˇ-ㄗㄞˋ 你在 -4.55136339
ㄋㄧˇ-ㄗㄞˋ 妳在 -5.74013569
ㄐㄧㄚ-ㄗㄞˋ 家在 -5.0
ㄒㄧㄢˋ-ㄗㄞˋ 現在 -2.0
ㄗㄞˋ-ㄐㄧㄚ 在家 -3.0
ㄏㄠˇ-ㄗㄞˋ 好在 -4.0
ㄏㄠˇ-ㄗㄞˋ 好再 -6.0
"""


class TestAddTwins(unittest.TestCase):
    def setUp(self):
        errors.clear()
        warnings.clear()
        self.lm = {}
        for line in TWIN_DATA.splitlines():
            reading, value, score = line.split(" ")
            self.lm.setdefault(reading, []).append((value, score))

    def values(self, reading):
        return [v for v, _ in self.lm[reading]]

    def test_adds_a_twin_a_tenth_below_the_zai_entry(self):
        added = add_twins(0, self.lm, ["會", "他"])
        self.assertEqual(
            [(r, v, t) for r, v, t, _ in added],
            [("ㄊㄚ-ㄗㄞˋ", "他在", "他再"), ("ㄏㄨㄟˋ-ㄗㄞˋ", "會在", "會再")],
        )
        self.assertEqual(
            self.lm["ㄏㄨㄟˋ-ㄗㄞˋ"], [("會在", "-4.22800493"), ("會再", "-4.32800493"), ("惠在", "-9.0")]
        )
        self.assertEqual(self.values("ㄊㄚ-ㄗㄞˋ"), ["他在", "他再", "她在", "牠在"])
        self.assertEqual(warnings, [])

    def test_only_the_listed_prefixes(self):
        add_twins(0, self.lm, ["會"])
        self.assertEqual(self.values("ㄊㄚ-ㄗㄞˋ"), ["他在", "她在", "牠在"])
        self.assertEqual(self.values("ㄋㄧˇ-ㄗㄞˋ"), ["你在", "妳在"])

    def test_words_where_zai_is_part_of_the_word_are_left_alone(self):
        # 現在 / 在家 are not "<prefix>在" for a listed prefix, 家在 only if 家 is listed
        add_twins(0, self.lm, ["會"])
        self.assertEqual(self.values("ㄒㄧㄢˋ-ㄗㄞˋ"), ["現在"])
        self.assertEqual(self.values("ㄗㄞˋ-ㄐㄧㄚ"), ["在家"])
        self.assertEqual(self.values("ㄗㄞˋ"), ["在", "再"])

    def test_an_existing_twin_is_kept(self):
        added = add_twins(0, self.lm, ["好"])
        self.assertEqual(added, [])
        self.assertEqual(self.lm["ㄏㄠˇ-ㄗㄞˋ"], [("好在", "-4.0"), ("好再", "-6.0")])

    def test_a_prefix_without_a_zai_entry_is_a_warning(self):
        add_twins(7, self.lm, ["會", "請"])
        self.assertEqual(warnings, [(7, "twin prefix 請 has no 請在 entry")])
        self.assertEqual(errors, [])

    def test_float_scores_stay_float(self):
        lm = {"ㄏㄨㄟˋ-ㄗㄞˋ": [("會在", -4.5)]}
        add_twins(0, lm, ["會"])
        self.assertEqual(lm["ㄏㄨㄟˋ-ㄗㄞˋ"][1][0], "會再")
        self.assertAlmostEqual(lm["ㄏㄨㄟˋ-ㄗㄞˋ"][1][1], -4.6)

    def test_read_twin_prefixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "twins.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("# comment\n\n會\n就  # trailing comment\n會\n應該\n")
            self.assertEqual(read_twin_prefixes(path), ["會", "就", "應該"])

    def test_the_shipped_prefix_list(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prefixes = read_twin_prefixes(os.path.join(here, "zai-twins.txt"))
        self.assertEqual(len(prefixes), 18)
        for part_of_word in ["實", "現", "存", "所", "人"]:
            self.assertNotIn(part_of_word, prefixes)
