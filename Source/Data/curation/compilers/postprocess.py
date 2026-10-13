import argparse
import os
from .compiler_utils import HEADER
from ..mandarin.grid import most_plausible_walk
import sys
import unittest

errors: list[tuple[int, str]] = []
warnings: list[tuple[int, str]] = []
epsilon = 0.0001


def set_epsilon(e):
    global epsilon
    epsilon = e


def accrue_error(lineno, err):
    errors.append((lineno, err))


def accrue_warning(lineno, warning):
    warnings.append((lineno, warning))


def show_errors_and_warnings():
    combined = [(lineno, f"error (line {lineno}): {msg}\n") for lineno, msg in errors]
    combined += [
        (lineno, f"warning (line {lineno}): {msg}\n") for lineno, msg in warnings
    ]

    combined = sorted(combined, key=lambda x: x[0])

    for _, msg in combined:
        sys.stderr.write(msg)


def segmented_values(nodes):
    return "-".join(n.value for n in nodes)


def find_top_unigram_in_lm(lm, reading):
    if reading not in lm:
        return None

    unigrams = lm[reading]
    v, s = unigrams[0]
    return (v, float(s))


def find_score_in_lm(lm, reading, value):
    if reading not in lm:
        return None

    unigrams = lm[reading]
    for unigram in unigrams:
        if unigram[0] == value:
            return float(unigram[1])

    return None


def replace_score_in_lm(lm, reading, value, new_score):
    if reading not in lm:
        raise ValueError(f"reading {reading} not in language model!")

    unigrams = lm[reading]

    has_replacement = False
    new_unigrams = []
    for unigram in unigrams:
        uv, us = unigram
        if uv == value:
            if type(us) == str:
                new_unigrams.append((uv, "%.8f" % new_score))
            else:
                new_unigrams.append((uv, new_score))
            has_replacement = True
        else:
            new_unigrams.append((uv, us))

    if not has_replacement:
        raise ValueError("reading:value %s:%s not in LM" % (reading, value))

    unigrams = sorted(new_unigrams, key=lambda x: float(x[1]), reverse=True)
    lm[reading] = unigrams
    return True


def promote_over_single_syllables(lineno, lm, value, reading):

    readings = reading.split("-")

    if len(value) != len(readings):
        return accrue_error(lineno, "number of codepoints don't match readings")

    our_score = find_score_in_lm(lm, reading, value)
    if not our_score:
        return accrue_error(lineno, f"reading:value {reading}:{value} not in LM")

    # validate data
    unigrams = [find_top_unigram_in_lm(lm, r) for r in readings]
    if not all(unigrams):
        return accrue_error(lineno, "cannot find all single-syllable readings")

    their_scores = sum(u[1] for u in unigrams)

    if their_scores <= our_score:
        return accrue_error(lineno, "no need to promote")

    our_score = their_scores + epsilon
    return replace_score_in_lm(lm, reading, value, our_score)


def promote_over_peers(lineno, lm, value, reading):
    our_score = find_score_in_lm(lm, reading, value)
    if not our_score:
        return accrue_error(lineno, f"reading:value {reading}:{value} not in LM")

    top_gram = find_top_unigram_in_lm(lm, reading)
    if not top_gram:
        return accrue_error(lineno, f"no unigrams found for reading: {reading}")

    top_value, top_score = top_gram
    if top_value == value:
        return accrue_error(lineno, f"value {value} already top among peers")

    our_score = float(top_score) + epsilon
    return replace_score_in_lm(lm, reading, value, our_score)


# The "再" twins (McBopomofoLM v3.4): "<prefix>在" without "<prefix>再" gets one.
TWIN_FROM = "在"
TWIN_TO = "再"
TWIN_SYLLABLE = "ㄗㄞˋ"
# log10 units, so the twin never outranks the entry it was made from
TWIN_SCORE_DELTA = 0.1


def read_twin_prefixes(path):
    """Reads the prefix list: one prefix per line, '#' starts a comment."""
    prefixes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line and line not in prefixes:
                prefixes.append(line)
    return prefixes


def add_twins(lineno, lm, prefixes):
    """Adds "<prefix>再" for every "<prefix>在" (reading ending in ㄗㄞˋ, one
    syllable per character) that has no twin yet, in the same reading, scored
    TWIN_SCORE_DELTA below the "在" entry. The new entry goes after the entries
    that score at least as high, so a reading stays ordered by score. Returns
    the list of (reading, "在 entry", "twin", twin score)."""
    wanted = set(prefixes)
    used = set()
    added = []

    for reading in sorted(lm.keys(), key=lambda x: x.encode()):
        syllables = reading.split("-")
        if syllables[-1] != TWIN_SYLLABLE:
            continue

        for value, score in list(lm[reading]):
            if len(value) < 2 or len(value) != len(syllables):
                continue
            if not value.endswith(TWIN_FROM) or value[:-1] not in wanted:
                continue

            used.add(value[:-1])
            twin = value[:-1] + TWIN_TO
            if any(v == twin for v, _ in lm[reading]):
                continue

            twin_score = float(score) - TWIN_SCORE_DELTA
            entry = (twin, "%.8f" % twin_score) if type(score) == str else (twin, twin_score)
            position = len(lm[reading])
            while position > 0 and float(lm[reading][position - 1][1]) < twin_score:
                position -= 1
            lm[reading].insert(position, entry)
            added.append((reading, value, twin, twin_score))

    for prefix in prefixes:
        if prefix not in used:
            accrue_warning(lineno, f"twin prefix {prefix} has no {prefix}{TWIN_FROM} entry")

    return added


def run_assert(lineno, lm, readings, expected, warn_only=False):
    nodes = most_plausible_walk(readings.split("-"), lm)
    result = segmented_values(nodes)

    if result != expected:
        if warn_only:
            return accrue_warning(lineno, f"expected: {expected}, actual: {result}")
        else:
            return accrue_error(lineno, f"expected: {expected}, actual: {result}")


def postprocess(input, directive, output):
    directive_dir = os.path.dirname(os.path.abspath(directive))
    lm = {}

    with open(input) as f:
        lines = f.readlines()

        for line in lines[1:]:
            # don't use bare split() since it also splits full-width spaces
            r, v, s = line.strip().split(" ")

            if r in lm:
                lm[r].append((v, s))
            else:
                lm[r] = [(v, s)]

    with open(directive) as f:
        lineno = 0

        for line in f:
            lineno += 1

            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue

            elements = line.split()
            if elements[0] == "assert":
                readings = elements[1]
                expected = elements[2]
                run_assert(lineno, lm, readings, expected)
            elif elements[0] == "before":
                readings = elements[1]
                expected = elements[2]
                run_assert(lineno, lm, readings, expected, warn_only=True)
            elif elements[0] == "promote-over-single-syllables":
                value = elements[1]
                reading = elements[2]
                promote_over_single_syllables(lineno, lm, value, reading)
            elif elements[0] == "promote-over-peers":
                value = elements[1]
                reading = elements[2]
                promote_over_peers(lineno, lm, value, reading)
            elif elements[0] == "add-twins":
                prefixes = read_twin_prefixes(os.path.join(directive_dir, elements[1]))
                add_twins(lineno, lm, prefixes)
            elif elements[0] == "epsilon":
                set_epsilon(float(elements[1]))
            else:
                accrue_error(lineno, f"unknown command: {elements[0]}")

    if errors or warnings:
        show_errors_and_warnings()

    if warnings:
        print("%d warning(s) found" % len(warnings))

    if errors:
        print("%d error(s) found" % len(errors))
        sys.exit(1)

    with open(output, "w") as f:
        f.write(HEADER)

        for r in sorted(lm.keys(), key=lambda x: x.encode()):
            for v, s in lm[r]:
                f.write("%s %s %s\n" % (r, v, s))


def main():
    parser = argparse.ArgumentParser(description="postprocess compiled phrase database")
    parser.add_argument("--input", required=True, help="path to source data")
    parser.add_argument("--directive", required=True, help="path to directive file")
    parser.add_argument("--output", required=True, help="path to postprocessed output")
    args = parser.parse_args()
    postprocess(input=args.input, directive=args.directive, output=args.output)


if __name__ == "__main__":
    main()
