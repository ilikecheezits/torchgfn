import math
from collections import Counter

import pytest

from src.gfn.constraints import get_valid_mask, get_designability_score, parse_target, sample_uniform_valid


def test_get_designability_score():
    """Gatekeeper Test 1: The "Designability Score" (Combinatorics Check)"""
    dot_bracket = "((...))"
    # P = 2, U = 3
    # log10(N) = 2 * log10(6) + 3 * log10(4)
    expected_log_n = 2 * math.log10(6) + 3 * math.log10(4) # Approx 3.362
    assert get_designability_score(dot_bracket) == pytest.approx(expected_log_n, abs=1e-2)


def test_lawyer_scenarios():
    """Gatekeeper Test 2: The "Lawyer" (Constraint Logic Check)"""
    dot_bracket = "((...))"
    pair_map = parse_target(dot_bracket)
    
    # Scenario A (Standard Pair)
    partial_seq_A = "A......"
    mask_A = get_valid_mask(6, partial_seq_A, pair_map)
    assert mask_A == [0, 1, 0, 0], "Scenario A failed: 'A' should only pair with 'U'"

    # Scenario B (Wobble Pair)
    partial_seq_B = "G......"
    mask_B = get_valid_mask(6, partial_seq_B, pair_map)
    assert mask_B == [0, 1, 1, 0], "Scenario B failed: 'G' should pair with 'C' or 'U'"
    
    # Scenario C (Impossible Pair - Edge Case)
    partial_seq_C = "C......"
    mask_C = get_valid_mask(6, partial_seq_C, pair_map)
    assert mask_C == [0, 0, 0, 1], "Scenario C failed: 'C' should only pair with 'G'"


def test_sampler_statistical_distribution():
    """Gatekeeper Test 3: The "Sampler" (Statistical Distribution Check)"""
    dot_bracket = "((...))"  # Length 7, 2 pairs, 3 unpaired
    num_samples = 1000
    
    generated_sequences = sample_uniform_valid(dot_bracket, num_samples)
    
    assert len(generated_sequences) == num_samples
    
    pair_map = parse_target(dot_bracket)
    n = len(dot_bracket)
    
    VALID_BASE_PAIRS_CHECK = {
        'A': ['U'], 'U': ['A', 'G'], 'C': ['G'], 'G': ['C', 'U']
    }

    # Check 1 (Validity)
    for seq in generated_sequences:
        assert len(seq) == n
        for i in range(n):
            if pair_map[i] != -1:  # Check paired bases
                j = pair_map[i]
                if i < j:
                    base_i = seq[i]
                    base_j = seq[j]
                    
                    # Check both directions of the pair
                    if base_j not in VALID_BASE_PAIRS_CHECK.get(base_i, []):
                         assert base_i in VALID_BASE_PAIRS_CHECK.get(base_j,[]), \
                            f"Invalid pair {base_i}-{base_j} at indices {i}-{j} in sequence {seq}"

    # Check 2 (Diversity/Uniformity)
    unpaired_bases = [seq[2] for seq in generated_sequences]
    counts = Counter(unpaired_bases)
    
    for base in ['A', 'C', 'G', 'U']:
        assert 200 < counts[base] < 300, f"Base {base} count {counts[base]} is not within the expected range (200-300)"


def test_parse_target_simple():
    dot_bracket = "((.))"
    expected_pairs = [4, 3, -1, 1, 0]
    assert parse_target(dot_bracket) == expected_pairs

def test_parse_target_unpaired():
    dot_bracket = "..."
    expected_pairs = [-1, -1, -1]
    assert parse_target(dot_bracket) == expected_pairs


def test_parse_target_complex_unbalanced():
    dot_bracket = ("((.())()).)")
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        parse_target(dot_bracket)


def test_parse_target_empty():
    dot_bracket = ""
    expected_pairs = []
    assert parse_target(dot_bracket) == expected_pairs


def test_parse_target_mismatched_open():
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        parse_target("(((")


def test_parse_target_mismatched_close():
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        parse_target(")))")


def test_parse_target_mismatched_interleaved():
    with pytest.raises(ValueError, match="Mismatched parentheses"):
        parse_target("(.))")


def test_parse_target_invalid_char():
    with pytest.raises(ValueError, match="Invalid character"):
        parse_target("([)]")


def test_parse_target_balanced_complex():
    dot_bracket = "((()))"
    expected_pairs = [5, 4, 3, 2, 1, 0]
    assert parse_target(dot_bracket) == expected_pairs


# Tests for get_valid_mask
def test_get_mask_case_a_unpaired():
    pair_map = [4, 3, -1, 1, 0]  # ((.))
    current_idx = 2  # '.' is unpaired
    partial_seq = "((."
    expected_mask = [1, 1, 1, 1]
    assert get_valid_mask(current_idx, partial_seq, pair_map) == expected_mask


def test_get_mask_case_a_beyond_map_length():
    pair_map = [4, 3, -1, 1, 0]  # ((.))
    current_idx = 5  # Index beyond the pair_map length
    partial_seq = "((.))"
    expected_mask = [1, 1, 1, 1]
    assert get_valid_mask(current_idx, partial_seq, pair_map) == expected_mask


def test_get_mask_case_b_opening_bracket():
    pair_map = [4, 3, -1, 1, 0]  # ((.))
    current_idx = 0  # '(' is an opening bracket, pairs with 4
    partial_seq = "("
    expected_mask = [1, 1, 1, 1]
    assert get_valid_mask(current_idx, partial_seq, pair_map) == expected_mask


def test_get_mask_case_b_opening_bracket_further_in():
    pair_map = [4, 3, -1, 1, 0]  # ((.))
    current_idx = 1  # '(' is an opening bracket, pairs with 3
    partial_seq = "(("
    expected_mask = [1, 1, 1, 1]
    assert get_valid_mask(current_idx, partial_seq, pair_map) == expected_mask


def test_get_mask_other_case_closing_bracket():
    pair_map = [4, 3, -1, 1, 0]  # ((.))
    current_idx = 4  # ')' is a closing bracket, pairs with 0
    partial_seq = "((.))"
    expected_mask = [0, 0, 0, 0]  # Should return all zeros
    assert get_valid_mask(current_idx, partial_seq, pair_map) == expected_mask


def test_get_mask_case_c_paired_with_A():
    # Target: AU
    # pair_map: [1, 0]
    # partial_seq: "AU"
    # current_idx = 1 (closing 'U'), paired with 0 ('A')
    pair_map = [1, 0]
    current_idx = 1
    partial_seq = "AU"
    expected_mask = [0, 1, 0, 0]  # Only U allowed for A
    assert get_valid_mask(current_idx, partial_seq, pair_map) == expected_mask


def test_get_mask_case_c_paired_with_U():
    # Target: UA
    # pair_map: [1, 0]
    # partial_seq: "UA"
    # current_idx = 1 (closing 'A'), paired with 0 ('U')
    pair_map = [1, 0]
    current_idx = 1
    partial_seq = "UA"
    expected_mask = [1, 0, 0, 1]  # A or G allowed for U
    assert get_valid_mask(current_idx, partial_seq, pair_map) == expected_mask


def test_get_mask_case_c_paired_with_C():
    # Target: CG
    # pair_map: [1, 0]
    # partial_seq: "CG"
    # current_idx = 1 (closing 'G'), paired with 0 ('C')
    pair_map = [1, 0]
    current_idx = 1
    partial_seq = "CG"
    expected_mask = [0, 0, 0, 1]  # Only G allowed for C
    assert get_valid_mask(current_idx, partial_seq, pair_map) == expected_mask


def test_get_mask_case_c_paired_with_G():
    # Target: GC
    # pair_map: [1, 0]
    # partial_seq: "GC"
    # current_idx = 1 (closing 'C'), paired with 0 ('G')
    pair_map = [1, 0]
    current_idx = 1
    partial_seq = "GC"
    expected_mask = [0, 1, 1, 0]  # U or C allowed for G
    assert get_valid_mask(current_idx, partial_seq, pair_map) == expected_mask


def test_get_mask_case_c_paired_with_G_longer_seq():
    # Target: G...C
    # pair_map: [3, -1, -1, 0]
    # partial_seq: "GAUC"
    # current_idx = 3 (closing 'C'), paired with 0 ('G')
    pair_map = [3, -1, -1, 0]
    current_idx = 3
    partial_seq = "GAUC"
    expected_mask = [0, 1, 1, 0]  # C or U allowed for G
    assert get_valid_mask(current_idx, partial_seq, pair_map) == expected_mask


def test_get_mask_case_c_paired_base_invalid():
    # Target: X...Y (where X is an invalid base)
    pair_map = [1, 0]
    current_idx = 1
    partial_seq = "XY"  # X is invalid
    expected_mask = [0, 0, 0, 0]  # Should return all zeros
    assert get_valid_mask(current_idx, partial_seq, pair_map) == expected_mask
