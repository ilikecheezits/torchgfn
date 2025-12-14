import math
import random


def parse_target(dot_bracket_str: str) -> list[int]:
    n = len(dot_bracket_str)
    pairs = [-1] * n
    stack = []

    for i, char in enumerate(dot_bracket_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if not stack:
                raise ValueError("Mismatched parentheses in dot-bracket string")
            j = stack.pop()
            pairs[i] = j
            pairs[j] = i
        elif char == ".":
            pass
        else:
            raise ValueError(f"Invalid character in dot-bracket string: {char}")

    if stack:
        raise ValueError("Mismatched parentheses in dot-bracket string")

    return pairs


BASE_TO_INDEX = {"A": 0, "U": 1, "C": 2, "G": 3}
INDEX_TO_BASE = {0: "A", 1: "U", 2: "C", 3: "G"}


def _get_allowed_bases_mask(paired_base: str) -> list[int]:
    """
    Returns a mask of allowed bases for current_idx based on the base at the paired index.
    """
    mask = [0, 0, 0, 0]  # A, U, C, G
    if paired_base == "A":
        mask[BASE_TO_INDEX["U"]] = 1  # A pairs with U
    elif paired_base == "U":
        mask[BASE_TO_INDEX["A"]] = 1  # U pairs with A
        mask[BASE_TO_INDEX["G"]] = 1  # U pairs with G
    elif paired_base == "C":
        mask[BASE_TO_INDEX["G"]] = 1  # C pairs with G
    elif paired_base == "G":
        mask[BASE_TO_INDEX["C"]] = 1  # G pairs with C
        mask[BASE_TO_INDEX["U"]] = 1  # G pairs with U
    return mask


def get_valid_mask(current_idx: int, partial_seq: str, pair_map: list[int]) -> list[int]:
    """
    Determines the allowed actions (mask) for a given current index in a sequence,
    considering RNA base-pairing rules for closing brackets.

    Args:
        current_idx: The current index being considered in the sequence.
        partial_seq: The sequence constructed so far (can be dot-bracket or actual bases).
                     When current_idx is a closing bracket, this is assumed to be the
                     actual base sequence (e.g., "AUCG").
        pair_map: A list of integers representing the pairing of the target structure,
                  where pair_map[i] = j means index i binds to j, and -1 if unpaired.

    Returns:
        A list of integers (mask) where 1 means allowed and 0 means disallowed.
        The order of bases in the mask is A, U, C, G.
    """
    if current_idx >= len(pair_map) or (
        current_idx < len(pair_map) and pair_map[current_idx] == -1
    ):
        # Case A: Unpaired position or beyond map length, all bases allowed for now
        return [1, 1, 1, 1]

    j = pair_map[current_idx]

    if j > current_idx:
        # Case B: current_idx is an opening bracket, all bases allowed for now
        return [1, 1, 1, 1]

    if j < current_idx:
        # Case C: current_idx is a closing bracket (pair is in past at index j)
        # Assumes partial_seq contains actual bases for this case.
        if j < 0 or j >= len(partial_seq) or partial_seq[j] not in BASE_TO_INDEX:
            # Handle error or return a default mask if partial_seq[j] is invalid
            # For now, return all zeros if the paired base is unknown or invalid.
            return [0, 0, 0, 0]

        base_at_j = partial_seq[j]
        return _get_allowed_bases_mask(base_at_j)

    # Fallback for any other unexpected cases
    return [0, 0, 0, 0]


def get_designability_score(dot_bracket_str: str) -> float:
    """
    Calculates the log10 of the approximate search space size for a given secondary structure.

    The approximate number of valid sequences is N ≈ 6^P * 4^U, where P is the number of
    base pairs and U is the number of unpaired bases.

    Args:
        dot_bracket_str: The dot-bracket string representing the target structure.

    Returns:
        The log10 of the approximate search space size.
    """
    p_count = dot_bracket_str.count("(")
    u_count = dot_bracket_str.count(".")

    # log10(N) = P * log10(6) + U * log10(4)
    log_n = p_count * math.log10(6) + u_count * math.log10(4)
    return log_n


import random


def sample_uniform_valid(dot_bracket_str: str, num_samples: int) -> list[str]:
    """
    Generates valid RNA sequences uniformly at random for a given secondary structure.

    Args:
        dot_bracket_str: The dot-bracket string representing the target structure.
        num_samples: The number of sequences to generate.

    Returns:
        A list of generated RNA sequences.
    """
    n = len(dot_bracket_str)
    pair_map = parse_target(dot_bracket_str)

    # Define valid base pairs (Watson-Crick + Wobble)
    VALID_BASE_PAIRS = {
        "A": ["U"],
        "U": ["A", "G"],
        "C": ["G"],
        "G": ["C", "U"],
    }

    generated_sequences = []

    for _ in range(num_samples):
        sequence = [""] * n

        # Keep track of already processed paired indices to avoid redundant assignment
        # assigned_paired_indices = set() # This is not strictly necessary if we only assign when i < j

        for i in range(n):
            if sequence[i] != "":  # Already assigned as part of a pair
                continue

            if pair_map[i] == -1:  # Unpaired base
                sequence[i] = random.choice(list(BASE_TO_INDEX.keys()))
            else:  # Paired base
                j = pair_map[i]

                # Only process the pair once (e.g., when i < j)
                if i < j:
                    # Determine all valid base pairs for i and j
                    compatible_pairs = []
                    for base_i in list(BASE_TO_INDEX.keys()):
                        for base_j in VALID_BASE_PAIRS.get(base_i, []):
                            compatible_pairs.append((base_i, base_j))

                    if not compatible_pairs:
                        raise ValueError(
                            f"No compatible base pairs found for indices {i} and {j}"
                        )

                    chosen_pair = random.choice(compatible_pairs)
                    sequence[i] = chosen_pair[0]
                    sequence[j] = chosen_pair[1]
                # No need for else here, if i > j, it would have been assigned when processing j

        generated_sequences.append("".join(sequence))

    return generated_sequences
