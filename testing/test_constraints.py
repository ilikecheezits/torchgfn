import pytest
from src.gfn.constraints import parse_target

def test_parse_target_simple():
    dot_bracket = "((.))"
    expected_pairs = [4, 3, -1, 1, 0]
    assert parse_target(dot_bracket) == expected_pairs

def test_parse_target_unpaired():
    dot_bracket = "..."
    expected_pairs = [-1, -1, -1]
    assert parse_target(dot_bracket) == expected_pairs

def test_parse_target_complex_unbalanced():
    dot_bracket = "((.())()).)" # This string is deliberately unbalanced (6 open, 5 close)
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