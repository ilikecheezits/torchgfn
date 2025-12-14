def parse_target(dot_bracket_str: str) -> list[int]:
    n = len(dot_bracket_str)
    pairs = [-1] * n
    stack = []

    for i, char in enumerate(dot_bracket_str):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if not stack:
                raise ValueError("Mismatched parentheses in dot-bracket string")
            j = stack.pop()
            pairs[i] = j
            pairs[j] = i
        elif char == '.':
            pass
        else:
            raise ValueError(f"Invalid character in dot-bracket string: {char}")

    if stack:
        raise ValueError("Mismatched parentheses in dot-bracket string")

    return pairs
