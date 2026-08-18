from __future__ import annotations


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, 1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1]


def normalized_edit_error(reference: str | list[str], hypothesis: str | list[str]) -> float | None:
    # An empty reference in the dataset means that this metric is not
    # annotated, not that the expected document contains zero characters.
    if not reference or (isinstance(reference, str) and not reference.strip()):
        return None
    denominator = max(len(reference), 1)
    return edit_distance(reference, hypothesis) / denominator


def cer(reference: str, hypothesis: str) -> float | None:
    return normalized_edit_error(reference, hypothesis)


def wer(reference: str, hypothesis: str) -> float | None:
    if not reference.strip():
        return None
    return normalized_edit_error(reference.split(), hypothesis.split())
