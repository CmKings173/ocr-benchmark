def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, 1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1]


def normalized_edit_error(reference: str, hypothesis: str) -> float:
    denominator = max(len(reference), 1)
    return edit_distance(reference, hypothesis) / denominator


def cer(reference: str, hypothesis: str) -> float:
    return normalized_edit_error(reference, hypothesis)


def wer(reference: str, hypothesis: str) -> float:
    return normalized_edit_error(reference.split(), hypothesis.split())
