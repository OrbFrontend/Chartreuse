"""Self-check for opener_repeat: collapse scores ~1, varied prose scores ~0."""
from depurple.objective import opener_repeat


def test_opener_repeat():
    collapse = "She nods. She turns. She doesn't. She waits."   # one opener, 3/3 dups
    assert opener_repeat([collapse]) == 1.0

    varied = "The rain fell. Marco laughed. Nobody moved. Quietly, she left."
    assert opener_repeat([varied]) == 0.0

    # hedge-slop varies its openers too -> reads ~base, NOT flagged
    slop = "It was almost imperceptible. Seeming to settle, the dust hung. Albeit regal, she bowed."
    assert opener_repeat([slop]) == 0.0

    half = "He runs. He hides. Dawn breaks."                    # 1 dup of 2 gaps
    assert opener_repeat([half]) == 0.5

    assert opener_repeat([""]) == 0.0                            # empty -> no crash


if __name__ == "__main__":
    test_opener_repeat()
    print("ok")
