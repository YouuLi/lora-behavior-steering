from lora_behavior_steering.metrics import bigram_repetition, content_unstable, sentence_count, word_count


def test_basic_counts():
    text = "Sleep supports memory. It also helps recovery."
    assert word_count(text) == 7
    assert sentence_count(text) == 2
    assert not content_unstable(text)


def test_repetition_is_unstable():
    text = "go now go now go now go now go now go now"
    assert bigram_repetition(text) >= 0.35
    assert content_unstable(text)

