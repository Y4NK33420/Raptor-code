import itertools

import pytest

from raptor.chunking import CHUNKERS, SentenceChunker, default_tokenizer, sentence_spans, split_text

TOK = default_tokenizer()


def ntok(s: str) -> int:
    return len(TOK.encode(s))


def test_sentence_spans_keep_punctuation():
    text = 'Hello there. "Is it?" she asked! Yes.\nNext line'
    sents = [text[a:b] for a, b in sentence_spans(text)]
    assert sents == ["Hello there.", '"Is it?"', "she asked!", "Yes.", "Next line"]


def test_chunks_preserve_punctuation_and_offsets(document):
    chunks = SentenceChunker(max_tokens=50).split(document)
    assert chunks
    for c in chunks:
        assert document[c.start : c.end] == c.text
        assert ntok(c.text) <= 50 + 2  # sentence join may add a token at boundaries
    # The old implementation dropped every period; the new one keeps them.
    assert any(c.text.endswith(".") for c in chunks)


def test_chunks_cover_all_sentences(document):
    chunks = SentenceChunker(max_tokens=80).split(document)
    joined = " ".join(c.text for c in chunks)
    for a, b in sentence_spans(document):
        assert document[a:b] in joined


def test_overlap_carries_tail_within_budget(document):
    chunker = SentenceChunker(max_tokens=60, overlap_tokens=20)
    chunks = chunker.split(document)
    shared = 0
    for prev, cur in itertools.pairwise(chunks):
        if cur.start < prev.end:  # overlapping
            shared += 1
            assert ntok(document[cur.start : prev.end]) <= 20 + 2
    assert shared > 0


def test_oversized_sentence_is_split_below_limit():
    long_sentence = ", ".join(f"clause number {i} keeps going" for i in range(60)) + "."
    chunks = SentenceChunker(max_tokens=30).split(long_sentence)
    assert len(chunks) > 1
    assert all(ntok(c.text) <= 32 for c in chunks)


def test_unpunctuated_giant_token_run_is_windowed():
    text = "word " * 500
    chunks = SentenceChunker(max_tokens=40).split(text)
    assert len(chunks) >= 10
    assert all(ntok(c.text) <= 41 for c in chunks)


def test_empty_and_whitespace():
    assert SentenceChunker(10).split("") == []
    assert SentenceChunker(10).split("   \n\n  ") == []


def test_invalid_params():
    with pytest.raises(ValueError):
        SentenceChunker(0)
    with pytest.raises(ValueError):
        SentenceChunker(10, overlap_tokens=10)


def test_split_text_compat_and_registry():
    assert split_text("A b. C d.", TOK, 100) == ["A b. C d."]
    chunker = CHUNKERS.create("sentence", max_tokens=5)
    assert isinstance(chunker, SentenceChunker)
    with pytest.raises(ValueError, match="unknown chunker"):
        CHUNKERS.create("nope")
