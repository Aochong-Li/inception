"""
Unit tests for evaluation/scripts/token_helpers.py (Subtask 2).

These tests are ACCEPTANCE TESTS written before implementation. They are
expected to FAIL with ImportError until token_helpers.py is created.

Each test follows the AAA pattern (Arrange / Act / Assert) and is fully
self-contained. No file I/O, no network calls, no shared mutable state.

Run with:
    cd /home/md2292/inception-eval && uv run python -m pytest evaluation/scripts/test_token_helpers.py -v

All tests will fail with ImportError until implementation is complete.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module import — fail loudly with a clear message if not implemented yet.
# We do NOT use pytest.importorskip because we want the specific ImportError
# to surface as a test failure rather than a silent skip.  The instruction to
# make tests FAIL (not skip) until implementation is complete requires this.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from evaluation.scripts.token_helpers import (
        count_tokens,
        get_encoding,
        normalize_for_token_count,
        split_at_think_close,
    )
    _IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    _IMPORT_ERROR = exc
    # Create stubs so the module parses; each test that calls these will fail
    # at the assertion that verifies the import succeeded.
    split_at_think_close = None  # type: ignore[assignment]
    normalize_for_token_count = None  # type: ignore[assignment]
    count_tokens = None  # type: ignore[assignment]
    get_encoding = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Fixture — fail every test immediately if the module isn't importable yet
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def require_module():
    """All tests in this file require token_helpers to be importable."""
    if _IMPORT_ERROR is not None:
        pytest.fail(
            f"evaluation.scripts.token_helpers is not yet implemented.\n"
            f"ImportError: {_IMPORT_ERROR}\n"
            f"This test is expected to fail until token_helpers.py is created."
        )


# ===========================================================================
# Helpers
# ===========================================================================

def _make_encoding():
    """Return the cl100k_base tiktoken encoding for use in count_tokens tests."""
    return get_encoding("cl100k_base")


# ===========================================================================
# split_at_think_close
# ===========================================================================

class TestSplitAtThinkClose:
    """
    split_at_think_close(text) -> (pre_think, post_think, had_delimiter)

    Mirrors the exact semantics of src/main.py split("</think>"):
      pre_think  = text.split("</think>")[0]
      post_think = text.split("</think>")[1]   (empty string when no delimiter)
      had_delimiter = "</think>" in text
    """

    # --- Happy path ---

    def test_WithSingleDelimiter_SplitsAtFirstOccurrence(self):
        # Arrange
        text = "chain of thought</think>visible content"

        # Act
        pre, post, had = split_at_think_close(text)

        # Assert
        assert pre == "chain of thought", (
            "pre_think must be everything before the first </think>"
        )
        assert post == "visible content", (
            "post_think must be everything between first and second </think> "
            "(i.e. str.split()[1] semantics)"
        )
        assert had is True, "had_delimiter must be True when </think> is present"

    def test_WithDelimiter_HadDelimiterIsTrue(self):
        # Arrange
        text = "some thinking</think>some response"

        # Act
        _, _, had = split_at_think_close(text)

        # Assert
        assert had is True

    def test_WithDelimiter_ReturnsExactStringSlices(self):
        # Arrange — use a string whose split result is fully determined
        text = "AAAA</think>BBBB"

        # Act
        pre, post, had = split_at_think_close(text)

        # Assert
        assert pre == "AAAA"
        assert post == "BBBB"
        assert had is True

    # --- No delimiter ---

    def test_WithoutDelimiter_ReturnsFullTextAsPreThink(self):
        # Arrange
        text = "just plain text with no closing think tag"

        # Act
        pre, post, had = split_at_think_close(text)

        # Assert
        assert pre == text, (
            "When no </think> delimiter is present, pre_think must be the full input text"
        )

    def test_WithoutDelimiter_PostThinkIsEmptyString(self):
        # Arrange
        text = "no delimiter here"

        # Act
        _, post, _ = split_at_think_close(text)

        # Assert
        assert post == "", "post_think must be empty string when no </think> is present"

    def test_WithoutDelimiter_HadDelimiterIsFalse(self):
        # Arrange
        text = "no delimiter here"

        # Act
        _, _, had = split_at_think_close(text)

        # Assert
        assert had is False

    # --- Empty string ---

    def test_EmptyString_ReturnsTripleOfEmptyStringEmptyStringFalse(self):
        # Arrange
        text = ""

        # Act
        pre, post, had = split_at_think_close(text)

        # Assert
        assert pre == "", "pre_think of empty input must be empty string"
        assert post == "", "post_think of empty input must be empty string"
        assert had is False, "had_delimiter of empty input must be False"

    # --- Multiple delimiters ---

    def test_WithMultipleDelimiters_SplitsAtFirstOccurrence(self):
        # Arrange — three segments separated by two </think> tags
        # str.split("</think>") = ["pre", "middle", "post"]
        # [0] = "pre", [1] = "middle" (NOT "middle</think>post")
        text = "pre</think>middle</think>post"

        # Act
        pre, post, had = split_at_think_close(text)

        # Assert
        assert pre == "pre", (
            "With multiple </think>, pre_think must be the text before the FIRST delimiter"
        )
        assert post == "middle", (
            "With multiple </think>, post_think must be the text between the FIRST "
            "and SECOND delimiter — matching str.split('</think>')[1] semantics, "
            "NOT everything after the first delimiter"
        )
        assert had is True

    def test_WithThreeDelimiters_PostThinkIsSecondSegmentOnly(self):
        # Arrange
        text = "A</think>B</think>C</think>D"

        # Act
        pre, post, had = split_at_think_close(text)

        # Assert
        assert pre == "A"
        assert post == "B", (
            "post_think must be str.split()[1], which is 'B' — not 'B</think>C</think>D'"
        )
        assert had is True

    # --- Delimiter at position 0 ---

    def test_DelimiterAtStartOfString_PreThinkIsEmptyString(self):
        # Arrange — delimiter is the very first characters
        text = "</think>rest of the text"

        # Act
        pre, post, had = split_at_think_close(text)

        # Assert
        assert pre == "", (
            "When </think> appears at position 0, pre_think must be empty string"
        )
        assert post == "rest of the text"
        assert had is True

    # --- Delimiter at end ---

    def test_DelimiterAtEndOfString_PostThinkIsEmptyString(self):
        # Arrange — delimiter is at the very end
        text = "body text here</think>"

        # Act
        pre, post, had = split_at_think_close(text)

        # Assert
        assert pre == "body text here", (
            "When </think> appears at end, pre_think must be the full body"
        )
        assert post == "", (
            "When </think> appears at end, post_think must be empty string"
        )
        assert had is True

    # --- Return type ---

    def test_ReturnType_IsThreeTuple(self):
        # Arrange
        text = "hello</think>world"

        # Act
        result = split_at_think_close(text)

        # Assert
        assert isinstance(result, tuple), "split_at_think_close must return a tuple"
        assert len(result) == 3, "split_at_think_close must return a 3-tuple"

    def test_ReturnType_ThirdElementIsBool(self):
        # Arrange
        text = "hello"

        # Act
        _, _, had = split_at_think_close(text)

        # Assert
        assert isinstance(had, bool), (
            "had_delimiter must be a Python bool, not an int or other truthy type"
        )

    def test_ReturnType_ThirdElementIsBoolWhenTrue(self):
        # Arrange
        text = "hello</think>world"

        # Act
        _, _, had = split_at_think_close(text)

        # Assert
        assert isinstance(had, bool), (
            "had_delimiter must be a Python bool, not an int or other truthy type"
        )


# ===========================================================================
# normalize_for_token_count
# ===========================================================================

class TestNormalizeForTokenCount:
    """
    normalize_for_token_count(text) -> str

    UNLIKE strip_special_tags() in token_cost_analysis.py, this function
    does NOT remove the body content of <think>...</think> blocks.  It only
    removes the delimiter tags themselves and XML wrapper tags.

    Critical semantic difference:
      strip_special_tags("<think>CoT body</think>content")  -> "content"
      normalize_for_token_count("<think>CoT body</think>content") -> "CoT bodycontent"
    """

    # --- <think> and </think> tag removal ---

    def test_RemovesOpenThinkTag(self):
        # Arrange
        text = "<think>some body text"

        # Act
        result = normalize_for_token_count(text)

        # Assert
        assert "<think>" not in result, (
            "normalize_for_token_count must remove the literal <think> tag"
        )

    def test_RemovesCloseThinkTag(self):
        # Arrange
        text = "some body text</think>"

        # Act
        result = normalize_for_token_count(text)

        # Assert
        assert "</think>" not in result, (
            "normalize_for_token_count must remove the literal </think> tag"
        )

    def test_RemovesBothThinkDelimiterTags_PreservesBodyText(self):
        # Arrange — the body text must be preserved, unlike strip_special_tags
        text = "<think>chain of thought body</think>visible content"

        # Act
        result = normalize_for_token_count(text)

        # Assert
        assert "chain of thought body" in result, (
            "normalize_for_token_count must PRESERVE the body text inside <think> tags. "
            "This is the critical difference from strip_special_tags(), which deletes "
            "the entire block. We want to COUNT thinking tokens, not erase them."
        )
        assert "visible content" in result, (
            "normalize_for_token_count must also preserve post-think content"
        )
        assert "<think>" not in result
        assert "</think>" not in result

    def test_DoesNotStripThinkBlockContent_UnlikeStripSpecialTags(self):
        # Arrange
        thinking_body = "I am reasoning step by step about the problem"
        text = f"<think>{thinking_body}</think>answer"

        # Act
        result = normalize_for_token_count(text)

        # Assert — this is the KEY behavioral contract
        assert thinking_body in result, (
            "CRITICAL: normalize_for_token_count must NOT strip the content inside "
            "<think>...</think> blocks. strip_special_tags() removes entire blocks; "
            "this function only removes the delimiter tags themselves. "
            "We need the token count to INCLUDE the thinking body."
        )

    # --- XML wrapper tag removal ---

    def test_RemovesReasoningWrapperTag(self):
        # Arrange
        text = "<reasoning>some reasoning text</reasoning>"

        # Act
        result = normalize_for_token_count(text)

        # Assert
        assert "<reasoning>" not in result
        assert "</reasoning>" not in result

    def test_RemovesAnalysisWrapperTag(self):
        # Arrange
        text = "<analysis>some analysis</analysis>"

        # Act
        result = normalize_for_token_count(text)

        # Assert
        assert "<analysis>" not in result
        assert "</analysis>" not in result

    def test_RemovesXmlWrapperTag_PreservesBodyText(self):
        # Arrange
        text = "<reasoning>preserved body text</reasoning>"

        # Act
        result = normalize_for_token_count(text)

        # Assert
        assert "preserved body text" in result, (
            "XML wrapper tag bodies must be preserved — only the tags are stripped"
        )

    # --- None / empty / whitespace input ---

    def test_NoneInput_ReturnsEmptyString(self):
        # Arrange
        text = None  # type: ignore[assignment]

        # Act
        result = normalize_for_token_count(text)

        # Assert
        assert result == "", (
            "normalize_for_token_count must return empty string for None input "
            "(matches strip_special_tags behavior for NaN/None)"
        )

    def test_EmptyString_ReturnsEmptyString(self):
        # Arrange
        text = ""

        # Act
        result = normalize_for_token_count(text)

        # Assert
        assert result == ""

    def test_PureBodyText_ReturnedUnchanged(self):
        # Arrange — no tags present, text should pass through unmodified
        text = "plain text with no special tags at all"

        # Act
        result = normalize_for_token_count(text)

        # Assert
        assert result == text, (
            "Text with no special tags must be returned unchanged"
        )

    # --- NaN input (pandas float NaN common in DataFrames) ---

    def test_FloatNanInput_ReturnsEmptyString(self):
        # Arrange
        import math
        text = float("nan")  # type: ignore[assignment]

        # Act
        result = normalize_for_token_count(text)

        # Assert
        assert result == "", (
            "normalize_for_token_count must return empty string for float NaN input, "
            "which is the common sentinel value in pandas DataFrames for missing cells"
        )

    # --- Return type ---

    def test_ReturnType_IsAlwaysStr(self):
        # Arrange
        text = "<think>body</think>"

        # Act
        result = normalize_for_token_count(text)

        # Assert
        assert isinstance(result, str), "normalize_for_token_count must always return str"


# ===========================================================================
# count_tokens
# ===========================================================================

class TestCountTokens:
    """
    count_tokens(text, enc) -> int

    Uses enc.encode_ordinary(text) to avoid special token injection.
    Returns 0 for empty or whitespace-only text.
    """

    # --- Empty and whitespace ---

    def test_EmptyString_ReturnsZero(self):
        # Arrange
        enc = _make_encoding()
        text = ""

        # Act
        result = count_tokens(text, enc)

        # Assert
        assert result == 0, (
            "count_tokens must return 0 for empty string"
        )

    def test_WhitespaceOnlyString_ReturnsZero(self):
        # Arrange — "   " encodes to 1 token via encode_ordinary but the
        # function contract says it returns 0 for whitespace-only text
        enc = _make_encoding()
        text = "   "

        # Act
        result = count_tokens(text, enc)

        # Assert
        assert result == 0, (
            "count_tokens must return 0 for whitespace-only text. "
            "Rationale: a whitespace-only cell represents an absent response, "
            "not meaningful content to count. The function should check "
            "text.strip() == '' before encoding."
        )

    def test_TabAndNewlineOnly_ReturnsZero(self):
        # Arrange
        enc = _make_encoding()
        text = "\t\n  \r\n"

        # Act
        result = count_tokens(text, enc)

        # Assert
        assert result == 0, (
            "count_tokens must return 0 for strings containing only whitespace "
            "characters (tabs, newlines, spaces)"
        )

    # --- Known token counts (cl100k_base, encode_ordinary) ---

    def test_HelloWorld_ReturnsTwoTokens(self):
        # Arrange — "hello world" tokenises as ["hello", " world"] with cl100k_base
        enc = _make_encoding()
        text = "hello world"

        # Act
        result = count_tokens(text, enc)

        # Assert
        assert result == 2, (
            f"'hello world' must tokenize to exactly 2 tokens with cl100k_base "
            f"encode_ordinary, got {result}"
        )

    def test_SingleWord_ReturnsOneToken(self):
        # Arrange — "hello" is one token
        enc = _make_encoding()
        text = "hello"

        # Act
        result = count_tokens(text, enc)

        # Assert
        assert result == 1, (
            f"'hello' must tokenize to exactly 1 token with cl100k_base, got {result}"
        )

    def test_FourWordPhrase_ReturnsFourTokens(self):
        # Arrange — "The quick brown fox" = 4 tokens with cl100k_base
        enc = _make_encoding()
        text = "The quick brown fox"

        # Act
        result = count_tokens(text, enc)

        # Assert
        assert result == 4, (
            f"'The quick brown fox' must tokenize to 4 tokens with cl100k_base, "
            f"got {result}"
        )

    # --- Return type ---

    def test_ReturnType_IsInt(self):
        # Arrange
        enc = _make_encoding()

        # Act
        result = count_tokens("hello", enc)

        # Assert
        assert isinstance(result, int), (
            "count_tokens must return an int, not float or other numeric type"
        )

    def test_ReturnType_IsIntForEmptyString(self):
        # Arrange
        enc = _make_encoding()

        # Act
        result = count_tokens("", enc)

        # Assert
        assert isinstance(result, int)
        assert result == 0

    # --- Non-negative ---

    def test_ResultIsNonNegative(self):
        # Arrange
        enc = _make_encoding()

        # Act — any non-empty text
        result = count_tokens("some text here", enc)

        # Assert
        assert result >= 0, "count_tokens must always return a non-negative integer"

    # --- encode_ordinary (no special tokens) ---

    def test_UsesEncodeOrdinary_NotEncode(self):
        # Arrange — verify that count_tokens does NOT inject special tokens
        # by checking that the token count for a simple string matches
        # encode_ordinary (which never adds BOS/EOS), not encode (which may).
        # With cl100k_base, encode_ordinary and encode happen to agree for most
        # text, but the contract is encode_ordinary.
        enc = _make_encoding()
        text = "foo bar"
        expected = len(enc.encode_ordinary(text))

        # Act
        result = count_tokens(text, enc)

        # Assert
        assert result == expected, (
            "count_tokens must use encode_ordinary(), not encode(). "
            f"encode_ordinary gives {expected} tokens for {repr(text)}, got {result}."
        )


# ===========================================================================
# get_encoding
# ===========================================================================

class TestGetEncoding:
    """
    get_encoding(name="cl100k_base") -> tiktoken.Encoding
    """

    def test_DefaultEncoding_ReturnsTiktokenEncoding(self):
        # Arrange
        import tiktoken

        # Act
        enc = get_encoding()

        # Assert
        assert isinstance(enc, tiktoken.Encoding), (
            "get_encoding() must return a tiktoken.Encoding instance"
        )

    def test_DefaultEncoding_IsCl100kBase(self):
        # Arrange / Act
        enc = get_encoding()

        # Assert
        assert enc.name == "cl100k_base", (
            f"Default encoding must be cl100k_base, got {enc.name!r}"
        )

    def test_ExplicitCl100kBase_ReturnsCl100kBaseEncoding(self):
        # Arrange / Act
        enc = get_encoding("cl100k_base")

        # Assert
        assert enc.name == "cl100k_base"

    def test_ReturnedEncoding_HasEncodeOrdinaryMethod(self):
        # Arrange / Act
        enc = get_encoding()

        # Assert
        assert hasattr(enc, "encode_ordinary"), (
            "The returned tiktoken.Encoding must have encode_ordinary() method. "
            "This is required for special-token-free token counting."
        )

    def test_ReturnedEncoding_EncodeOrdinaryIsCallable(self):
        # Arrange / Act
        enc = get_encoding()

        # Assert
        assert callable(enc.encode_ordinary), (
            "encode_ordinary must be callable on the returned encoding"
        )

    def test_ReturnedEncoding_ProducesExpectedTokenCountForKnownString(self):
        # Arrange — sanity-check the returned encoding actually works
        enc = get_encoding()

        # Act
        tokens = enc.encode_ordinary("hello world")

        # Assert
        assert len(tokens) == 2, (
            "get_encoding() must return a fully functional cl100k_base encoding. "
            f"'hello world' should give 2 tokens, got {len(tokens)}"
        )
