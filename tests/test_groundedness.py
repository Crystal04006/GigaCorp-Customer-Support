"""
tests/test_groundedness.py
Plain-assertion unit tests (no pytest dependency needed) for the numeric
groundedness check specifically -- the piece added to catch the "right
topic, wrong number" hallucination that semantic similarity structurally
cannot see (a wrong price is still a sentence "about" shipping cost).

Run: python tests/test_groundedness.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groundedness import check_numeric_grounding, evaluate_groundedness
from fake_backends import FakeEmbeddings

CONTEXT = (
    "[Shipping Policies] Standard International Shipping to India costs a "
    "flat rate of $14.99 per order. Express International Shipping to "
    "India costs $34.99 per order. Orders over $75 qualify for free "
    "Standard International Shipping to any supported country."
)


def test_correct_number_passes():
    answer = "Standard International Shipping to India costs $14.99 per order."
    result = check_numeric_grounding(answer, CONTEXT)
    assert result["passed"] is True, f"Expected pass, got {result}"
    assert result["unsupported_numbers"] == []


def test_hallucinated_number_fails():
    # $99.99 is nowhere in the context -- this is exactly the failure mode
    # semantic similarity would miss, since the sentence is still entirely
    # on-topic (shipping cost to India).
    answer = "Standard International Shipping to India costs $99.99 per order."
    result = check_numeric_grounding(answer, CONTEXT)
    assert result["passed"] is False, f"Expected fail, got {result}"
    assert "99.99" in result["unsupported_numbers"]


def test_semantic_check_alone_would_miss_it():
    # Prove the claim: semantic similarity scores the hallucinated-number
    # sentence just as well as the correct one, because both are equally
    # "about" the same topic. This is why the numeric check must exist
    # as a separate gate, not a replacement for the semantic one.
    from groundedness import score_groundedness

    embeddings = FakeEmbeddings()
    correct = "Standard International Shipping to India costs $14.99 per order."
    wrong = "Standard International Shipping to India costs $99.99 per order."

    correct_score = score_groundedness(correct, CONTEXT, embeddings)["score"]
    wrong_score = score_groundedness(wrong, CONTEXT, embeddings)["score"]

    # The two scores should be close (within a small tolerance) -- semantic
    # similarity genuinely cannot distinguish these, which is the point.
    assert abs(correct_score - wrong_score) < 0.05, (
        f"Expected semantic scores to be near-identical (proving semantic "
        f"similarity can't catch this), got correct={correct_score:.3f} "
        f"vs wrong={wrong_score:.3f}"
    )


def test_evaluate_groundedness_combines_both_gates():
    embeddings = FakeEmbeddings()
    wrong = "Standard International Shipping to India costs $99.99 per order."
    result = evaluate_groundedness(wrong, CONTEXT, embeddings)
    # Semantic score is fine (on-topic)...
    assert result["score"] > 0.3, f"Expected reasonable semantic score, got {result['score']}"
    # ...but the numeric gate must still catch it.
    assert result["numeric_passed"] is False
    assert "99.99" in result["unsupported_numbers"]


def test_ignored_small_numbers_dont_false_positive():
    # "1" and "2" are common enough (list positions, single-item counts)
    # that flagging them would create noisy false positives.
    answer = "You have 1 item in your order and 2 options for shipping."
    result = check_numeric_grounding(answer, CONTEXT)
    assert result["passed"] is True, f"Expected pass (ignored numbers), got {result}"


def test_empty_context_or_answer_is_vacuously_grounded():
    assert check_numeric_grounding("", CONTEXT)["passed"] is True
    assert check_numeric_grounding("Some answer with no numbers at all.", CONTEXT)["passed"] is True


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed.")
    sys.exit(0 if failed == 0 else 1)
