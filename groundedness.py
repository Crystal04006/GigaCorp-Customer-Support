"""
groundedness.py
Post-generation hallucination check.

The standard failure mode of RAG systems isn't "no citation" -- it's a
*confident-looking* answer with a citation slapped on that doesn't actually
say what the answer claims. Most demo RAG apps never check this; they trust
whatever the LLM produced.

Rather than spending a second, expensive LLM call to fact-check the first
one (which itself can hallucinate a lenient judgment), this does a cheap,
deterministic semantic-similarity check: split the generated answer into
sentences, embed each one, and compare it against the retrieved/grounding
context using the same embedding model already loaded for retrieval. If a
sentence has nothing similar in the source context, it's likely invented --
flag the whole answer as weakly grounded rather than let it through.

This is a heuristic, not a proof -- it catches topic drift (a sentence
about something the context never discussed) but NOT a wrong number inside
an otherwise on-topic sentence: "shipping to India costs $99.99" scores
just as well semantically as the correct "$14.99," because both sentences
are equally *about* shipping cost to India. That failure mode is the most
damaging one for a support agent specifically -- a wrong price is worse
than a vague non-answer -- so `check_numeric_grounding` below handles it
with a separate, deterministic check: every number that appears in the
answer must also appear somewhere in the retrieved context. It won't catch
a fabricated number that happens to coincidentally match a different digit
also present in the context, but it reliably catches invented figures,
which is the common case.
"""

import re
from math import sqrt

# Common framing preambles ("Regarding \"...\":", "Sure, happy to help!") echo
# the question or add pleasantries rather than make a factual claim. Left in,
# a naive punctuation-based sentence splitter can carve one out as its own
# "substantive" (>=6 word) sentence -- and since it's about the *question*,
# not the *context*, it legitimately scores near-zero similarity, unfairly
# dragging down the groundedness score of an otherwise well-grounded answer.
# Real LLMs use this exact framing style too, so this isn't just a fake-model
# artifact -- it's worth stripping before scoring either way.
_FRAMING_PREAMBLE = re.compile(r'^\s*(Regarding|Re|In response to)\s+".*?"\s*:\s*', re.IGNORECASE | re.DOTALL)
_FRAMING_FILLER = re.compile(r'^\s*(Sure|Thanks for asking|Happy to help)[,!.]?\s*', re.IGNORECASE)

# Matches plain numbers, currency amounts, percentages, and simple dates.
# Deliberately just the numeric token itself (not the currency symbol) so
# "$14.99" in the answer matches "14.99" appearing anywhere in the context,
# including inside a differently-formatted mention.
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")

# Numbers this common are near-meaningless as a hallucination signal on
# their own (order counts, list positions, single-digit day ranges that
# appear throughout the FAQ) -- excluding them cuts false positives without
# meaningfully weakening the check, since a fabricated price/date is almost
# never expressible as a single common small integer alone.
_IGNORED_NUMBERS = {"1", "2"}


def _strip_framing(text: str) -> str:
    text = _FRAMING_PREAMBLE.sub("", text, count=1)
    text = _FRAMING_FILLER.sub("", text, count=1)
    return text


def _sentences(text: str) -> list[str]:
    # Simple, dependency-free sentence splitter -- good enough for short
    # support-agent answers.
    text = _strip_framing(text)
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 3]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a)) or 1e-8
    norm_b = sqrt(sum(y * y for y in b)) or 1e-8
    return dot / (norm_a * norm_b)


def score_groundedness(answer: str, context: str, embeddings) -> dict:
    """
    Returns {"score": float in [0,1], "worst_sentence": str|None,
    "sentence_scores": list[(sentence, score)]}.
    `score` is the *minimum* per-sentence similarity to the context, since a
    single fabricated sentence is enough to make an answer misleading --
    averaging would let one bad sentence hide behind several good ones.
    """
    if not context.strip() or not answer.strip():
        return {"score": 1.0, "worst_sentence": None, "sentence_scores": []}

    sentences = _sentences(answer)
    if not sentences:
        return {"score": 1.0, "worst_sentence": None, "sentence_scores": []}

    context_vec = embeddings.embed_query(context[:4000])  # cap for speed
    sentence_vecs = embeddings.embed_documents(sentences)

    scores = [_cosine(vec, context_vec) for vec in sentence_vecs]
    pairs = list(zip(sentences, scores))

    # Short framing/filler sentences ("Sure, happy to help!", "Regarding
    # your question:") carry no factual claim and will legitimately have
    # near-zero overlap with the source context even in a perfectly
    # well-grounded answer. Scoring the *substantive* sentences (>= 6 words)
    # by their worst case is a much more honest hallucination check than
    # letting a pleasantry tank the whole score. If everything is short,
    # fall back to scoring everything.
    substantive = [(s, sc) for s, sc in pairs if len(s.split()) >= 6]
    scored_pairs = substantive or pairs

    min_score = min(sc for _, sc in scored_pairs)
    worst = next(s for s, sc in scored_pairs if sc == min_score)

    return {"score": min_score, "worst_sentence": worst, "sentence_scores": pairs}


def check_numeric_grounding(answer: str, context: str) -> dict:
    """
    Deterministic complement to score_groundedness: every number in the
    answer (price, day-count, percentage, date fragment) must appear
    somewhere in the retrieved context. Catches the "right topic, wrong
    number" failure mode that semantic similarity structurally can't see.

    Returns {"unsupported_numbers": [...], "passed": bool}.
    """
    answer_numbers = set(_NUMBER_PATTERN.findall(_strip_framing(answer))) - _IGNORED_NUMBERS
    context_numbers = set(_NUMBER_PATTERN.findall(context))
    unsupported = sorted(answer_numbers - context_numbers, key=lambda n: (len(n), n))
    return {"unsupported_numbers": unsupported, "passed": len(unsupported) == 0}


def evaluate_groundedness(answer: str, context: str, embeddings) -> dict:
    """
    Combined gate used by the graph: an answer must pass BOTH the semantic
    similarity check and the numeric-claim check to be considered grounded.
    Either one failing is enough to reject the answer -- they catch
    different failure modes (topic drift vs. a fabricated figure) and
    neither substitutes for the other.
    """
    semantic = score_groundedness(answer, context, embeddings)
    numeric = check_numeric_grounding(answer, context)
    return {
        "score": semantic["score"],
        "worst_sentence": semantic["worst_sentence"],
        "sentence_scores": semantic["sentence_scores"],
        "unsupported_numbers": numeric["unsupported_numbers"],
        "numeric_passed": numeric["passed"],
    }
