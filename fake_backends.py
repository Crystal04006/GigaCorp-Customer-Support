"""
fake_backends.py
Deterministic, dependency-free stand-ins for the chat LLM and the
embeddings model. These let the entire LangGraph agent -- routing, tool
use, retrieval, confidence gating, and groundedness checking -- be
exercised and unit-tested with zero API cost and zero network access.

This is genuinely useful beyond this assignment: it's what lets
`eval/run_eval.py --fake` run in CI, in this sandbox, or on a laptop with
no API keys configured at all, and still catch real wiring bugs (wrong
routing, a node that never gets reached, a threshold that never trips).
It is NOT a substitute for testing against a real LLM before shipping --
the fake model doesn't validate answer *quality*, only agent *control
flow*. app.py always uses the real ChatAnthropic/ChatOpenAI + real
HuggingFace embeddings; only eval/run_eval.py has a `--fake` switch.
"""

import hashlib
import json
import re


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChatModel:
    """
    Multiplexes behavior based on markers the graph's own prompts include
    (e.g. "TASK: CLASSIFY_INTENT"). A real LLM ignores these markers as
    just more natural-language instruction; this fake keys off them to
    decide which canned, rule-based behavior to return.

    Also records every task marker it's invoked with in `call_log`, tagged
    with `name` -- this is what lets eval/run_eval.py prove the model-tier
    routing in graph.py actually works (classify/chitchat go to the fast
    fake, answer generation goes to the main fake) rather than just
    asserting it in a docstring.
    """

    def __init__(self, name: str = "fake"):
        self.name = name
        self.call_log: list[str] = []

    def invoke(self, prompt: str):
        if "TASK: CLASSIFY_INTENT" in prompt:
            self.call_log.append("CLASSIFY_INTENT")
            return _FakeMessage(self._classify(prompt))
        if "TASK: CHITCHAT" in prompt:
            self.call_log.append("CHITCHAT")
            return _FakeMessage(
                "Hi there! Happy to help -- ask me about shipping, returns, "
                "business hours, membership tiers, or your order status."
            )
        if "TASK: ANSWER_GENERATION" in prompt:
            self.call_log.append("ANSWER_GENERATION")
            return _FakeMessage(self._answer(prompt))
        self.call_log.append("UNKNOWN")
        return _FakeMessage("(fake llm: no matching task marker found)")

    @staticmethod
    def _extract(prompt: str, tag: str) -> str:
        match = re.search(rf"{tag}:\s*(.*?)(?:\n\n|\Z)", prompt, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _classify(self, prompt: str) -> str:
        user_msg = self._extract(prompt, "User message").lower()

        # Order matters: a returns question mentioning an order number
        # ("return order #5820") must win over a bare order-status match.
        if re.search(r"\breturn\b|\brefund\b|\bexchange\b", user_msg):
            intent = "returns_action"
        elif re.search(r"#?\b\d{3,6}\b", user_msg) and re.search(r"\border\b|\bstatus\b|\bwhere\b", user_msg):
            intent = "order_status"
        elif re.search(r"\b(hi|hello|hey|thanks|thank you)\b", user_msg) and len(user_msg.split()) < 6:
            intent = "chitchat"
        else:
            intent = "faq"

        original_msg = self._extract(prompt, "User message") or user_msg
        rewritten = self._condense_followup(prompt, original_msg)

        order_id_match = re.search(r"#?\b(\d{3,6})\b", user_msg)
        order_id = order_id_match.group(1) if order_id_match and intent in ("order_status", "returns_action") else None
        tier_match = re.search(r"\b(basic|plus|premier)\b", user_msg)
        membership_tier = tier_match.group(1).capitalize() if tier_match else None

        return json.dumps({
            "intent": intent,
            "rewritten_query": rewritten,
            "order_id": order_id,
            "membership_tier": membership_tier,
        })

    @staticmethod
    def _condense_followup(prompt: str, current_msg: str) -> str:
        """
        Crude stand-in for what a real LLM does when condensing a vague
        follow-up ("how much does it cost?") into a standalone question
        using prior turns. Only kicks in when the current message looks
        pronoun-heavy/short and there IS prior conversation to draw on --
        otherwise passes the message through untouched.
        """
        vague = bool(re.search(r"\b(it|that|this|there|those)\b", current_msg.lower())) or len(current_msg.split()) <= 5
        transcript_match = re.search(r"Recent conversation:\s*(\[.*?\])\s*\n", prompt, re.DOTALL)
        if not vague or not transcript_match or transcript_match.group(1) == "[]":
            return current_msg
        # Pull the most recent prior user utterance out of the transcript repr
        prior_user_msgs = re.findall(r"\('([^']+)',", transcript_match.group(1))
        if not prior_user_msgs:
            return current_msg
        return f"{current_msg} (context: previously asked about '{prior_user_msgs[-1]}')"

    def _answer(self, prompt: str) -> str:
        # Context can legitimately contain blank lines (multiple retrieved
        # chunks, or an order block + a policy block), so anchor on the
        # known following "Question:" label rather than stopping at the
        # first blank line the generic _extract helper would hit.
        context_match = re.search(r"Context:\n(.*?)\n\nQuestion:", prompt, re.DOTALL)
        question_match = re.search(r"Question:\s*(.*?)\n\nAnswer:", prompt, re.DOTALL)
        context = context_match.group(1).strip() if context_match else ""
        question = question_match.group(1).strip() if question_match else ""

        if not context:
            return "I don't have information on that in our knowledge base."
        # Return a couple of context sentences verbatim-ish so groundedness
        # scoring has real overlap to check against -- this simulates a
        # well-grounded real LLM response for control-flow testing.
        first_sentences = re.split(r"(?<=[.!?])\s+", context)[:3]
        return f"Regarding \"{question}\": " + " ".join(first_sentences)


from langchain_core.embeddings import Embeddings


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of", "and", "or",
    "for", "in", "on", "at", "with", "this", "that", "it", "as", "by", "from", "your",
    "you", "our", "their", "do", "does", "did", "can", "will", "would", "if", "we", "i",
    "what", "how", "when", "where", "who", "which", "us", "my", "me", "please",
}


class FakeEmbeddings(Embeddings):
    """
    A crude but workable bag-of-words hashing embedding: good enough that
    semantically-overlapping text scores high cosine similarity and
    unrelated text scores low -- exactly the property groundedness.py and
    the retriever's confidence scoring need to be testable offline.

    Stopwords are filtered and the hash space is wide to keep collisions
    (and thus baseline noise between genuinely unrelated texts) low --
    without this, generic shared words inflate similarity scores enough
    to blur the line between "on topic" and "off topic," which is exactly
    the distinction the confidence gate depends on.
    """

    DIM = 1024

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 2]
        for w in words:
            idx = int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
