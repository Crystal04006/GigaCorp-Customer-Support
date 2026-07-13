"""
graph.py
The agent's brain, built as a LangGraph state machine instead of a single
linear chain. This is the core architectural difference from a typical
"RAG chatbot" submission:

    guardrail -> classify_intent -> [order_status | returns_action | faq | chitchat]
                                          |              |            |
                                    order_lookup    order_lookup+   retrieve
                                          |          faq_retrieve       |
                                          +---------------+-------------+
                                                          |
                                                  confidence_gate --(low)--> escalate (+ticket) --> END
                                                          |(ok)
                                                   generate_answer <-------------------+
                                                          |                            | retry once
                                                  groundedness_check --(low, 1st time)--+
                                                          |
                                                   (low, 2nd time)
                                                          |
                                              low_groundedness (+ticket) --> END
                                                          |(ok)
                                                       finalize --> END

Why this shape, concretely:
  - "returns_action" is the one that actually needs to look like an agent:
    a question like "what's the return status of order #4471" requires
    BOTH a tool call (order lookup) AND a knowledge-base lookup (return
    policy), merged into one answer -- not just top-k similarity search.
  - confidence_gate uses the retriever's own similarity/lookup signal to
    refuse to answer rather than guess, before any generation happens.
  - groundedness_check runs *after* generation and checks TWO independent
    things: semantic drift (does the answer discuss what the context
    discusses) and numeric accuracy (does every number in the answer
    actually appear in the context). These catch different failure modes;
    see groundedness.py for why neither substitutes for the other.
  - A single retry is attempted before giving up: the model is told
    specifically what was unsupported and asked to correct or drop it,
    rather than immediately handing the customer a refusal. Capped at one
    retry to bound latency/cost.
  - Every escalation path (low confidence, order not found, groundedness
    still failing after retry) creates a persisted support ticket via
    tools/tickets.py, so a human agent has something real to pick up
    instead of the interaction just vanishing.
  - Two models are used, not one: classification and query-rewriting are
    cheap, high-volume, low-stakes tasks routed to `llm_fast`; final answer
    generation -- the thing the customer actually reads -- uses `llm_main`.
    Paying full price per token for an intent classification call is pure
    waste; this is the same tiering a real production agent would use.
"""

import json
import re
from typing import Optional, TypedDict

from tools.orders import extract_order_id, lookup_order, format_order_summary
from tools.tickets import create_ticket
from guardrails import detect_injection, SAFE_DEFLECTION_RESPONSE
from groundedness import evaluate_groundedness

try:
    from langgraph.graph import StateGraph, END
except ImportError as e:  # pragma: no cover
    raise ImportError("langgraph is required: pip install langgraph") from e


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    user_input: str
    recent_transcript: list
    memory_context: str

    is_injection: bool
    intent: str
    rewritten_query: str
    llm_order_id: Optional[str]
    llm_membership_tier: Optional[str]

    order_id: Optional[str]
    order_record: Optional[dict]
    retrieved_docs: list
    retrieval_confidence: float
    grounding_context: str

    escalate: bool
    escalate_reason: str
    ticket_id: Optional[str]

    draft_answer: str
    groundedness_score: float
    groundedness_detail: dict
    unsupported_numbers: list
    numeric_passed: bool
    groundedness_retry_count: int
    correction_note: str

    final_answer: str
    citations: list
    debug: dict


DEFAULT_CONFIDENCE_THRESHOLD = 0.28
DEFAULT_GROUNDEDNESS_THRESHOLD = 0.40
DEFAULT_DISTANCE_SCALE = 1.2  # tune against your embedding model's typical L2 distance range
MAX_GROUNDEDNESS_RETRIES = 1

ESCALATION_MESSAGE_TEMPLATE = (
    "I don't have confident information on that in our knowledge base right now, "
    "and I'd rather not guess and risk giving you the wrong answer. "
    "I've opened ticket {ticket_id} so a member of our support team can follow "
    "up -- you can reference that number if you reach out to "
    "support@gigacorp-example.com in the meantime."
)

LOW_GROUNDEDNESS_MESSAGE_TEMPLATE = (
    "I found some related information, but I'm not confident my answer "
    "precisely matches our official policy, so I don't want to risk telling "
    "you something inaccurate. I've opened ticket {ticket_id} so a team "
    "member can confirm the exact details for you."
)


def _robust_json_parse(text: str, fallback: dict) -> dict:
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return fallback


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------

def make_guardrail_node():
    def node(state: AgentState) -> dict:
        flagged, pattern = detect_injection(state["user_input"])
        debug = dict(state.get("debug", {}))
        if flagged:
            debug["guardrail_triggered_pattern"] = pattern
            return {
                "is_injection": True,
                "final_answer": SAFE_DEFLECTION_RESPONSE,
                "citations": [],
                "debug": debug,
            }
        return {"is_injection": False, "debug": debug}

    return node


def make_classify_node(llm_fast):
    """
    Uses the cheap/fast model. Besides intent + standalone-question
    rewriting, this call also extracts order_id and membership_tier
    directly via the model's language understanding -- reusing a call
    that already has to happen every turn, rather than adding a second
    LLM round-trip just for extraction. A regex fallback still exists in
    memory.py for offline/fake-backend runs where this call is stubbed.
    """

    def node(state: AgentState) -> dict:
        prompt = f"""TASK: CLASSIFY_INTENT

You are analyzing one message in an ongoing customer support conversation
for GigaCorp. Do three things:
1. Classify the customer's CURRENT message into exactly one intent.
2. Rewrite it as a standalone question that includes any context needed
   from the conversation so far (e.g. resolve "how much does it cost" into
   "how much does shipping to India cost" if that's what's being asked
   about based on prior turns).
3. Extract an order_id (digits only, no # or "order") and/or a
   membership_tier (Basic/Plus/Premier) IF the customer's message or the
   recent conversation clearly references one; otherwise null.

Intents:
- "order_status": asking about the status/location/delivery of a specific order
- "returns_action": asking about returning, refunding, or exchanging an order
- "faq": general policy questions (shipping, business hours, membership tiers, etc.)
- "chitchat": greetings, thanks, small talk with no support question

Session memory so far:
{state.get('memory_context', 'None')}

Recent conversation:
{state.get('recent_transcript', [])}

User message: {state['user_input']}

Respond with ONLY a JSON object, no other text:
{{"intent": "...", "rewritten_query": "...", "order_id": "... or null", "membership_tier": "... or null"}}"""
        raw = llm_fast.invoke(prompt).content
        parsed = _robust_json_parse(
            raw,
            fallback={"intent": "faq", "rewritten_query": state["user_input"], "order_id": None, "membership_tier": None},
        )
        intent = parsed.get("intent", "faq")
        if intent not in ("order_status", "returns_action", "faq", "chitchat"):
            intent = "faq"
        return {
            "intent": intent,
            "rewritten_query": parsed.get("rewritten_query") or state["user_input"],
            "llm_order_id": parsed.get("order_id") or None,
            "llm_membership_tier": parsed.get("membership_tier") or None,
        }

    return node


def route_by_intent(state: AgentState) -> str:
    return state.get("intent", "faq")


def route_after_guardrail(state: AgentState) -> str:
    return "end" if state.get("is_injection") else "classify"


def _resolve_order_id(state: AgentState) -> Optional[str]:
    """LLM-extracted order ID first (real language understanding), regex
    fallback second (offline/fake-backend runs, or if the LLM missed it)."""
    return (
        state.get("llm_order_id")
        or extract_order_id(state.get("rewritten_query", ""))
        or extract_order_id(state["user_input"])
    )


def make_order_lookup_node():
    def node(state: AgentState) -> dict:
        oid = _resolve_order_id(state)
        record = lookup_order(oid) if oid else None
        if record:
            return {
                "order_id": oid,
                "order_record": record,
                "grounding_context": "Order details:\n" + format_order_summary(record),
                "retrieval_confidence": 1.0,
            }
        return {
            "order_id": oid,
            "order_record": None,
            "grounding_context": "",
            "retrieval_confidence": 0.0,
            "escalate": True,
            "escalate_reason": "order_not_found",
        }

    return node


def _retrieve_faq(retriever, query: str, distance_scale: float):
    docs_and_scores = retriever.vectorstore.similarity_search_with_score(query, k=3)
    docs = [d for d, _ in docs_and_scores]
    distances = [s for _, s in docs_and_scores]
    min_dist = min(distances) if distances else 999.0
    confidence = max(0.0, min(1.0, 1.0 - (min_dist / distance_scale))) if distances else 0.0
    context = "\n\n".join(f"[{d.metadata.get('section', 'General')}] {d.page_content}" for d in docs)
    return docs, confidence, context


def make_retrieve_node(retriever, distance_scale: float = DEFAULT_DISTANCE_SCALE):
    def node(state: AgentState) -> dict:
        docs, confidence, context = _retrieve_faq(retriever, state.get("rewritten_query", state["user_input"]), distance_scale)
        return {"retrieved_docs": docs, "retrieval_confidence": confidence, "grounding_context": context}

    return node


def make_returns_gather_node(retriever, distance_scale: float = DEFAULT_DISTANCE_SCALE):
    """
    The 'agent, not just retriever' node: merges a live order lookup (tool
    call) with a policy retrieval (RAG) into one grounding context, so the
    generation step can synthesize a single answer that draws on both.
    """

    def node(state: AgentState) -> dict:
        query = state.get("rewritten_query", state["user_input"])
        oid = _resolve_order_id(state)
        record = lookup_order(oid) if oid else None

        docs, retrieval_confidence, faq_context = _retrieve_faq(retriever, query, distance_scale)

        parts = []
        if record:
            parts.append("Order details:\n" + format_order_summary(record))
        parts.append("Return/refund policy:\n" + faq_context)

        confidence = retrieval_confidence
        if oid and not record:
            confidence = min(confidence, 0.15)

        return {
            "order_id": oid,
            "order_record": record,
            "retrieved_docs": docs,
            "retrieval_confidence": confidence,
            "grounding_context": "\n\n".join(parts),
        }

    return node


def confidence_gate_node(state: AgentState) -> dict:
    threshold = state.get("_confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
    if state.get("escalate"):
        return {}
    if state.get("retrieval_confidence", 0.0) < threshold:
        return {"escalate": True, "escalate_reason": "low_retrieval_confidence"}
    return {"escalate": False}


def route_after_confidence(state: AgentState) -> str:
    return "escalate" if state.get("escalate") else "generate"


def escalate_response_node(state: AgentState) -> dict:
    reason = state.get("escalate_reason") or "unknown"
    ticket = create_ticket(reason=reason, user_message=state["user_input"], order_id=state.get("order_id"))
    debug = dict(state.get("debug", {}))
    debug["escalated"] = True
    debug["escalate_reason"] = reason
    return {
        "final_answer": ESCALATION_MESSAGE_TEMPLATE.format(ticket_id=ticket["ticket_id"]),
        "citations": [],
        "ticket_id": ticket["ticket_id"],
        "debug": debug,
    }


def make_generate_answer_node(llm_main):
    def node(state: AgentState) -> dict:
        correction_note = state.get("correction_note", "")
        correction_block = (
            f"\nIMPORTANT CORRECTION NEEDED: your previous answer had a problem: "
            f"{correction_note}\nProduce a corrected answer using ONLY numbers and "
            f"claims that appear in the Context below. If you're unsure of an exact "
            f"figure, omit it rather than guess.\n"
            if correction_note else ""
        )
        prompt = f"""TASK: ANSWER_GENERATION

You are the GigaCorp Customer Support Assistant. Answer using ONLY the
information in the Context below. Never invent policies, prices, dates, or
order details that are not present in the Context. Be concise and warm.
{correction_block}
Session memory:
{state.get('memory_context', 'None')}

Context:
[Company Profile] GigaCorp is a premium, high-performance developer hardware enterprise. We specialize in designing next-generation mechanical keyboards, hot-swappable desk peripherals, and ergonomic productivity workstations tailored specifically for software engineers, data professionals, and elite creators. Our main goal is to engineer high-tactility, hyper-durable infrastructure that optimizes typing speeds, ergonomics, and daily workflow efficiencies.

{state.get('grounding_context', '')}

Question: {state.get('rewritten_query', state['user_input'])}


Answer:"""
        answer = llm_main.invoke(prompt).content
        return {"draft_answer": answer}

    return node


def make_groundedness_node(embeddings):
    def node(state: AgentState) -> dict:
        result = evaluate_groundedness(state.get("draft_answer", ""), state.get("grounding_context", ""), embeddings)
        return {
            "groundedness_score": result["score"],
            "groundedness_detail": result,
            "unsupported_numbers": result["unsupported_numbers"],
            "numeric_passed": result["numeric_passed"],
        }

    return node


def route_after_groundedness(state: AgentState) -> str:
    threshold = state.get("_groundedness_threshold", DEFAULT_GROUNDEDNESS_THRESHOLD)
    semantic_ok = state.get("groundedness_score", 1.0) >= threshold
    numeric_ok = state.get("numeric_passed", True)
    if semantic_ok and numeric_ok:
        return "ok"
    if state.get("groundedness_retry_count", 0) < MAX_GROUNDEDNESS_RETRIES:
        return "retry"
    return "low"


def prepare_retry_node(state: AgentState) -> dict:
    """
    Builds a specific correction instruction for the one allowed retry,
    rather than a generic "try again" -- naming exactly what was wrong
    gives the model a real shot at fixing it instead of repeating the same
    mistake.
    """
    notes = []
    if not state.get("numeric_passed", True):
        nums = ", ".join(state.get("unsupported_numbers", []))
        notes.append(f"these numbers are not supported by the context and must be removed or corrected: {nums}")
    detail = state.get("groundedness_detail", {})
    if state.get("groundedness_score", 1.0) < state.get("_groundedness_threshold", DEFAULT_GROUNDEDNESS_THRESHOLD):
        worst = detail.get("worst_sentence")
        if worst:
            notes.append(f'this claim does not appear to be supported by the context: "{worst}"')
    return {
        "correction_note": "; ".join(notes) or "the previous answer was not well-grounded in the provided context",
        "groundedness_retry_count": state.get("groundedness_retry_count", 0) + 1,
    }


def low_groundedness_node(state: AgentState) -> dict:
    ticket = create_ticket(reason="low_groundedness", user_message=state["user_input"], order_id=state.get("order_id"))
    debug = dict(state.get("debug", {}))
    debug["groundedness_rejected"] = True
    debug["worst_sentence"] = state.get("groundedness_detail", {}).get("worst_sentence")
    debug["unsupported_numbers"] = state.get("unsupported_numbers", [])
    return {
        "final_answer": LOW_GROUNDEDNESS_MESSAGE_TEMPLATE.format(ticket_id=ticket["ticket_id"]),
        "citations": [],
        "escalate": True,
        "escalate_reason": "low_groundedness",
        "ticket_id": ticket["ticket_id"],
        "debug": debug,
    }


def _build_citations(state: AgentState) -> list:
    citations = []
    for doc in state.get("retrieved_docs", []) or []:
        meta = doc.metadata
        citations.append(
            f"{meta.get('source', 'unknown')}, lines {meta.get('start_line', '?')}-{meta.get('end_line', '?')} "
            f"(Section: {meta.get('section', 'General')})"
        )
    if state.get("order_id") and state.get("order_record"):
        citations.append(f"mock_orders.json, order #{state['order_id']}")
    seen = set()
    unique = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def finalize_answer_node(state: AgentState) -> dict:
    return {"final_answer": state.get("draft_answer", ""), "citations": _build_citations(state)}


def make_chitchat_node(llm_fast):
    def node(state: AgentState) -> dict:
        prompt = f"""TASK: CHITCHAT

You are the GigaCorp Customer Support Assistant. Respond briefly and
warmly to this small-talk message, then invite the customer to ask about
shipping, returns, business hours, order status, or membership tiers.

Message: {state['user_input']}"""
        answer = llm_fast.invoke(prompt).content
        return {"final_answer": answer, "citations": [], "groundedness_score": 1.0}

    return node


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(
    llm_main,
    retriever,
    embeddings,
    llm_fast=None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    groundedness_threshold: float = DEFAULT_GROUNDEDNESS_THRESHOLD,
    distance_scale: float = DEFAULT_DISTANCE_SCALE,
):
    """
    llm_fast defaults to llm_main if not provided, so existing callers that
    only pass one model still work -- but app.py always supplies both, with
    llm_fast pointed at a cheaper/faster model for classification, query
    rewriting, and chitchat.
    """
    llm_fast = llm_fast or llm_main

    graph = StateGraph(AgentState)

    graph.add_node("guardrail", make_guardrail_node())
    graph.add_node("classify", make_classify_node(llm_fast))
    graph.add_node("order_lookup", make_order_lookup_node())
    graph.add_node("returns_gather", make_returns_gather_node(retriever, distance_scale))
    graph.add_node("retrieve", make_retrieve_node(retriever, distance_scale))
    graph.add_node(
        "confidence_gate",
        lambda s: confidence_gate_node({**s, "_confidence_threshold": confidence_threshold}),
    )
    graph.add_node("escalate_response", escalate_response_node)
    graph.add_node("generate_answer", make_generate_answer_node(llm_main))
    graph.add_node("groundedness_check", make_groundedness_node(embeddings))
    graph.add_node("prepare_retry", prepare_retry_node)
    graph.add_node(
        "low_groundedness",
        lambda s: low_groundedness_node({**s, "_groundedness_threshold": groundedness_threshold}),
    )
    graph.add_node("finalize", finalize_answer_node)
    graph.add_node("chitchat", make_chitchat_node(llm_fast))

    graph.set_entry_point("guardrail")
    graph.add_conditional_edges("guardrail", route_after_guardrail, {"end": END, "classify": "classify"})
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "order_status": "order_lookup",
            "returns_action": "returns_gather",
            "faq": "retrieve",
            "chitchat": "chitchat",
        },
    )
    graph.add_edge("order_lookup", "confidence_gate")
    graph.add_edge("returns_gather", "confidence_gate")
    graph.add_edge("retrieve", "confidence_gate")
    graph.add_conditional_edges(
        "confidence_gate",
        route_after_confidence,
        {"escalate": "escalate_response", "generate": "generate_answer"},
    )
    graph.add_edge("escalate_response", END)
    graph.add_edge("generate_answer", "groundedness_check")
    graph.add_conditional_edges(
        "groundedness_check",
        lambda s: route_after_groundedness({**s, "_groundedness_threshold": groundedness_threshold}),
        {"ok": "finalize", "retry": "prepare_retry", "low": "low_groundedness"},
    )
    graph.add_edge("prepare_retry", "generate_answer")
    graph.add_edge("low_groundedness", END)
    graph.add_edge("finalize", END)
    graph.add_edge("chitchat", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience wrapper used by app.py and the eval harness
# ---------------------------------------------------------------------------

def run_turn(compiled_graph, session_memory, user_input: str, recent_transcript: list) -> dict:
    initial_state: AgentState = {
        "user_input": user_input,
        "recent_transcript": recent_transcript,
        "memory_context": session_memory.as_context_string(),
        "groundedness_retry_count": 0,
        "correction_note": "",
        "debug": {},
    }
    result = compiled_graph.invoke(initial_state)
    session_memory.update_from_turn(
        user_text=user_input,
        detected_intent=result.get("intent", "chitchat"),
        order_record=result.get("order_record"),
        llm_order_id=result.get("llm_order_id"),
        llm_membership_tier=result.get("llm_membership_tier"),
    )
    return result
