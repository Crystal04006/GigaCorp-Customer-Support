"""
memory.py
Structured "slot" memory, as opposed to just replaying the raw transcript.

The common approach (what most RAG chatbot demos do) is to shove the whole
conversation history into the prompt every turn so the model can "remember."
That works for short demos but degrades as conversations grow: it's
expensive, it dilutes the model's attention with irrelevant turns, and it
doesn't survive a topic switch and switch-back well.

Here, alongside a short rolling transcript (for natural back-and-forth
phrasing), we maintain a small set of durable structured facts extracted
from the conversation -- the things a real support agent would jot down on
a notepad: which order the customer is asking about, their membership
tier, and what topics have already come up. These slots are cheap to keep
around for the entire session and get injected into prompts directly,
rather than relying on the LLM to re-derive them from a growing transcript.
"""

import re
from dataclasses import dataclass, field


_TIER_PATTERN = re.compile(r"\b(basic|plus|premier)\b", re.IGNORECASE)
_ORDER_PATTERN = re.compile(r"#?\b(\d{3,6})\b")


@dataclass
class SessionMemory:
    last_order_id: str | None = None
    membership_tier: str | None = None
    topics_discussed: list = field(default_factory=list)
    turn_count: int = 0

    def update_from_turn(
        self,
        user_text: str,
        detected_intent: str,
        order_record: dict | None = None,
        llm_order_id: str | None = None,
        llm_membership_tier: str | None = None,
    ):
        """
        Priority order for each slot, highest first:
          1. A value CONFIRMED by an actual DB lookup this turn (order_record) --
             this is ground truth, not an inference.
          2. A value the classify_intent LLM call already extracted from the
             message using real language understanding (llm_order_id /
             llm_membership_tier) -- this is what lets "check on the one I
             ordered last week" resolve correctly, which a digit-matching
             regex fundamentally cannot do.
          3. A regex fallback, for when no LLM extraction was available
             (e.g. an offline/fake-backend run) -- catches the common
             explicit case ("order #4471") but can misfire on any bare
             3-6 digit number in a sentence ("I've ordered 4 times this
             year"), which is exactly why it's the last resort, not the
             primary mechanism.
        """
        self.turn_count += 1

        if order_record:
            self.last_order_id = order_record.get("order_id")
            if order_record.get("membership_tier"):
                self.membership_tier = order_record["membership_tier"]
        elif llm_order_id:
            self.last_order_id = llm_order_id
        else:
            match = _ORDER_PATTERN.search(user_text)
            if match:
                self.last_order_id = match.group(1)

        if llm_membership_tier:
            self.membership_tier = llm_membership_tier
        else:
            tier_match = _TIER_PATTERN.search(user_text)
            if tier_match:
                self.membership_tier = tier_match.group(1).capitalize()

        if detected_intent and detected_intent not in self.topics_discussed:
            self.topics_discussed.append(detected_intent)

    def as_context_string(self) -> str:
        """Compact summary injected into prompts, instead of a full transcript."""
        parts = []
        if self.last_order_id:
            parts.append(f"Customer's most recently referenced order ID: {self.last_order_id}")
        if self.membership_tier:
            parts.append(f"Customer's membership tier: {self.membership_tier}")
        if self.topics_discussed:
            parts.append(f"Topics already discussed this session: {', '.join(self.topics_discussed)}")
        return "\n".join(parts) if parts else "No prior context yet."

    def to_dict(self) -> dict:
        return {
            "last_order_id": self.last_order_id,
            "membership_tier": self.membership_tier,
            "topics_discussed": list(self.topics_discussed),
            "turn_count": self.turn_count,
        }
