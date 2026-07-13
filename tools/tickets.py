"""
tools/tickets.py
A mock support-ticket system. When the agent escalates (low confidence,
failed groundedness, order not found), it shouldn't just print "I'll
connect you with a human" and forget the interaction ever happened -- a
real handoff means a human agent can actually pick up the case later. This
creates a persisted ticket record (order ID if known, the reason for
escalation, and the question that triggered it) and returns a ticket ID the
user can reference.

Persisted as local JSON for the same reason mock_orders.json is: this is
demonstrating the tool-use pattern without a fake external dependency. The
storage path is overridable via GIGACORP_TICKETS_PATH so tests/eval runs
never write into the real data file.
"""

import json
import os
import uuid
from datetime import datetime, timezone

_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "support_tickets.json")


def _tickets_path() -> str:
    return os.environ.get("GIGACORP_TICKETS_PATH", _DEFAULT_PATH)


def _load_all() -> dict:
    path = _tickets_path()
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save_all(tickets: dict) -> None:
    path = _tickets_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tickets, f, indent=2)


def create_ticket(reason: str, user_message: str, order_id: str | None = None) -> dict:
    """
    Creates and persists a support ticket, returns the ticket record
    (including its ticket_id) so the caller can surface it to the user.
    """
    ticket = {
        "ticket_id": f"TCK-{uuid.uuid4().hex[:8].upper()}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "user_message": user_message,
        "order_id": order_id,
        "status": "open",
    }
    tickets = _load_all()
    tickets[ticket["ticket_id"]] = ticket
    _save_all(tickets)
    return ticket
