"""
tools/orders.py
A mock "order management system" lookup tool. In a real deployment this
would call an internal API (Shopify, an ERP, etc.) — here it reads from a
local JSON file so the assignment's RAG focus isn't muddied by a fake
external dependency, while still demonstrating tool-use / function-calling
style agent behavior rather than pure retrieval.
"""

import json
import os
import re

ORDERS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mock_orders.json")

_ORDER_ID_PATTERN = re.compile(r"#?\b(\d{3,6})\b")


def extract_order_id(text: str) -> str | None:
    """
    Pulls a plausible order ID out of free text, e.g. "where's order #4471"
    or "check on 4471 please". Returns the first 3-6 digit number found.
    """
    match = _ORDER_ID_PATTERN.search(text)
    return match.group(1) if match else None


def lookup_order(order_id: str) -> dict | None:
    """Looks up an order by ID in the mock orders database."""
    if not os.path.isfile(ORDERS_PATH):
        return None
    with open(ORDERS_PATH, "r", encoding="utf-8") as f:
        orders = json.load(f)
    return orders.get(order_id)


def format_order_summary(order: dict) -> str:
    """Human-readable summary of an order record, for the agent to quote from."""
    return (
        f"Order #{order['order_id']} for {order['customer_name']}: "
        f"status is '{order['status']}'. "
        f"Items: {', '.join(order['items'])}. "
        f"Shipping to {order['destination_country']} via {order['shipping_method']}. "
        f"Ordered on {order['order_date']}, estimated delivery {order['estimated_delivery']}. "
        f"Membership tier: {order['membership_tier']}."
    )
