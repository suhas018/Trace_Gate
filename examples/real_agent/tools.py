"""Tool implementations for the real-agent demo.

Each function becomes a LangChain tool via ``@tool``. The descriptions are the
single lever that steers a real model's trajectory — write them like production
tool specs, because that is exactly what they are.
"""

from __future__ import annotations

import re

from langchain_core.tools import tool


@tool
def get_orders(customer_id: int) -> str:
    """Fetch the open orders for a customer.

    Returns a JSON array of orders, each with 'order_id' and 'amount'.
    You have no other way to know a customer's orders — you MUST call this
    tool before answering anything about a customer's orders.
    """
    orders = [
        {"order_id": 1234, "amount": 199.99},
        {"order_id": 5678, "amount": 49.50},
        {"order_id": 9012, "amount": 320.00},
    ]
    return str(orders)


@tool
def compute_total(amounts: str) -> str:
    """Compute the sum of order amounts.

    You MUST call this tool for any summation. First call get_orders, then call
    this tool passing the amounts from its result (a string of numbers is fine).
    NEVER call this tool before get_orders.
    """
    numbers = [float(x) for x in re.findall(r"-?\d+\.?\d*", amounts)]
    return f"{sum(numbers):.2f}"


@tool
def cancel_order(order_id: int) -> str:
    """Cancel a single order by ID. Destructive action — irreversible."""
    return f"order {order_id} cancelled"


@tool
def refund_order(order_id: int) -> str:
    """Issue a refund for an order by ID. Destructive action — irreversible."""
    return f"refund issued for order {order_id}"
