"""
Router Node

This module contains the routing logic to determine which nodes to execute next.
"""

import logging
from typing import Literal

from ..state import AgentState

logger = logging.getLogger(__name__)


def route_query(state: AgentState) -> Literal["search_profiles", "search_events", "search_mixed"]:
    """
    Route query based on intent type and extracted information

    This router determines the search path based on what was extracted by the intent node:
    - profile: Only profile attributes were found -> search_profiles
    - event: Only events were found -> search_events
    - mixed: Both profile attributes and events were found -> search_mixed

    Args:
        state: Current agent state

    Returns:
        Routing target: "search_profiles", "search_events", or "search_mixed"
    """
    intent_type = state.get("intent_type", "profile")
    profile_attributes = state.get("profile_attributes", [])
    events = state.get("events", [])

    has_profiles = len(profile_attributes) > 0
    has_events = len(events) > 0

    # Determine routing based on what was extracted
    if has_profiles and has_events:
        route = "search_mixed"
    elif has_events:
        route = "search_events"
    else:
        route = "search_profiles"

    logger.info(
        f"[router] Routing to: {route} "
        f"(intent_type={intent_type}, profiles={len(profile_attributes)}, events={len(events)})"
    )

    return route
