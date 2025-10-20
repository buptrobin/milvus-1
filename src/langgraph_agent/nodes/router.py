"""
Router Node

This module contains the routing logic to determine which nodes to execute next.
"""

import logging
from typing import Literal

from ..state import AgentState

logger = logging.getLogger(__name__)


def route_query(state: AgentState) -> Literal["search_profiles"]:
    """
    Route query based on intent type and extracted information

    Args:
        state: Current agent state

    Returns:
        Routing target: "search_profiles"
    """
    # Always route to profile search since we only support profiles now
    logger.info("[router] Routing to: search_profiles")
    return "search_profiles"
