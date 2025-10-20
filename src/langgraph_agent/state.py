"""
LangGraph Agent State Schema

This module defines the state structure used throughout the LangGraph workflow.
"""

from typing import TypedDict, List, Dict, Optional, Annotated
from operator import add


class AgentState(TypedDict):
    """
    LangGraph agent state schema

    This state object is passed between nodes in the LangGraph workflow.
    Fields marked with Annotated[..., add] will accumulate values across nodes.
    """

    # Input
    query: str
    """User's original natural language query"""

    # Intent analysis results
    intent_type: str
    """Query intent type: 'profile', 'event', or 'mixed'"""

    confidence: float
    """Intent classification confidence score (0.0-1.0)"""

    # LLM extracted structured information
    profile_attributes: List[Dict]
    """
    Extracted profile attributes from query
    Format: [{"attribute_name": "年龄", "query_text": "年龄: 25到35岁"}]
    """

    # Milvus query results (with original query context)
    profile_results: Annotated[List[Dict], add]
    """
    Profile attribute search results (accumulated)
    Structure: [{
        "matched_field": {...},  # Milvus returned field info
        "original_query": "25到35岁",  # Original query text
        "original_attribute": "年龄"  # LLM extracted attribute name
    }]
    """


    # Final output
    final_result: Optional[Dict]
    """Formatted final result with aggregated information"""

    error: Optional[str]
    """Error message if any error occurred during processing"""
