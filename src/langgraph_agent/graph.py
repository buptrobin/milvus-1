"""
LangGraph Workflow Definition

This module defines the complete workflow graph for the natural language query agent.
"""

import logging
import time
from typing import Any, Dict
from functools import partial

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    intent_classification_node,
    route_query,
    search_profiles_node,
    search_events_node,
    search_event_attributes_node,
    aggregate_results_node,
)

logger = logging.getLogger(__name__)


def create_agent_graph(
    llm_extractor: Any,
    milvus_client: Any,
    embedding_manager: Any,
    similarity_threshold: float = 0.5,
    ambiguity_threshold: float = 0.75
) -> Any:
    """
    Create and compile the LangGraph workflow

    Args:
        llm_extractor: VolcanoLLMExtractor instance for intent classification
        milvus_client: MilvusClient instance for vector search
        embedding_manager: EmbeddingManager instance for generating embeddings
        similarity_threshold: Minimum similarity score for results
        ambiguity_threshold: Score threshold for ambiguity detection

    Returns:
        Compiled LangGraph application
    """
    logger.info("Creating LangGraph workflow...")

    # Create graph
    workflow = StateGraph(AgentState)

    # Create wrapped node functions with dependencies injected
    def intent_node_wrapper(state: AgentState) -> Dict[str, Any]:
        return intent_classification_node(state, llm_extractor)

    def profiles_node_wrapper(state: AgentState) -> Dict[str, Any]:
        return search_profiles_node(
            state, milvus_client, embedding_manager, similarity_threshold
        )

    def events_node_wrapper(state: AgentState) -> Dict[str, Any]:
        return search_events_node(
            state, milvus_client, embedding_manager, similarity_threshold
        )

    def event_attrs_node_wrapper(state: AgentState) -> Dict[str, Any]:
        return search_event_attributes_node(
            state, milvus_client, embedding_manager, similarity_threshold
        )

    def profiles_and_events_wrapper(state: AgentState) -> Dict[str, Any]:
        """Combined node that searches both profiles and events in parallel"""
        # Search profiles
        profile_results = search_profiles_node(
            state, milvus_client, embedding_manager, similarity_threshold
        )
        # Search events
        event_results = search_events_node(
            state, milvus_client, embedding_manager, similarity_threshold
        )
        # Merge results (both return updates to different state fields)
        return {**profile_results, **event_results}

    def aggregate_node_wrapper(state: AgentState) -> Dict[str, Any]:
        return aggregate_results_node(state, similarity_threshold, ambiguity_threshold)

    # Add nodes to the graph
    workflow.add_node("intent_classification", intent_node_wrapper)
    workflow.add_node("search_profiles", profiles_node_wrapper)
    workflow.add_node("search_events", events_node_wrapper)
    workflow.add_node("search_event_attributes", event_attrs_node_wrapper)
    workflow.add_node("search_profiles_and_events", profiles_and_events_wrapper)
    workflow.add_node("aggregate_results", aggregate_node_wrapper)

    # Set entry point
    workflow.set_entry_point("intent_classification")

    # Add conditional routing from intent_classification
    workflow.add_conditional_edges(
        "intent_classification",
        route_query,
        {
            "search_profiles": "search_profiles",
            "search_events": "search_events",
            "search_mixed": "search_profiles_and_events"
        }
    )

    # Profile-only path: profiles -> aggregate
    workflow.add_edge("search_profiles", "aggregate_results")

    # Event path: events -> event_attributes -> aggregate
    workflow.add_edge("search_events", "search_event_attributes")
    workflow.add_edge("search_event_attributes", "aggregate_results")

    # Mixed path: profiles_and_events -> event_attributes -> aggregate
    workflow.add_edge("search_profiles_and_events", "search_event_attributes")

    # Add edge from aggregate to END
    workflow.add_edge("aggregate_results", END)

    # Compile the graph
    app = workflow.compile()

    logger.info("LangGraph workflow compiled successfully")

    return app


def run_agent(
    app: Any,
    query: str,
    initial_state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Run the agent with a user query

    Args:
        app: Compiled LangGraph application
        query: User's natural language query
        initial_state: Optional initial state (for debugging/testing)

    Returns:
        Final state with results
    """
    start_time = time.time()

    logger.info(f"Running agent with query: {query}")

    # Prepare initial state
    if initial_state is None:
        initial_state = {
            "query": query,
            "intent_type": "",
            "confidence": 0.0,
            "profile_attributes": [],
            "events": [],
            "profile_results": [],
            "event_results": [],
            "event_attr_results": [],
            "final_result": None,
            "error": None
        }
    else:
        initial_state["query"] = query

    try:
        # Run the graph
        final_state = app.invoke(initial_state)

        # Add execution time to final result
        execution_time = time.time() - start_time
        if final_state.get("final_result"):
            final_state["final_result"]["execution_time"] = round(execution_time, 2)

        logger.info(
            f"Agent execution completed in {execution_time:.2f}s - "
            f"intent={final_state.get('intent_type', 'unknown')}, "
            f"results={final_state.get('final_result', {}).get('total_results', 0)}"
        )

        return final_state

    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        return {
            "query": query,
            "error": str(e),
            "final_result": {
                "query": query,
                "error": f"执行失败: {str(e)}",
                "profile_attributes": [],
                "events": [],
                "event_attributes": [],
                "summary": "执行失败",
                "total_results": 0,
                "confidence_score": 0.0,
                "has_ambiguity": False,
                "ambiguous_options": [],
                "execution_time": round(time.time() - start_time, 2)
            }
        }
