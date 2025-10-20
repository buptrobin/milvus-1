"""
Search Events Node

This node searches for events in Milvus.
"""

import logging
from typing import Dict, Any, List

from ..state import AgentState

logger = logging.getLogger(__name__)


def search_events_node(
    state: AgentState,
    milvus_client: Any,  # MilvusClient instance
    embedding_manager: Any,  # EmbeddingManager instance
    similarity_threshold: float = 0.65
) -> Dict[str, Any]:
    """
    Search for events in Milvus

    Args:
        state: Current agent state
        milvus_client: Milvus client instance
        embedding_manager: Embedding manager instance
        similarity_threshold: Minimum similarity score threshold

    Returns:
        Updated state with event_results
    """
    logger.info("[search_events] Starting event search")

    events = state.get("events", [])
    if not events:
        logger.info("[search_events] No events to search")
        return {"event_results": []}

    try:
        all_results = []

        # Generate embeddings for all events
        query_texts = [event["event_description"] for event in events]
        logger.info(f"[search_events] Generating embeddings for {len(query_texts)} queries")
        logger.debug(f"[search_events] Query texts: {query_texts}")

        embeddings = embedding_manager.encode(query_texts)
        logger.debug(f"[search_events] Generated {len(embeddings)} embeddings, each with dimension {len(embeddings[0]) if embeddings else 'N/A'}")

        # Search for each event
        for idx, (event, embedding) in enumerate(zip(events, embeddings)):
            logger.info(f"[search_events] Searching for event: {event['event_description']}")
            logger.debug(f"[search_events] Embedding #{idx+1} (first 5): [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, {embedding[3]:.4f}, {embedding[4]:.4f}, ...]")

            # Search in Milvus (EVENT type)
            search_results = milvus_client.search_events(
                query_vector=embedding,
                limit=5
            )

            logger.debug(f"[search_events] Got {len(search_results)} results from Milvus for '{event['event_description']}'")

            # Filter results by threshold and keep only the highest scoring one
            valid_results = [r for r in search_results if r["score"] >= similarity_threshold]

            if valid_results:
                # Sort by score descending and take the top one
                best_result = max(valid_results, key=lambda x: x["score"])

                logger.debug(f"[search_events] Best result: id={best_result['id']}, score={best_result['score']:.4f}, idname={best_result.get('idname', 'N/A')}, source_name={best_result.get('source_name', 'N/A')}")

                formatted_result = {
                    "matched_field": {
                        "id": best_result["id"],
                        "score": best_result["score"],
                        "source_type": best_result.get("source_type", "EVENT"),
                        "source": best_result.get("source", ""),  # Table/source name
                        "source_name": best_result.get("source_name", ""),  # Display name
                        "idname": best_result.get("idname", ""),  # Unique event identifier
                        "raw_metadata": best_result.get("raw_metadata", {})
                    },
                    "original_query": event["event_description"],
                    "event_attributes": event.get("event_attributes", [])  # Store event attributes for later search
                }
                all_results.append(formatted_result)
                logger.info(
                    f"[search_events] Matched (best): {best_result.get('idname', 'N/A')} (source: {best_result.get('source', 'N/A')}) "
                    f"(score={best_result['score']:.3f})"
                )

                # Log filtered out results
                for result in search_results:
                    if result["id"] != best_result["id"]:
                        if result["score"] >= similarity_threshold:
                            logger.debug(f"[search_events] Filtered out (lower score): {result.get('idname', 'N/A')} (score={result['score']:.3f})")
                        else:
                            logger.debug(f"[search_events] Filtered out (below threshold): {result.get('idname', 'N/A')} (score={result['score']:.3f})")
            else:
                logger.info(f"[search_events] No results above threshold for '{event['event_description']}'")

        logger.info(f"[search_events] Found {len(all_results)} event matches (after threshold filtering)")

        return {"event_results": all_results}

    except Exception as e:
        logger.error(f"[search_events] Error: {e}", exc_info=True)
        return {
            "event_results": [],
            "error": f"Event search failed: {str(e)}"
        }
