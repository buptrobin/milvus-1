"""
Search Event Attributes Node

This node searches for event attributes in Milvus.
"""

import logging
from typing import Dict, Any, List

from ..state import AgentState

logger = logging.getLogger(__name__)


def search_event_attributes_node(
    state: AgentState,
    milvus_client: Any,  # MilvusClient instance
    embedding_manager: Any,  # EmbeddingManager instance
    similarity_threshold: float = 0.65,
    ambiguity_threshold: float = 0.05  # If top 2 scores differ by less than this, mark as ambiguous
) -> Dict[str, Any]:
    """
    Search for event attributes in Milvus

    This node searches for event attributes based on the events found in the previous step.
    It correlates event attributes with their parent events.

    Args:
        state: Current agent state
        milvus_client: Milvus client instance
        embedding_manager: Embedding manager instance
        similarity_threshold: Minimum similarity score threshold
        ambiguity_threshold: If top 2 scores differ by less than this value, mark as ambiguous

    Returns:
        Updated state with event_attr_results
    """
    logger.info("[search_event_attrs] Starting event attribute search")

    event_results = state.get("event_results", [])
    if not event_results:
        logger.info("[search_event_attrs] No events found, skipping event attribute search")
        return {"event_attr_results": []}

    try:
        all_results = []

        # For each matched event, search for its attributes
        for event_result in event_results:
            event_source = event_result.get("matched_field", {}).get("idname", "")
            event_source_name = event_result.get("matched_field", {}).get("source_name", "")
            event_attributes = event_result.get("event_attributes", [])

            if not event_source:
                logger.warning(f"[search_event_attrs] Event result missing idname, skipping")
                continue

            if not event_attributes:
                logger.debug(f"[search_event_attrs] No attributes to search for event '{event_source}'")
                continue

            logger.info(f"[search_event_attrs] Searching attributes for event: {event_source} (attributes: {len(event_attributes)})")

            # Generate embeddings for all event attributes
            query_texts = []
            for attr in event_attributes:
                # Handle both string and dict formats
                if isinstance(attr, str):
                    query_texts.append(attr)
                else:
                    # If it's a dict, it should already be formatted as "key: value"
                    query_texts.append(str(attr))

            logger.debug(f"[search_event_attrs] Generating embeddings for {len(query_texts)} attribute queries")
            logger.debug(f"[search_event_attrs] Query texts: {query_texts}")

            embeddings = embedding_manager.encode(query_texts)
            logger.debug(f"[search_event_attrs] Generated {len(embeddings)} embeddings")

            # Search for each attribute
            for idx, (attr_query, embedding) in enumerate(zip(query_texts, embeddings)):
                logger.info(f"[search_event_attrs] Searching for attribute: '{attr_query}' of event '{event_source}'")
                logger.debug(f"[search_event_attrs] Embedding #{idx+1} (first 5): [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, {embedding[3]:.4f}, {embedding[4]:.4f}, ...]")

                # Search in Milvus (EVENT_ATTRIBUTE type, filtered by event_source)
                search_results = milvus_client.search_event_attributes(
                    query_vector=embedding,
                    source=event_source,
                    limit=5
                )

                logger.debug(f"[search_event_attrs] Got {len(search_results)} results from Milvus for '{attr_query}' (event: {event_source})")

                # Filter results by threshold and keep only the highest scoring one
                valid_results = [r for r in search_results if r["score"] >= similarity_threshold]

                if valid_results:
                    # Sort by score descending
                    sorted_results = sorted(valid_results, key=lambda x: x["score"], reverse=True)
                    best_result = sorted_results[0]

                    # Check for ambiguity: if top 2 scores are very close
                    has_ambiguity = False
                    if len(sorted_results) >= 2:
                        score_diff = sorted_results[0]["score"] - sorted_results[1]["score"]
                        has_ambiguity = score_diff < ambiguity_threshold
                        if has_ambiguity:
                            logger.warning(
                                f"[search_event_attrs] Ambiguity detected for '{attr_query}' (event: {event_source}): "
                                f"top score={sorted_results[0]['score']:.4f} ({sorted_results[0].get('idname', 'N/A')}), "
                                f"2nd score={sorted_results[1]['score']:.4f} ({sorted_results[1].get('idname', 'N/A')}), "
                                f"diff={score_diff:.4f} < threshold={ambiguity_threshold}"
                            )

                    logger.debug(
                        f"[search_event_attrs] Best result: id={best_result['id']}, score={best_result['score']:.4f}, "
                        f"idname={best_result.get('idname', 'N/A')}, source={best_result.get('source', 'N/A')}"
                    )

                    formatted_result = {
                        "matched_field": {
                            "id": best_result["id"],
                            "score": best_result["score"],
                            "source_type": best_result.get("source_type", "EVENT_ATTRIBUTE"),
                            "source_name": best_result.get("source_name", ""),
                            "idname": best_result.get("idname", ""),  # Attribute idname
                            "source": best_result.get("source", event_source),  # Parent event idname
                            "raw_metadata": best_result.get("raw_metadata", {}),
                            "has_ambiguity": has_ambiguity  # Flag indicating if result is ambiguous
                        },
                        "original_query": attr_query,
                        "source": event_source,
                        "event_name": event_source_name
                    }
                    all_results.append(formatted_result)
                    logger.info(
                        f"[search_event_attrs] Matched (best): {best_result.get('idname', 'N/A')} "
                        f"(event: {event_source}, score={best_result['score']:.3f}, has_ambiguity={has_ambiguity})"
                    )

                    # Log filtered out results
                    for result in search_results:
                        if result["id"] != best_result["id"]:
                            if result["score"] >= similarity_threshold:
                                logger.debug(
                                    f"[search_event_attrs] Filtered out (lower score): "
                                    f"{result.get('idname', 'N/A')} (score={result['score']:.3f})"
                                )
                            else:
                                logger.debug(
                                    f"[search_event_attrs] Filtered out (below threshold): "
                                    f"{result.get('idname', 'N/A')} (score={result['score']:.3f})"
                                )
                else:
                    logger.info(f"[search_event_attrs] No results above threshold for '{attr_query}' (event: {event_source})")

        logger.info(f"[search_event_attrs] Found {len(all_results)} event attribute matches (after threshold filtering)")

        return {"event_attr_results": all_results}

    except Exception as e:
        logger.error(f"[search_event_attrs] Error: {e}", exc_info=True)
        return {
            "event_attr_results": [],
            "error": f"Event attribute search failed: {str(e)}"
        }
