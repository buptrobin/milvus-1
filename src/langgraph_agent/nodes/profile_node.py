"""
Search Profiles Node

This node searches for profile attributes in Milvus.
"""

import logging
from typing import Dict, Any, List

from ..state import AgentState

logger = logging.getLogger(__name__)


def search_profiles_node(
    state: AgentState,
    milvus_client: Any,  # MilvusClient instance
    embedding_manager: Any,  # EmbeddingManager instance
    similarity_threshold: float = 0.65
) -> Dict[str, Any]:
    """
    Search for profile attributes in Milvus

    Args:
        state: Current agent state
        milvus_client: Milvus client instance
        embedding_manager: Embedding manager instance
        similarity_threshold: Minimum similarity score threshold

    Returns:
        Updated state with profile_results
    """
    logger.info("[search_profiles] Starting profile attribute search")

    profile_attributes = state.get("profile_attributes", [])
    if not profile_attributes:
        logger.info("[search_profiles] No profile attributes to search")
        return {"profile_results": []}

    try:
        all_results = []

        # Generate embeddings for all profile attributes
        query_texts = [attr["query_text"] for attr in profile_attributes]
        logger.info(f"[search_profiles] Generating embeddings for {len(query_texts)} queries")
        logger.debug(f"[search_profiles] Query texts: {query_texts}")

        embeddings = embedding_manager.encode(query_texts)
        logger.debug(f"[search_profiles] Generated {len(embeddings)} embeddings, each with dimension {len(embeddings[0]) if embeddings else 'N/A'}")

        # Search for each attribute
        for idx, (attr, embedding) in enumerate(zip(profile_attributes, embeddings)):
            logger.info(f"[search_profiles] Searching for attribute: {attr['attribute_name']} (query: '{attr['query_text']}')")
            logger.debug(f"[search_profiles] Embedding #{idx+1} (first 5): [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, {embedding[3]:.4f}, {embedding[4]:.4f}, ...]")

            # Search in Milvus (PROFILE_ATTRIBUTE type)
            # Note: Based on the milvus_client code, we need to check the actual field names
            search_results = milvus_client.search_profile_attributes(
                query_vector=embedding,
                limit=5
            )

            logger.debug(f"[search_profiles] Got {len(search_results)} results from Milvus for '{attr['attribute_name']}'")

            # Format results with original query context
            for result in search_results:
                logger.debug(f"[search_profiles] Result: id={result['id']}, score={result['score']:.4f}, idname={result.get('idname', 'N/A')}, source_name={result.get('source_name', 'N/A')}, threshold={similarity_threshold}")

                if result["score"] >= similarity_threshold:
                    formatted_result = {
                        "matched_field": {
                            "id": result["id"],
                            "score": result["score"],
                            "source_type": result.get("source_type", "PROFILE_ATTRIBUTE"),
                            "source": result.get("source_name", ""),  # Table/source name (e.g., "pampers_customer")
                            "source_name": result.get("source_name", ""),  # Display name (same as source for now)
                            "idname": result.get("idname", ""),  # Unique field identifier (e.g., "age_group")
                            "raw_metadata": result.get("raw_metadata", {})
                        },
                        "original_query": attr["query_text"],
                        "original_attribute": attr["attribute_name"]
                    }
                    all_results.append(formatted_result)
                    logger.info(
                        f"[search_profiles] Matched: {result.get('idname', 'N/A')} (source: {result.get('source_name', 'N/A')}) "
                        f"(score={result['score']:.3f})"
                    )
                else:
                    logger.debug(f"[search_profiles] Filtered out (below threshold): {result.get('idname', 'N/A')} (score={result['score']:.3f})")

        logger.info(f"[search_profiles] Found {len(all_results)} profile attribute matches (after threshold filtering)")

        return {"profile_results": all_results}

    except Exception as e:
        logger.error(f"[search_profiles] Error: {e}", exc_info=True)
        return {
            "profile_results": [],
            "error": f"Profile search failed: {str(e)}"
        }
