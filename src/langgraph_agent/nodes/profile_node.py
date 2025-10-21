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
    similarity_threshold: float = 0.65,
    ambiguity_threshold: float = 0.05  # If top 2 scores differ by less than this, mark as ambiguous
) -> Dict[str, Any]:
    """
    Search for profile attributes in Milvus

    Args:
        state: Current agent state
        milvus_client: Milvus client instance
        embedding_manager: Embedding manager instance
        similarity_threshold: Minimum similarity score threshold
        ambiguity_threshold: If top 2 scores differ by less than this value, mark as ambiguous

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
                            f"[search_profiles] Ambiguity detected for '{attr['attribute_name']}': "
                            f"top score={sorted_results[0]['score']:.4f} ({sorted_results[0].get('idname', 'N/A')}), "
                            f"2nd score={sorted_results[1]['score']:.4f} ({sorted_results[1].get('idname', 'N/A')}), "
                            f"diff={score_diff:.4f} < threshold={ambiguity_threshold}"
                        )

                logger.debug(f"[search_profiles] Best result: id={best_result['id']}, score={best_result['score']:.4f}, idname={best_result.get('idname', 'N/A')}, source_name={best_result.get('source_name', 'N/A')}")

                formatted_result = {
                    "matched_field": {
                        "id": best_result["id"],
                        "score": best_result["score"],
                        "source_type": best_result.get("source_type", "PROFILE_ATTRIBUTE"),
                        "source": best_result.get("source_name", ""),  # Table/source name (e.g., "pampers_customer")
                        "source_name": best_result.get("source_name", ""),  # Display name (same as source for now)
                        "idname": best_result.get("idname", ""),  # Unique field identifier (e.g., "age_group")
                        "raw_metadata": best_result.get("raw_metadata", {}),
                        "has_ambiguity": has_ambiguity  # Flag indicating if result is ambiguous
                    },
                    "original_query": attr["query_text"],
                    "original_attribute": attr["attribute_name"]
                }
                all_results.append(formatted_result)
                logger.info(
                    f"[search_profiles] Matched (best): {best_result.get('idname', 'N/A')} (source: {best_result.get('source_name', 'N/A')}) "
                    f"(score={best_result['score']:.3f}, has_ambiguity={has_ambiguity})"
                )

                # Log filtered out results
                for result in search_results:
                    if result["id"] != best_result["id"]:
                        if result["score"] >= similarity_threshold:
                            logger.debug(f"[search_profiles] Filtered out (lower score): {result.get('idname', 'N/A')} (score={result['score']:.3f})")
                        else:
                            logger.debug(f"[search_profiles] Filtered out (below threshold): {result.get('idname', 'N/A')} (score={result['score']:.3f})")
            else:
                logger.info(f"[search_profiles] No results above threshold for '{attr['attribute_name']}'")

        logger.info(f"[search_profiles] Found {len(all_results)} profile attribute matches (after threshold filtering)")

        return {"profile_results": all_results}

    except Exception as e:
        logger.error(f"[search_profiles] Error: {e}", exc_info=True)
        return {
            "profile_results": [],
            "error": f"Profile search failed: {str(e)}"
        }
