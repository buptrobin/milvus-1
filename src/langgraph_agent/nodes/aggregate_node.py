"""
Aggregate Results Node

This node aggregates, deduplicates, and formats all search results.
"""

import logging
import time
from typing import Dict, Any, List
from collections import defaultdict

from ..state import AgentState

logger = logging.getLogger(__name__)


def aggregate_results_node(
    state: AgentState,
    similarity_threshold: float = 0.65,
    ambiguity_threshold: float = 0.75
) -> Dict[str, Any]:
    """
    Aggregate and format all search results

    This node:
    1. Deduplicates results by ID
    2. Sorts by score
    3. Applies similarity threshold
    4. Detects ambiguities (multiple high-scoring results for same query)
    5. Generates summary and confidence score

    Args:
        state: Current agent state
        similarity_threshold: Minimum score to include in results
        ambiguity_threshold: Score threshold for ambiguity detection

    Returns:
        Updated state with final_result
    """
    logger.info("[aggregate_results] Starting result aggregation")

    try:
        # Collect all results
        profile_results = state.get("profile_results", [])
        event_results = state.get("event_results", [])
        event_attr_results = state.get("event_attr_results", [])

        logger.info(
            f"[aggregate_results] Input counts - "
            f"profiles: {len(profile_results)}, "
            f"events: {len(event_results)}, "
            f"event_attrs: {len(event_attr_results)}"
        )

        # Process all result types
        profile_attributes = _process_profile_results(profile_results, similarity_threshold)
        events = _process_event_results(event_results, similarity_threshold)
        event_attributes = _process_event_attr_results(event_attr_results, similarity_threshold)

        # Detect ambiguities across all result types
        ambiguous_options, has_ambiguity = _detect_ambiguities(
            profile_results, event_results, event_attr_results, ambiguity_threshold
        )

        # Generate summary
        summary = _generate_summary(profile_attributes, events, event_attributes)

        # Calculate overall confidence
        all_scores = []
        all_scores.extend([attr["score"] for attr in profile_attributes])
        all_scores.extend([evt["score"] for evt in events])
        all_scores.extend([attr["score"] for attr in event_attributes])

        confidence_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # Build final result
        final_result = {
            "query": state.get("query", ""),
            "intent_type": state.get("intent_type", "unknown"),
            "profile_attributes": profile_attributes,
            "events": events,
            "event_attributes": event_attributes,
            "summary": summary,
            "total_results": len(profile_attributes) + len(events) + len(event_attributes),
            "confidence_score": round(confidence_score, 2),
            "has_ambiguity": has_ambiguity,
            "ambiguous_options": ambiguous_options,
            "execution_time": 0.0  # Will be set by caller
        }

        logger.info(
            f"[aggregate_results] Final result - "
            f"profiles: {len(profile_attributes)}, "
            f"events: {len(events)}, "
            f"event_attrs: {len(event_attributes)}, "
            f"ambiguities: {len(ambiguous_options)}"
        )

        return {"final_result": final_result}

    except Exception as e:
        logger.error(f"[aggregate_results] Error: {e}", exc_info=True)
        return {
            "final_result": {
                "query": state.get("query", ""),
                "error": f"Aggregation failed: {str(e)}",
                "profile_attributes": [],
                "events": [],
                "event_attributes": [],
                "summary": "处理失败",
                "total_results": 0,
                "confidence_score": 0.0,
                "has_ambiguity": False,
                "ambiguous_options": []
            }
        }


def _process_profile_results(
    profile_results: List[Dict],
    threshold: float
) -> List[Dict]:
    """Process and deduplicate profile results"""
    seen_ids = set()
    processed = []

    # Sort by score (descending)
    sorted_results = sorted(
        profile_results,
        key=lambda x: x.get("matched_field", {}).get("score", 0),
        reverse=True
    )

    for result in sorted_results:
        matched_field = result.get("matched_field", {})
        result_id = matched_field.get("id")
        score = matched_field.get("score", 0)

        # Skip if below threshold or duplicate
        if score < threshold or result_id in seen_ids:
            continue

        seen_ids.add(result_id)

        # Determine confidence level
        confidence_level = _get_confidence_level(score)

        processed.append({
            "idname": matched_field.get("idname", ""),
            "source_name": matched_field.get("source_name", ""),
            "source": matched_field.get("source", ""),
            "original_query": result.get("original_query", ""),
            "original_attribute": result.get("original_attribute", ""),
            "score": round(score, 3),
            "confidence_level": confidence_level,
            "explanation": str(matched_field.get("raw_metadata", {}))
        })

    return processed


def _process_event_results(
    event_results: List[Dict],
    threshold: float
) -> List[Dict]:
    """Process and deduplicate event results"""
    seen_ids = set()
    processed = []

    sorted_results = sorted(
        event_results,
        key=lambda x: x.get("matched_field", {}).get("score", 0),
        reverse=True
    )

    for result in sorted_results:
        matched_field = result.get("matched_field", {})
        result_id = matched_field.get("id")
        score = matched_field.get("score", 0)

        if score < threshold or result_id in seen_ids:
            continue

        seen_ids.add(result_id)

        confidence_level = _get_confidence_level(score)

        processed.append({
            "idname": matched_field.get("idname", ""),
            "source_name": matched_field.get("source_name", ""),
            "source": matched_field.get("source", ""),
            "original_query": result.get("original_query", ""),
            "score": round(score, 3),
            "confidence_level": confidence_level,
            "explanation": str(matched_field.get("raw_metadata", {}))
        })

    return processed


def _process_event_attr_results(
    event_attr_results: List[Dict],
    threshold: float
) -> List[Dict]:
    """Process and deduplicate event attribute results"""
    seen_ids = set()
    processed = []

    sorted_results = sorted(
        event_attr_results,
        key=lambda x: x.get("matched_field", {}).get("score", 0),
        reverse=True
    )

    for result in sorted_results:
        matched_field = result.get("matched_field", {})
        result_id = matched_field.get("id")
        score = matched_field.get("score", 0)

        if score < threshold or result_id in seen_ids:
            continue

        seen_ids.add(result_id)

        confidence_level = _get_confidence_level(score)

        event_idname = result.get("event_idname", "")

        processed.append({
            "idname": matched_field.get("idname", ""),
            "source_name": matched_field.get("source_name", ""),
            "event_idname": event_idname,
            "event_name": result.get("event_name", event_idname),
            "original_query": result.get("original_query", ""),
            "score": round(score, 3),
            "confidence_level": confidence_level,
            "explanation": str(matched_field.get("raw_metadata", {}))
        })

    return processed


def _get_confidence_level(score: float) -> str:
    """Determine confidence level from score"""
    if score >= 0.85:
        return "high"
    elif score >= 0.70:
        return "medium"
    else:
        return "low"


def _detect_ambiguities(
    profile_results: List[Dict],
    event_results: List[Dict],
    event_attr_results: List[Dict],
    threshold: float
) -> tuple[List[Dict], bool]:
    """
    Detect ambiguous matches (multiple high-scoring results for same query)

    Args:
        profile_results: Profile search results
        event_results: Event search results
        event_attr_results: Event attribute search results
        threshold: Score threshold for ambiguity (default 0.75)

    Returns:
        Tuple of (ambiguous_options, has_ambiguity)
    """
    ambiguous_options = []

    # Group profile results by original query
    profile_groups = defaultdict(list)
    for result in profile_results:
        original_query = result.get("original_query", "")
        matched_field = result.get("matched_field", {})
        score = matched_field.get("score", 0)

        if score >= threshold:
            profile_groups[original_query].append({
                "idname": matched_field.get("idname", ""),
                "source_name": matched_field.get("source_name", ""),
                "score": round(score, 3)
            })

    # Check for ambiguities in profiles
    for query, candidates in profile_groups.items():
        if len(candidates) > 1:
            ambiguous_options.append({
                "category": "profile",
                "original_query": query,
                "candidates": candidates
            })

    # Group event results by original query
    event_groups = defaultdict(list)
    for result in event_results:
        original_query = result.get("original_query", "")
        matched_field = result.get("matched_field", {})
        score = matched_field.get("score", 0)

        if score >= threshold:
            event_groups[original_query].append({
                "idname": matched_field.get("idname", ""),
                "source_name": matched_field.get("source_name", ""),
                "score": round(score, 3)
            })

    # Check for ambiguities in events
    for query, candidates in event_groups.items():
        if len(candidates) > 1:
            ambiguous_options.append({
                "category": "event",
                "original_query": query,
                "candidates": candidates
            })

    # Group event attribute results by original query
    event_attr_groups = defaultdict(list)
    for result in event_attr_results:
        original_query = result.get("original_query", "")
        matched_field = result.get("matched_field", {})
        score = matched_field.get("score", 0)

        if score >= threshold:
            event_attr_groups[original_query].append({
                "idname": matched_field.get("idname", ""),
                "source_name": matched_field.get("source_name", ""),
                "event_idname": result.get("event_idname", ""),
                "score": round(score, 3)
            })

    # Check for ambiguities in event attributes
    for query, candidates in event_attr_groups.items():
        if len(candidates) > 1:
            ambiguous_options.append({
                "category": "event_attribute",
                "original_query": query,
                "candidates": candidates
            })

    has_ambiguity = len(ambiguous_options) > 0

    return ambiguous_options, has_ambiguity


def _generate_summary(
    profile_attributes: List[Dict],
    events: List[Dict],
    event_attributes: List[Dict]
) -> str:
    """Generate a human-readable summary of the results"""
    parts = []

    if profile_attributes:
        attr_summaries = []
        for attr in profile_attributes[:3]:  # Show top 3
            attr_summaries.append(
                f"{attr['source_name']}(查询条件:{attr.get('original_query', '')})"
            )
        parts.append("已识别用户属性: " + ", ".join(attr_summaries))

    if events:
        event_summaries = []
        for evt in events[:3]:  # Show top 3
            event_summaries.append(f"{evt['source_name']}")
        parts.append("已识别事件: " + ", ".join(event_summaries))

    if event_attributes:
        attr_summaries = []
        for attr in event_attributes[:3]:  # Show top 3
            attr_summaries.append(
                f"{attr['source_name']}(事件:{attr.get('event_name', '')})"
            )
        parts.append("已识别事件属性: " + ", ".join(attr_summaries))

    if not parts:
        return "未找到匹配的字段"

    return "; ".join(parts)
