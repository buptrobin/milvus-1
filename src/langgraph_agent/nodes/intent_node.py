"""
Intent Classification Node

This node uses LLM to understand user query and extract structured information.
"""

import logging
from typing import Dict, Any

from ...llm_extractor import VolcanoLLMExtractor
from ..state import AgentState

logger = logging.getLogger(__name__)


def intent_classification_node(
    state: AgentState,
    llm_extractor: VolcanoLLMExtractor
) -> Dict[str, Any]:
    """
    Intent classification and information extraction node

    This node:
    1. Calls LLM to understand user query
    2. Extracts profile attributes and events
    3. Determines intent type (profile/event/mixed)

    Args:
        state: Current agent state
        llm_extractor: LLM extractor instance

    Returns:
        Updated state with intent_type, confidence, profile_attributes, and events
    """
    logger.info(f"[intent_classification] Processing query: {state['query']}")

    try:
        # Call LLM to extract information
        extracted_info = llm_extractor.extract(state["query"])

        # Log extracted info details
        logger.debug(f"[intent_classification] ExtractedInfo - intent_type: {extracted_info.intent_type}, confidence: {extracted_info.intent_confidence}")
        logger.debug(f"[intent_classification] ExtractedInfo - has_structured_query: {extracted_info.structured_query is not None}")
        if extracted_info.structured_query:
            import json
            logger.debug(f"[intent_classification] structured_query: {json.dumps(extracted_info.structured_query, ensure_ascii=False)}")

        # Parse structured query from LLM response
        profile_attributes = []
        events = []
        intent_type = "profile"  # default

        if extracted_info.structured_query:
            sq = extracted_info.structured_query

            # Handle new format (person_attributes as dict, behavioral_events as list)
            if "person_attributes" in sq:
                person_attrs = sq["person_attributes"]
                logger.debug(f"[intent_classification] person_attributes type: {type(person_attrs)}, value: {person_attrs}")

                if isinstance(person_attrs, dict):
                    # New format: {"年龄": "25到35岁", "性别": "男性"}
                    for attr_name, attr_value in person_attrs.items():
                        profile_attributes.append({
                            "attribute_name": attr_name,
                            "query_text": f"{attr_name}: {attr_value}"
                        })
                elif isinstance(person_attrs, list):
                    # Old format: ["年龄", "性别"]
                    for attr_name in person_attrs:
                        profile_attributes.append({
                            "attribute_name": attr_name,
                            "query_text": attr_name
                        })

            # Handle behavioral_events (new format) or events (old format)
            behavioral_events = sq.get("behavioral_events", sq.get("events", []))
            logger.debug(f"[intent_classification] behavioral_events count: {len(behavioral_events) if behavioral_events else 0}")

            if isinstance(behavioral_events, list):
                for event in behavioral_events:
                    if isinstance(event, dict):
                        # Extract event_type or event_description
                        event_desc = event.get("event_type") or event.get("event_description", "")

                        # Extract event attributes
                        event_attributes = []
                        attrs = event.get("attributes", {})

                        if isinstance(attrs, dict):
                            # New format: {"时间范围": "过去90天", "频率": "至少3次"}
                            for attr_name, attr_value in attrs.items():
                                event_attributes.append(f"{attr_name}: {attr_value}")
                        elif isinstance(attrs, list):
                            # Old format: ["购买金额", "购买时间"]
                            event_attributes = attrs

                        events.append({
                            "event_description": event_desc,
                            "event_attributes": event_attributes
                        })

        # Log parsed attributes
        logger.debug(f"[intent_classification] Parsed {len(profile_attributes)} profile attributes:")
        for idx, attr in enumerate(profile_attributes):
            logger.debug(f"[intent_classification]   #{idx+1}: {attr}")
        logger.debug(f"[intent_classification] Parsed {len(events)} events:")
        for idx, evt in enumerate(events):
            logger.debug(f"[intent_classification]   #{idx+1}: {evt}")

        # Determine intent type
        has_profiles = len(profile_attributes) > 0
        has_events = len(events) > 0

        if has_profiles and has_events:
            intent_type = "mixed"
        elif has_events:
            intent_type = "event"
        elif has_profiles:
            intent_type = "profile"
        else:
            # Fallback: try to infer from query keywords
            query_lower = state["query"].lower()
            if any(kw in query_lower for kw in ["购买", "登录", "下单", "浏览", "注册", "事件"]):
                intent_type = "event"
            else:
                intent_type = "profile"

        # Get confidence from LLM or use default
        confidence = extracted_info.intent_confidence if extracted_info.intent_confidence > 0 else 0.8

        logger.info(
            f"[intent_classification] Extracted: intent_type={intent_type}, "
            f"confidence={confidence:.2f}, "
            f"profile_attributes={len(profile_attributes)}, "
            f"events={len(events)}"
        )

        return {
            "intent_type": intent_type,
            "confidence": confidence,
            "profile_attributes": profile_attributes,
            "events": events,
        }

    except Exception as e:
        logger.error(f"[intent_classification] Error: {e}", exc_info=True)
        return {
            "intent_type": "profile",
            "confidence": 0.5,
            "profile_attributes": [],
            "events": [],
            "error": f"Intent classification failed: {str(e)}"
        }
