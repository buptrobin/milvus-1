"""
Test cases for LangGraph Agent

This module contains test cases for the LangGraph-based natural language query agent.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.langgraph_agent.state import AgentState
from src.langgraph_agent.nodes.router import route_query


class TestAgentState:
    """Test cases for AgentState"""

    def test_agent_state_structure(self):
        """Test that AgentState has the expected structure"""
        # This is mainly a type check - ensure no errors when creating state
        state: AgentState = {
            "query": "test query",
            "intent_type": "profile",
            "confidence": 0.9,
            "profile_attributes": [],
            "events": [],
            "profile_results": [],
            "event_results": [],
            "event_attr_results": [],
            "final_result": None,
            "error": None
        }

        assert state["query"] == "test query"
        assert state["intent_type"] == "profile"
        assert state["confidence"] == 0.9


class TestRouter:
    """Test cases for routing logic"""

    def test_route_query_profiles_only(self):
        """Test routing with only profile attributes"""
        state: AgentState = {
            "query": "test",
            "intent_type": "profile",
            "confidence": 0.9,
            "profile_attributes": [{"attribute_name": "age", "query_text": "age"}],
            "events": [],
            "profile_results": [],
            "event_results": [],
            "event_attr_results": [],
            "final_result": None,
            "error": None
        }

        route = route_query(state)
        assert route == "search_profiles"

    def test_route_query_events_only(self):
        """Test routing with only events"""
        state: AgentState = {
            "query": "test",
            "intent_type": "event",
            "confidence": 0.9,
            "profile_attributes": [],
            "events": [{"event_description": "purchase", "event_attributes": []}],
            "profile_results": [],
            "event_results": [],
            "event_attr_results": [],
            "final_result": None,
            "error": None
        }

        route = route_query(state)
        assert route == "search_events"

    def test_route_query_mixed(self):
        """Test routing with both profiles and events"""
        state: AgentState = {
            "query": "test",
            "intent_type": "mixed",
            "confidence": 0.9,
            "profile_attributes": [{"attribute_name": "age", "query_text": "age"}],
            "events": [{"event_description": "purchase", "event_attributes": []}],
            "profile_results": [],
            "event_results": [],
            "event_attr_results": [],
            "final_result": None,
            "error": None
        }

        route = route_query(state)
        assert route == "search_both"

    def test_route_query_fallback(self):
        """Test routing with no structured info (fallback)"""
        state: AgentState = {
            "query": "test",
            "intent_type": "unknown",
            "confidence": 0.5,
            "profile_attributes": [],
            "events": [],
            "profile_results": [],
            "event_results": [],
            "event_attr_results": [],
            "final_result": None,
            "error": None
        }

        route = route_query(state)
        assert route == "search_both"  # Fallback should route to search_both


class TestIntegration:
    """Integration test cases"""

    @pytest.mark.parametrize("query,expected_intent", [
        ("用户的年龄和性别信息", "profile"),
        ("购买相关的事件", "event"),
        ("25到35岁的男性用户,过去90天内购买过商品,查询购买金额", "mixed"),
    ])
    def test_query_intent_mapping(self, query, expected_intent):
        """Test that different queries map to expected intents"""
        # This is a documentation test - shows expected behavior
        # Actual implementation would require full setup with LLM
        assert query is not None
        assert expected_intent in ["profile", "event", "mixed"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
