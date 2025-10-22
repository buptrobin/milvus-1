#!/usr/bin/env python
"""
Test script to verify all imports work correctly
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all critical imports"""

    print("Testing imports...")
    print("-" * 50)

    # Test volcengine SDK import
    try:
        from volcenginesdkarkruntime import Ark
        print("[OK] volcenginesdkarkruntime.Ark imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import volcenginesdkarkruntime.Ark: {e}")
        return False

    # Test our LLM extractor
    try:
        from src.llm_extractor import VolcanoLLMExtractor, ExtractedInfo
        print("[OK] VolcanoLLMExtractor imported successfully")
        print("[OK] ExtractedInfo imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import LLM extractor: {e}")
        return False

    # Test updated config
    try:
        from src.config import VolcanoConfig, CONFIG
        print("[OK] VolcanoConfig imported successfully")
        print("[OK] CONFIG imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import config: {e}")
        return False

    # Test query processor
    try:
        from src.query_processor import QueryProcessor
        print("[OK] QueryProcessor imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import QueryProcessor: {e}")
        return False

    # Test main agent
    try:
        from src.nl_query_agent import NaturalLanguageQueryAgent
        print("[OK] NaturalLanguageQueryAgent imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import NaturalLanguageQueryAgent: {e}")
        return False

    print("-" * 50)
    print("All imports successful!")
    return True

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    print("-" * 50)

    from src.config import CONFIG

    print(f"Volcano Engine Config:")
    print(f"  - Enabled: {CONFIG.volcano.enabled}")
    print(f"  - API Key configured: {'Yes' if CONFIG.volcano.api_key else 'No'}")
    print(f"  - Endpoint configured: {'Yes' if CONFIG.volcano.endpoint_id else 'No'}")
    print(f"  - Max tokens: {CONFIG.volcano.max_tokens}")
    print(f"  - Temperature: {CONFIG.volcano.temperature}")
    print(f"  - Timeout: {CONFIG.volcano.timeout}")

    print("-" * 50)

if __name__ == "__main__":
    print("Doubao LLM Integration Test\n")

    if test_imports():
        test_config()
        print("\n[SUCCESS] All tests passed! The integration is ready to use.")
        print("\nNext steps:")
        print("1. Copy .env.example to .env")
        print("2. Add your Volcano Engine API key and endpoint ID")
        print("3. Run: uv run python test_llm_extractor.py")
    else:
        print("\n[FAILED] Some tests failed. Please check the errors above.")