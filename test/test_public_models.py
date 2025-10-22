#!/usr/bin/env python
"""
Test script for Volcano Engine Public Models
火山引擎公共模型测试脚本
"""
import json
import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.llm_extractor import VolcanoLLMExtractor
from src.volcano_models import (
    CHAT_MODELS,
    DEFAULT_MODEL,
    PRODUCTION_MODEL,
    get_model_info,
    get_model_description,
    list_available_models,
    recommend_model,
    ModelCategory,
)


def test_model_utilities():
    """Test model utility functions"""
    print("Testing Model Utility Functions")
    print("=" * 80)

    # List all available models
    print("\n[1] Available Chat Models:")
    for model in CHAT_MODELS:
        info = get_model_info(model)
        if info:
            print(f"   - {info.display_name} ({model}): {info.context_window:,} tokens, Cost: {info.cost_level}")

    # Test model recommendation
    print("\n[2] Model Recommendations:")
    scenarios = [
        ("general", 0, True),
        ("extraction", 0, False),
        ("long_document", 50000, True),
        ("very_long_document", 200000, False),
    ]

    for task, context_len, cost_sensitive in scenarios:
        recommended = recommend_model(task, context_len, cost_sensitive)
        print(f"   - Task: {task}, Context: {context_len:,}, Cost-sensitive: {cost_sensitive}")
        print(f"     Recommended: {get_model_description(recommended)}")

    print("\n[3] Model Categories:")
    for category in ModelCategory:
        models = list_available_models(category)
        print(f"   - {category.value}: {len(models)} models")

    print("-" * 80)


def test_public_model_extraction(model_name: str = None):
    """Test extraction with a public model"""

    # Get API key
    api_key = os.getenv("VOLCANO_API_KEY")
    if not api_key:
        print("[ERROR] Please configure VOLCANO_API_KEY in .env file")
        return False

    # Use provided model or default
    if not model_name:
        model_name = os.getenv("VOLCANO_MODEL", DEFAULT_MODEL)

    print(f"\nTesting Public Model: {model_name}")
    print("=" * 80)

    # Get model info
    model_info = get_model_info(model_name)
    if model_info:
        print(f"Model Details: {get_model_description(model_name)}")
    else:
        print(f"Using model: {model_name}")

    # Initialize extractor
    print("\nInitializing extractor...")
    try:
        extractor = VolcanoLLMExtractor(
            api_key=api_key,
            model=model_name
        )
        print("[OK] Extractor initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        return False

    # Test queries
    test_queries = [
        "查找年龄大于30岁的用户",
        "统计2024年的订单总额",
        "Find users registered in Beijing",
    ]

    print("\nTesting extraction:")
    print("-" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}]: {query}")

        try:
            start_time = time.time()
            result = extractor.extract(query)
            elapsed = time.time() - start_time

            print(f"   Time: {elapsed:.2f}s")
            print(f"   Intent: {result.intent_type} (confidence: {result.intent_confidence:.2f})")

            if result.key_terms:
                print(f"   Keywords: {', '.join(result.key_terms[:5])}")

            if result.entities:
                print(f"   Entities: {len(result.entities)} found")
                for entity in result.entities[:3]:
                    print(f"     - {entity.get('type', '?')}: {entity.get('value', '?')}")

        except Exception as e:
            print(f"   [ERROR]: {e}")

    return True


def compare_models():
    """Compare different public models"""

    api_key = os.getenv("VOLCANO_API_KEY")
    if not api_key:
        print("[ERROR] Please configure VOLCANO_API_KEY")
        return

    print("\nComparing Public Models")
    print("=" * 80)

    # Models to compare
    models_to_test = [
        "doubao-lite-32k",
        "doubao-pro-32k",
    ]

    query = "查找所有VIP会员的消费记录并计算总额"
    print(f"\nTest Query: {query}")
    print("-" * 60)

    results = {}

    for model in models_to_test:
        print(f"\nTesting {model}...")

        try:
            extractor = VolcanoLLMExtractor(
                api_key=api_key,
                model=model,
                temperature=0.1
            )

            start_time = time.time()
            result = extractor.extract(query)
            elapsed = time.time() - start_time

            results[model] = {
                "time": elapsed,
                "intent": result.intent_type,
                "confidence": result.intent_confidence,
                "entities": len(result.entities),
                "keywords": len(result.key_terms)
            }

            print(f"   [OK] Completed in {elapsed:.2f}s")

        except Exception as e:
            print(f"   [ERROR]: {e}")
            results[model] = {"error": str(e)}

    # Display comparison
    print("\n" + "=" * 80)
    print("Comparison Results:")
    print("-" * 60)

    for model, result in results.items():
        info = get_model_info(model)
        print(f"\n{model}:")
        if info:
            print(f"   Cost Level: {info.cost_level}")
            print(f"   Context: {info.context_window:,} tokens")

        if "error" in result:
            print(f"   Error: {result['error']}")
        else:
            print(f"   Response Time: {result['time']:.2f}s")
            print(f"   Intent: {result['intent']} ({result['confidence']:.2f})")
            print(f"   Entities: {result['entities']}, Keywords: {result['keywords']}")


def test_custom_vs_public():
    """Test custom endpoint vs public model"""

    api_key = os.getenv("VOLCANO_API_KEY")
    endpoint_id = os.getenv("VOLCANO_ENDPOINT_ID", "")
    public_model = os.getenv("VOLCANO_MODEL", DEFAULT_MODEL)

    if not api_key:
        print("[ERROR] Please configure VOLCANO_API_KEY")
        return

    print("\nTesting Custom Endpoint vs Public Model")
    print("=" * 80)

    # Test public model
    print(f"\n[1] Public Model: {public_model}")
    try:
        extractor = VolcanoLLMExtractor(
            api_key=api_key,
            model=public_model
        )
        print("   [OK] Public model initialized")
    except Exception as e:
        print(f"   [ERROR]: {e}")

    # Test custom endpoint if configured
    if endpoint_id and endpoint_id.startswith("ep-"):
        print(f"\n[2] Custom Endpoint: {endpoint_id}")
        try:
            extractor = VolcanoLLMExtractor(
                api_key=api_key,
                model=endpoint_id
            )
            print("   [OK] Custom endpoint initialized")
        except Exception as e:
            print(f"   [ERROR]: {e}")
    else:
        print("\n[2] Custom Endpoint: Not configured")


def main():
    """Main test function"""
    print("Volcano Engine Public Models Test Suite")
    print("=" * 80)

    # Check API key
    if not os.getenv("VOLCANO_API_KEY"):
        print("\n[WARNING] VOLCANO_API_KEY not configured")
        print("Please set VOLCANO_API_KEY in your .env file to run tests")
        print("\nYou can still view available models:")

    # Test model utilities (doesn't need API key)
    test_model_utilities()

    # Run extraction tests if API key is available
    if os.getenv("VOLCANO_API_KEY"):
        print("\n" + "=" * 80)

        # Test with default model
        if test_public_model_extraction():
            print("\n[SUCCESS] Basic extraction test passed")

        # Optional: Compare models
        print("\nWould you like to compare different models? (This will make multiple API calls)")
        print("Uncomment the line below to run comparison:")
        # compare_models()

        # Test custom vs public
        test_custom_vs_public()

    print("\n" + "=" * 80)
    print("Test suite completed!")


if __name__ == "__main__":
    main()