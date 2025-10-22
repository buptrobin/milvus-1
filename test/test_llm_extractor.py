#!/usr/bin/env python
"""
Test script for Volcano Engine (Doubao) LLM Extractor
"""
import json
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.llm_extractor import VolcanoLLMExtractor


def test_llm_extractor():
    """Test the LLM extractor with sample queries"""

    # Get configuration from environment
    api_key = os.getenv("VOLCANO_API_KEY")
    model = os.getenv("VOLCANO_MODEL") or os.getenv("VOLCANO_ENDPOINT_ID")

    if not api_key:
        print("[ERROR] Please configure VOLCANO_API_KEY in .env file")
        print("   Refer to .env.example for configuration template")
        return

    if not model:
        print("[INFO] No model configured, using default: doubao-lite-32k")
        model = "doubao-lite-32k"

    # Initialize extractor
    print(f"Initializing Volcano Engine (Doubao) LLM Extractor with model: {model}")
    try:
        extractor = VolcanoLLMExtractor(
            api_key=api_key,
            model=model
        )
        print("[OK] Initialization successful\n")
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        return

    # Test queries (Chinese and English)
    test_queries = [
        "查找年龄大于30岁的会员信息",
        "统计2024年的购买订单数量",
        "用户张三的积分历史记录",
        "会员绑定渠道是微信的用户",
        "最近7天内有过购买行为的用户",
        "Find users with age greater than 30",
        "Count purchase orders in 2024",
    ]

    print("=" * 80)
    print("Testing Information Extraction")
    print("=" * 80)

    for i, query in enumerate(test_queries, 1):
        print(f"\n[Test {i}]: {query}")
        print("-" * 60)

        try:
            # Extract information
            result = extractor.extract(query)

            # Display results
            print(f"[Intent]: {result.intent_type} (confidence: {result.intent_confidence:.2f})")

            if result.entities:
                print(f"[Entities] ({len(result.entities)} found):")
                for entity in result.entities:
                    print(f"   - {entity['type']}: {entity['value']}")

            if result.key_terms:
                print(f"[Keywords]: {', '.join(result.key_terms)}")

            if result.filters:
                print(f"[Filters]:")
                for filter_item in result.filters:
                    print(f"   - {filter_item.get('field', '?')} {filter_item.get('operator', '?')} {filter_item.get('value', '?')}")

            if result.temporal_info:
                print(f"[Temporal Info]: {json.dumps(result.temporal_info, ensure_ascii=False)}")

            if result.numerical_info:
                print(f"[Numerical Info]:")
                for num_info in result.numerical_info:
                    print(f"   - {num_info['value']} {num_info.get('unit', '')} ({num_info.get('context', '')})")

        except Exception as e:
            print(f"[ERROR] Extraction failed: {e}")

    print("\n" + "=" * 80)
    print("Test completed!")


def test_custom_prompt():
    """Test with custom extraction prompt"""

    api_key = os.getenv("VOLCANO_API_KEY")
    model = os.getenv("VOLCANO_MODEL") or os.getenv("VOLCANO_ENDPOINT_ID")

    if not api_key:
        print("[ERROR] Please configure API key")
        return

    if not model:
        model = "doubao-lite-32k"

    # Custom prompt template (Chinese)
    custom_prompt_cn = """请分析以下查询并提取关键信息：

查询: {query}

请返回JSON格式，包含以下字段：
- intent_type: 查询意图（search/filter/aggregate/analyze）
- intent_confidence: 意图置信度
- key_terms: 关键词列表
- entities: 实体列表，每个包含type和value

示例返回格式：
{{
    "intent_type": "search",
    "intent_confidence": 0.9,
    "key_terms": ["用户", "年龄"],
    "entities": [{{"type": "attribute", "value": "年龄"}}]
}}
"""

    # Custom prompt template (English)
    custom_prompt_en = """Analyze the following query and extract key information:

Query: {query}

Return JSON format with the following fields:
- intent_type: Query intent (search/filter/aggregate/analyze)
- intent_confidence: Confidence score
- key_terms: List of keywords
- entities: List of entities, each containing type and value

Example return format:
{{
    "intent_type": "search",
    "intent_confidence": 0.9,
    "key_terms": ["user", "age"],
    "entities": [{{"type": "attribute", "value": "age"}}]
}}
"""

    print("\n" + "=" * 80)
    print("Testing with custom prompt template")
    print("=" * 80)

    try:
        # Test with Chinese prompt
        extractor_cn = VolcanoLLMExtractor(
            api_key=api_key,
            model=model,
            extraction_prompt_template=custom_prompt_cn
        )

        query = "查询所有VIP会员的消费总额"
        print(f"\n[Test with Chinese prompt]")
        print(f"Query: {query}")

        result = extractor_cn.extract(query)
        print(f"Extraction result:")
        print(f"  Intent: {result.intent_type} ({result.intent_confidence:.2f})")
        print(f"  Keywords: {result.key_terms}")
        print(f"  Entities: {result.entities}")

        # Test with English prompt
        extractor_en = VolcanoLLMExtractor(
            api_key=api_key,
            model=model,
            extraction_prompt_template=custom_prompt_en
        )

        query = "Query total consumption of all VIP members"
        print(f"\n[Test with English prompt]")
        print(f"Query: {query}")

        result = extractor_en.extract(query)
        print(f"Extraction result:")
        print(f"  Intent: {result.intent_type} ({result.intent_confidence:.2f})")
        print(f"  Keywords: {result.key_terms}")
        print(f"  Entities: {result.entities}")

    except Exception as e:
        print(f"[ERROR]: {e}")


if __name__ == "__main__":
    print("Volcano Engine (Doubao) LLM Information Extraction Test\n")

    # Run basic tests
    test_llm_extractor()

    # Optionally test custom prompt
    # Uncomment the line below to test custom prompts
    # test_custom_prompt()