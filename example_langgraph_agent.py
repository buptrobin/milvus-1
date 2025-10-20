"""
Example: Using LangGraph Agent for Natural Language Query

This script demonstrates how to use the LangGraph Agent programmatically.
"""

import os
import json
from dotenv import load_dotenv

from src.config import MilvusConfig, CollectionConfig, EmbeddingConfig
from src.milvus_client import MilvusClient
from src.embedding_manager import EmbeddingManager
from src.llm_extractor import VolcanoLLMExtractor
from src.langgraph_agent import create_agent_graph
from src.langgraph_agent.graph import run_agent


def main():
    """Main example function"""
    # Load environment variables
    load_dotenv()

    print("="*80)
    print("LangGraph Agent Example")
    print("="*80)

    # 1. Initialize components
    print("\n1. Initializing components...")

    milvus_config = MilvusConfig(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=int(os.getenv("MILVUS_PORT", "19530"))
    )

    collection_config = CollectionConfig(
        metadata_collection=os.getenv("METADATA_COLLECTION", "Pampers_metadata"),
        vector_dimension=int(os.getenv("VECTOR_DIM", "1024"))
    )

    milvus_client = MilvusClient(milvus_config, collection_config)
    embedding_config = EmbeddingConfig(model_name="BAAI/bge-m3")
    embedding_manager = EmbeddingManager(config=embedding_config)

    volcano_api_key = os.getenv("VOLCANO_API_KEY")
    if not volcano_api_key:
        print("[ERROR] VOLCANO_API_KEY not found in environment")
        return

    llm_extractor = VolcanoLLMExtractor(
        api_key=volcano_api_key,
        model=os.getenv("VOLCANO_MODEL", "doubao-pro-32k"),
        prompt_file_path=os.getenv("PROMPT_FILE", "prompt.txt")
    )

    print("[OK] Components initialized")

    # 2. Create LangGraph Agent
    print("\n2. Creating LangGraph workflow...")

    agent_app = create_agent_graph(
        llm_extractor=llm_extractor,
        milvus_client=milvus_client,
        embedding_manager=embedding_manager,
        similarity_threshold=0.65,
        ambiguity_threshold=0.75
    )

    print("[OK] Workflow created")

    # 3. Run test queries
    test_queries = [
        "用户的年龄和性别信息",                              # Profile query
        "购买相关的事件",                                     # Event query
        "25到35岁的男性用户,过去90天内购买过商品,查询购买金额"  # Mixed query
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test Query {i}: {query}")
        print("="*80)

        # Run agent
        final_state = run_agent(agent_app, query)
        final_result = final_state.get("final_result", {})

        # Print results
        print(f"\n意图类型: {final_result.get('intent_type', 'unknown')}")
        print(f"置信度: {final_result.get('confidence_score', 0.0):.2f}")
        print(f"执行时间: {final_result.get('execution_time', 0.0):.2f}秒")

        print(f"\n摘要: {final_result.get('summary', '')}")

        # Profile attributes
        profile_attrs = final_result.get('profile_attributes', [])
        if profile_attrs:
            print(f"\n用户属性 ({len(profile_attrs)} 个):")
            for attr in profile_attrs:
                print(f"  - {attr['source_name']} (ID: {attr['idname']}, 分数: {attr['score']})")

        # Events
        events = final_result.get('events', [])
        if events:
            print(f"\n事件 ({len(events)} 个):")
            for evt in events:
                print(f"  - {evt['source_name']} (ID: {evt['idname']}, 分数: {evt['score']})")

        # Event attributes
        event_attrs = final_result.get('event_attributes', [])
        if event_attrs:
            print(f"\n事件属性 ({len(event_attrs)} 个):")
            for attr in event_attrs:
                print(
                    f"  - {attr['source_name']} "
                    f"(事件: {attr.get('event_name', '')}, 分数: {attr['score']})"
                )

        # Ambiguities
        if final_result.get('has_ambiguity'):
            print(f"\n[WARNING] 检测到 {len(final_result['ambiguous_options'])} 个歧义")

        # Error
        if final_result.get('error'):
            print(f"\n[ERROR] {final_result['error']}")

    # 4. Cleanup
    print(f"\n{'='*80}")
    print("Cleaning up...")
    milvus_client.disconnect()
    print("[OK] Done")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n中断退出")
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
