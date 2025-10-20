"""
LangGraph Agent Command Line Interface

This script provides a CLI for running the LangGraph-based natural language query agent.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import MilvusConfig, CollectionConfig, EmbeddingConfig
from src.milvus_client import MilvusClient
from src.embedding_manager import EmbeddingManager
from src.llm_extractor import VolcanoLLMExtractor
from src.langgraph_agent import create_agent_graph
from src.langgraph_agent.graph import run_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config():
    """Load configuration from environment variables"""
    load_dotenv()

    # Milvus configuration
    milvus_config = MilvusConfig(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=int(os.getenv("MILVUS_PORT", "19530")),
        alias=os.getenv("MILVUS_ALIAS", "default"),
        timeout=int(os.getenv("MILVUS_TIMEOUT", "10"))
    )

    # Collection configuration
    collection_config = CollectionConfig(
        metadata_collection=os.getenv("METADATA_COLLECTION", "Pampers_metadata"),
        vector_dimension=int(os.getenv("VECTOR_DIM", "1024"))
    )

    # LLM configuration
    volcano_api_key = os.getenv("VOLCANO_API_KEY")
    if not volcano_api_key:
        raise ValueError("VOLCANO_API_KEY environment variable is required")

    volcano_model = os.getenv("VOLCANO_MODEL", "doubao-pro-32k")
    prompt_file = os.getenv("PROMPT_FILE", "prompt.txt")

    return milvus_config, collection_config, volcano_api_key, volcano_model, prompt_file


def initialize_components(milvus_config, collection_config, volcano_api_key, volcano_model, prompt_file):
    """Initialize all components"""
    logger.info("Initializing components...")

    # Initialize Milvus client
    logger.info(f"Connecting to Milvus: {milvus_config.host}:{milvus_config.port}")
    milvus_client = MilvusClient(milvus_config, collection_config)

    # Initialize embedding manager
    logger.info("Initializing embedding manager (BGE-M3)...")
    embedding_config = EmbeddingConfig(model_name="BAAI/bge-m3")
    embedding_manager = EmbeddingManager(config=embedding_config)

    # Initialize LLM extractor
    logger.info(f"Initializing LLM extractor (model: {volcano_model})...")
    llm_extractor = VolcanoLLMExtractor(
        api_key=volcano_api_key,
        model=volcano_model,
        prompt_file_path=prompt_file,
        max_tokens=2048,
        temperature=0.1
    )

    logger.info("All components initialized successfully")

    return milvus_client, embedding_manager, llm_extractor


def print_results(final_state):
    """Print results in a formatted way"""
    final_result = final_state.get("final_result", {})
    print(final_result)
    print("\n" + "="*80)
    print("查询结果")
    print("="*80)

    print(f"\n原始查询: {final_result.get('query', '')}")
    print(f"意图类型: {final_result.get('intent_type', 'unknown')}")
    print(f"置信度: {final_result.get('confidence_score', 0.0):.2f}")
    print(f"执行时间: {final_result.get('execution_time', 0.0):.2f}秒")

    # Print summary
    print(f"\n摘要: {final_result.get('summary', '')}")

    # Print profile attributes
    profile_attrs = final_result.get('profile_attributes', [])
    if profile_attrs:
        print(f"\n用户属性 ({len(profile_attrs)} 个):")
        for i, attr in enumerate(profile_attrs, 1):
            print(f"  {i}. {attr['source_name']} (ID: {attr['idname']})")
            print(f"     原始查询: {attr.get('original_query', '')}")
            print(f"     匹配分数: {attr['score']} ({attr['confidence_level']})")

    # Print events
    events = final_result.get('events', [])
    if events:
        print(f"\n事件 ({len(events)} 个):")
        for i, evt in enumerate(events, 1):
            print(f"  {i}. {evt['source_name']} (ID: {evt['idname']})")
            print(f"     原始查询: {evt.get('original_query', '')}")
            print(f"     匹配分数: {evt['score']} ({evt['confidence_level']})")

    # Print event attributes
    event_attrs = final_result.get('event_attributes', [])
    if event_attrs:
        print(f"\n事件属性 ({len(event_attrs)} 个):")
        for i, attr in enumerate(event_attrs, 1):
            print(f"  {i}. {attr['source_name']} (ID: {attr['idname']})")
            print(f"     所属事件: {attr.get('event_name', '')} ({attr.get('event_idname', '')})")
            print(f"     原始查询: {attr.get('original_query', '')}")
            print(f"     匹配分数: {attr['score']} ({attr['confidence_level']})")

    # Print ambiguities if any
    if final_result.get('has_ambiguity'):
        print(f"\n[WARNING] 检测到歧义:")
        for amb in final_result.get('ambiguous_options', []):
            print(f"\n  类别: {amb['category']}")
            print(f"  查询: {amb['original_query']}")
            print(f"  候选项:")
            for candidate in amb['candidates']:
                print(f"    - {candidate['source_name']} (ID: {candidate['idname']}, 分数: {candidate['score']})")

    # Print error if any
    error = final_state.get('error') or final_result.get('error')
    if error:
        print(f"\n[ERROR] {error}")

    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="LangGraph Agent for Natural Language Query Processing"
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Natural language query to process"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load configuration
        milvus_config, collection_config, volcano_api_key, volcano_model, prompt_file = load_config()

        # Initialize components
        milvus_client, embedding_manager, llm_extractor = initialize_components(
            milvus_config, collection_config, volcano_api_key, volcano_model, prompt_file
        )

        # Create LangGraph agent
        logger.info("Creating LangGraph workflow...")
        agent_app = create_agent_graph(
            llm_extractor=llm_extractor,
            milvus_client=milvus_client,
            embedding_manager=embedding_manager,
            similarity_threshold=0.65,
            ambiguity_threshold=0.75
        )

        # Run agent
        if args.interactive:
            # Interactive mode
            print("\nLangGraph Agent - 交互模式")
            print("输入查询 (输入 'quit' 或 'exit' 退出):\n")

            while True:
                try:
                    query = input("查询> ").strip()
                    if query.lower() in ['quit', 'exit', 'q']:
                        break

                    if not query:
                        continue

                    # Run agent
                    final_state = run_agent(agent_app, query)

                    # Print results
                    if args.json:
                        print(json.dumps(final_state.get("final_result", {}), ensure_ascii=False, indent=2))
                    else:
                        print_results(final_state)

                except KeyboardInterrupt:
                    print("\n\n中断退出")
                    break
                except Exception as e:
                    logger.error(f"Error processing query: {e}", exc_info=True)
                    print(f"\n[ERROR] 处理查询时出错: {e}\n")

        elif args.query:
            # Single query mode
            final_state = run_agent(agent_app, args.query)

            # Print results
            if args.json:
                print(json.dumps(final_state.get("final_result", {}), ensure_ascii=False, indent=2))
            else:
                print_results(final_state)

        else:
            parser.print_help()
            return 1

        # Cleanup
        milvus_client.disconnect()
        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n[FATAL ERROR] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
