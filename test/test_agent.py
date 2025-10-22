#!/usr/bin/env python
"""
Test script for the Natural Language Query Agent
"""
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src import (
    NaturalLanguageQueryAgent,
    setup_logging,
    Timer,
    performance_monitor
)


def test_basic_functionality():
    """Test basic agent functionality"""
    print("🧪 测试基本功能...")

    # Setup logging
    setup_logging(log_level="INFO", console_output=True, colored_output=True)

    # Test queries
    test_queries = [
        "用户的年龄信息",
        "会员积分相关的事件",
        "购买订单中的金额字段",
        "会员绑定渠道",
        "积分变化时间",
        "member_id",
        "points",
        "兑换礼品"
    ]

    try:
        with Timer("Agent Initialization"):
            agent = NaturalLanguageQueryAgent()

        if not agent.is_ready():
            print("❌ 代理未就绪")
            return False

        print("✅ 代理初始化成功")

        # Show agent status
        status = agent.get_agent_status()
        print(f"\n📊 代理状态: {status['ready']}")
        print(f"🔗 Milvus: {status['config']['milvus_host']}:{status['config']['milvus_port']}")
        print(f"🧠 模型: {status['config']['embedding_model']}")

        print(f"\n🔍 开始测试 {len(test_queries)} 个查询...")

        # Test each query
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"测试 {i}/{len(test_queries)}: {query}")
            print('='*60)

            try:
                with Timer(f"Query {i}"):
                    result = agent.process_query(query)

                print(f"📝 查询: {result.query}")
                print(f"⏱️ 执行时间: {result.execution_time:.2f}秒")
                print(f"📊 总结果数: {result.total_results}")
                print(f"🎯 置信度: {result.confidence_score:.2f}")

                if result.has_ambiguity:
                    print("⚠️ 检测到歧义")

                print(f"💡 摘要: {result.summary}")

                # Show top results from each category
                if result.profile_attributes:
                    print(f"\n👤 个人属性 ({len(result.profile_attributes)}):")
                    for j, attr in enumerate(result.profile_attributes[:3], 1):
                        print(f"  {j}. {attr.source_name}.{attr.field_name} ({attr.score:.3f})")

                if result.events:
                    print(f"\n🎬 事件 ({len(result.events)}):")
                    for j, event in enumerate(result.events[:3], 1):
                        print(f"  {j}. {event.event_name} ({event.score:.3f})")

                if result.event_attributes:
                    print(f"\n🎯 事件属性 ({len(result.event_attributes)}):")
                    for j, attr in enumerate(result.event_attributes[:3], 1):
                        print(f"  {j}. {attr.source_name}.{attr.field_name} ({attr.score:.3f})")

                print("✅ 查询测试成功")

            except Exception as e:
                print(f"❌ 查询测试失败: {e}")
                import traceback
                traceback.print_exc()

        # Show performance metrics
        print(f"\n📈 性能统计:")
        metrics = performance_monitor.get_metrics()
        for operation, stats in metrics.items():
            print(f"  {operation}: {stats['count']} 次, 平均 {stats['avg_time']:.2f}s")

        # Test cache
        print(f"\n💾 测试缓存...")
        agent.clear_cache()
        cache_stats = agent.embedding_manager.get_cache_stats()
        print(f"缓存状态: {cache_stats}")

        print("\n✅ 所有测试完成")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if 'agent' in locals():
            agent.shutdown()


def test_error_handling():
    """Test error handling"""
    print("\n🧪 测试错误处理...")

    try:
        # Test with invalid configuration
        from src.config import MilvusConfig, CollectionConfig, EmbeddingConfig, SearchConfig, AgentConfig

        # Create invalid config
        invalid_config = AgentConfig(
            milvus=MilvusConfig(host="invalid_host", port="99999"),
            collections=CollectionConfig(),
            embedding=EmbeddingConfig(),
            search=SearchConfig()
        )

        print("尝试使用无效配置...")
        agent = None
        try:
            agent = NaturalLanguageQueryAgent(invalid_config)
            print("❌ 应该失败但没有失败")
            return False
        except Exception as e:
            print(f"✅ 正确捕获错误: {e}")

        print("✅ 错误处理测试通过")
        return True

    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        return False

    finally:
        if 'agent' in locals() and agent:
            agent.shutdown()


def main():
    """Main test function"""
    print("🚀 启动自然语言查询代理测试")
    print("="*80)

    success = True

    # Test basic functionality
    if not test_basic_functionality():
        success = False

    # Test error handling
    if not test_error_handling():
        success = False

    print("\n" + "="*80)
    if success:
        print("✅ 所有测试通过！")
        sys.exit(0)
    else:
        print("❌ 部分测试失败")
        sys.exit(1)


if __name__ == "__main__":
    main()