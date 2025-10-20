#!/usr/bin/env python
"""
Interactive Natural Language Query Agent for Pampers Metadata
"""
import sys
import logging
import time
from typing import Optional
import argparse
from datetime import datetime

from src import NaturalLanguageQueryAgent, CONFIG


class InteractiveQueryAgent:
    """Interactive command-line interface for the Natural Language Query Agent"""

    def __init__(self, debug: bool = False):
        self.agent = None
        self.debug = debug
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_level = logging.DEBUG if self.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )

        # Reduce noise from external libraries
        logging.getLogger('pymilvus').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)

    def initialize_agent(self) -> bool:
        """Initialize the agent with error handling"""
        try:
            print("[*] 正在初始化自然语言查询代理...")
            print(f"   - Milvus服务器: {CONFIG.milvus.host}:{CONFIG.milvus.port}")
            print(f"   - 嵌入模型: {CONFIG.embedding.model_name}")
            print(f"   - 缓存启用: {CONFIG.enable_cache}")

            self.agent = NaturalLanguageQueryAgent()

            if not self.agent.is_ready():
                print("[!] 代理初始化失败")
                return False

            print("[+] 代理初始化成功！")
            return True

        except Exception as e:
            print(f"[-] 初始化失败: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return False

    def show_welcome_message(self) -> None:
        """Display welcome message and instructions"""
        print("\n" + "="*80)
        print("[*] Pampers 自然语言查询代理")
        print("="*80)
        print("这个代理可以帮助您通过自然语言查询Pampers数据库中的属性和事件信息。")
        print("\n[SEARCH] 查询功能:")
        print("  1. 个人属性查询 - 查找用户/会员相关的属性字段")
        print("  2. 事件查询 - 查找相关的业务事件")
        print("  3. 事件属性查询 - 在特定事件中查找相关属性")
        print("\n[TIP] 示例查询:")
        print("  - '用户的年龄信息'")
        print("  - '会员积分相关的事件'")
        print("  - '购买订单中的金额字段'")
        print("  - '会员绑定渠道'")
        print("\n[LIST] 命令:")
        print("  help    - 显示帮助信息")
        print("  status  - 显示代理状态")
        print("  clear   - 清除缓存")
        print("  exit    - 退出程序")
        print("="*80)

    def show_help(self) -> None:
        """Display help information"""
        print("\n[HELP] 帮助信息:")
        print("="*50)
        print("[SEARCH] 查询类型:")
        print("  - 个人属性: 查询用户、会员的基本信息字段")
        print("  - 事件信息: 查询系统中的业务事件")
        print("  - 事件属性: 查询特定事件中的字段信息")
        print("\n[TIP] 查询技巧:")
        print("  - 使用自然语言描述您要查找的内容")
        print("  - 可以使用中文或英文")
        print("  - 包含关键词会提高查询准确性")
        print("  - 系统会智能判断查询意图并提供相关结果")
        print("\n[STATUS] 结果说明:")
        print("  [HIGH] 高置信度 (相似度 > 0.8)")
        print("  [MED] 中等置信度 (相似度 0.6-0.8)")
        print("  [LOW] 低置信度 (相似度 < 0.6)")
        print("  [WARN] 可能有歧义 - 系统检测到多个相似结果")
        print("="*50)

    def show_status(self) -> None:
        """Display agent status"""
        if not self.agent:
            print("[-] 代理未初始化")
            return

        try:
            status = self.agent.get_agent_status()
            print("\n[STATUS] 代理状态:")
            print("="*50)
            print(f"[ROBOT] 代理状态: {'[+] 就绪' if status['ready'] else '[-] 未就绪'}")
            print(f"[CONNECT] Milvus连接: {status['config']['milvus_host']}:{status['config']['milvus_port']}")
            print(f"[MODEL] 嵌入模型: {status['config']['embedding_model']}")
            print(f"[CACHE] 缓存状态: {'[+] 启用' if status['config']['cache_enabled'] else '[-] 禁用'}")

            if 'embedding' in status:
                embedding_info = status['embedding']
                print(f"[START] 设备: {embedding_info.get('device', 'unknown')}")
                print(f"[PKG] 模型加载: {'[+] 已加载' if embedding_info.get('model_loaded', False) else '[-] 未加载'}")

            if 'cache' in status and status['cache']['cache_enabled']:
                cache_info = status['cache']
                print(f"[CACHE] 缓存大小: {cache_info.get('cache_size', 0)}/{cache_info.get('cache_max_size', 0)}")

            if 'collections' in status:
                collections = status['collections']
                print(f"\n[LIST] 数据集合:")
                for name, info in collections.items():
                    if info:
                        print(f"  - {name}: {info.get('num_entities', 0)} 条记录")
                    else:
                        print(f"  - {name}: [-] 不可用")

            print("="*50)

        except Exception as e:
            print(f"[-] 获取状态失败: {e}")

    def clear_cache(self) -> None:
        """Clear agent cache"""
        if not self.agent:
            print("[-] 代理未初始化")
            return

        try:
            self.agent.clear_cache()
            print("[+] 缓存已清除")
        except Exception as e:
            print(f"[-] 清除缓存失败: {e}")

    def format_results(self, analysis_result) -> None:
        """Format and display analysis results"""
        print(f"\n[QUERY] 查询: {analysis_result.query}")
        print(f"[TIME] 执行时间: {analysis_result.execution_time:.2f}秒")
        print(f"[STATUS] 总结果数: {analysis_result.total_results}")
        print(f"[ATTR] 整体置信度: {analysis_result.confidence_score:.2f}")

        if analysis_result.has_ambiguity:
            print("[WARN] 检测到查询歧义 - 请考虑更具体的查询")

        print(f"\n[TIP] 摘要: {analysis_result.summary}")

        # Display profile attributes
        if analysis_result.profile_attributes:
            print(f"\n[PROFILE] 相关个人属性 ({len(analysis_result.profile_attributes)} 个):")
            print("-" * 60)
            for i, result in enumerate(analysis_result.profile_attributes[:5], 1):
                confidence_icon = {'high': '[HIGH]', 'medium': '[MED]', 'low': '[LOW]'}.get(result.confidence_level, '[?]')
                ambiguous_flag = ' [WARN]' if result.is_ambiguous else ''
                print(f"{i}. {confidence_icon} {result.source_name}.{result.field_name} "
                      f"(相似度: {result.score:.3f}){ambiguous_flag}")
                print(f"   [INFO] {result.explanation}")

        # Display events
        if analysis_result.events:
            print(f"\n[EVENT] 相关事件 ({len(analysis_result.events)} 个):")
            print("-" * 60)
            for i, result in enumerate(analysis_result.events[:5], 1):
                confidence_icon = {'high': '[HIGH]', 'medium': '[MED]', 'low': '[LOW]'}.get(result.confidence_level, '[?]')
                ambiguous_flag = ' [WARN]' if result.is_ambiguous else ''
                print(f"{i}. {confidence_icon} {result.event_name} "
                      f"(相似度: {result.score:.3f}){ambiguous_flag}")
                print(f"   [INFO] {result.explanation}")

        # Display event attributes
        if analysis_result.event_attributes:
            print(f"\n[ATTR] 相关事件属性 ({len(analysis_result.event_attributes)} 个):")
            print("-" * 60)
            for i, result in enumerate(analysis_result.event_attributes[:10], 1):
                confidence_icon = {'high': '[HIGH]', 'medium': '[MED]', 'low': '[LOW]'}.get(result.confidence_level, '[?]')
                ambiguous_flag = ' [WARN]' if result.is_ambiguous else ''
                print(f"{i}. {confidence_icon} {result.source_name}.{result.field_name} "
                      f"(相似度: {result.score:.3f}){ambiguous_flag}")
                print(f"   [INFO] {result.explanation}")

        # Show recommendation if no good results
        if analysis_result.total_results == 0:
            print("\n[TIP] 建议:")
            print("  - 尝试使用不同的关键词")
            print("  - 检查查询语句的拼写")
            print("  - 使用更通用的描述")

        print("-" * 80)

    def process_query(self, query: str) -> None:
        """Process a user query"""
        if not self.agent:
            print("[-] 代理未初始化")
            return

        try:
            print(f"\n[SEARCH] 正在处理查询...")
            analysis_result = self.agent.process_query(query)
            self.format_results(analysis_result)

        except Exception as e:
            print(f"[-] 查询处理失败: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

    def run_interactive(self) -> None:
        """Run interactive query loop"""
        if not self.initialize_agent():
            sys.exit(1)

        self.show_welcome_message()

        while True:
            try:
                # Get user input
                user_input = input("\n[ROBOT] 请输入您的查询 (或输入 'help' 查看帮助): ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n[BYE] 再见!")
                    break

                elif user_input.lower() == 'help':
                    self.show_help()

                elif user_input.lower() == 'status':
                    self.show_status()

                elif user_input.lower() == 'clear':
                    self.clear_cache()

                else:
                    # Process as query
                    self.process_query(user_input)

            except KeyboardInterrupt:
                print("\n\n[BYE] 程序被中断，再见!")
                break

            except Exception as e:
                print(f"[-] 发生错误: {e}")
                if self.debug:
                    import traceback
                    traceback.print_exc()

        # Cleanup
        if self.agent:
            self.agent.shutdown()

    def run_single_query(self, query: str) -> None:
        """Run a single query and exit"""
        if not self.initialize_agent():
            sys.exit(1)

        try:
            self.process_query(query)
        finally:
            if self.agent:
                self.agent.shutdown()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Pampers 自然语言查询代理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python natural_language_agent.py                           # 交互模式
  python natural_language_agent.py -q "用户的年龄信息"        # 单次查询
  python natural_language_agent.py --debug                   # 调试模式
        """
    )

    parser.add_argument(
        '-q', '--query',
        type=str,
        help='单次查询模式 - 执行单个查询后退出'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式 - 显示详细日志'
    )

    args = parser.parse_args()

    # Create and run agent
    agent = InteractiveQueryAgent(debug=args.debug)

    if args.query:
        # Single query mode
        agent.run_single_query(args.query)
    else:
        # Interactive mode
        agent.run_interactive()


if __name__ == "__main__":
    main()