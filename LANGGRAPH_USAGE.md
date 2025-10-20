# LangGraph Agent 使用指南

本文档说明如何使用基于 LangGraph 的自然语言查询代理。

## 概述

LangGraph Agent 是一个模块化的查询处理系统,可以:
- 理解用户的自然语言查询
- 识别查询意图(人的属性/事件/混合)
- 从 Milvus 向量数据库中检索相关信息
- 返回结构化的查询结果

## 架构

系统采用 LangGraph 工作流编排,包含以下节点:
1. **intent_classification**: 使用 LLM 理解查询并抽取结构化信息
2. **route_query**: 根据意图类型决定执行路径
3. **search_profiles**: 查询人的静态属性
4. **search_events**: 查询事件元数据
5. **search_event_attributes**: 查询事件的属性
6. **aggregate_results**: 聚合、去重、格式化结果

## 安装

依赖已在 `pyproject.toml` 中定义,使用 uv 安装:

```bash
uv sync
```

## 配置

在项目根目录创建 `.env` 文件:

```env
# Milvus 配置
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_ALIAS=default
MILVUS_TIMEOUT=10

# Collection 配置
METADATA_COLLECTION=Pampers_metadata
VECTOR_DIM=1024

# LLM 配置
VOLCANO_API_KEY=your_api_key_here
VOLCANO_MODEL=doubao-pro-32k
PROMPT_FILE=prompt.txt
```

## 使用方法

### 1. 命令行模式

**单次查询:**

```bash
uv run python langgraph_agent_cli.py "25到35岁的男性用户"
```

**交互模式:**

```bash
uv run python langgraph_agent_cli.py --interactive
```

**JSON 输出:**

```bash
uv run python langgraph_agent_cli.py "购买相关的事件" --json
```

**详细日志:**

```bash
uv run python langgraph_agent_cli.py "查询用户年龄" --verbose
```

### 2. 编程方式使用

```python
from src.config import MilvusConfig, CollectionConfig
from src.milvus_client import MilvusClient
from src.embedding_manager import EmbeddingManager
from src.llm_extractor import VolcanoLLMExtractor
from src.langgraph_agent import create_agent_graph
from src.langgraph_agent.graph import run_agent

# 初始化组件
milvus_config = MilvusConfig(host="localhost", port=19530)
collection_config = CollectionConfig(
    metadata_collection="Pampers_metadata",
    vector_dimension=1024
)

milvus_client = MilvusClient(milvus_config, collection_config)
embedding_manager = EmbeddingManager(model_name="BAAI/bge-m3")
llm_extractor = VolcanoLLMExtractor(
    api_key="your_api_key",
    model="doubao-pro-32k",
    prompt_file_path="prompt.txt"
)

# 创建 LangGraph Agent
agent_app = create_agent_graph(
    llm_extractor=llm_extractor,
    milvus_client=milvus_client,
    embedding_manager=embedding_manager,
    similarity_threshold=0.65,
    ambiguity_threshold=0.75
)

# 运行查询
query = "25到35岁的男性用户,过去90天内购买过商品,查询购买金额"
final_state = run_agent(agent_app, query)

# 获取结果
final_result = final_state["final_result"]
print(f"查询摘要: {final_result['summary']}")
print(f"置信度: {final_result['confidence_score']}")
print(f"用户属性数: {len(final_result['profile_attributes'])}")
print(f"事件数: {len(final_result['events'])}")
print(f"事件属性数: {len(final_result['event_attributes'])}")
```

## 输出格式

查询结果包含以下信息:

```json
{
  "query": "原始查询",
  "intent_type": "mixed",
  "profile_attributes": [
    {
      "idname": "age_group",
      "source_name": "年龄段",
      "source": "pampers_customer",
      "original_query": "25到35岁",
      "original_attribute": "年龄",
      "score": 0.85,
      "confidence_level": "high",
      "explanation": "字段描述"
    }
  ],
  "events": [
    {
      "idname": "buy_online",
      "source_name": "线上购买",
      "source": "pampers_customer",
      "original_query": "购买",
      "score": 0.90,
      "confidence_level": "high",
      "explanation": "事件描述"
    }
  ],
  "event_attributes": [
    {
      "idname": "purchase_amount",
      "source_name": "购买金额",
      "event_idname": "buy_online",
      "event_name": "线上购买",
      "original_query": "购买金额",
      "score": 0.88,
      "confidence_level": "high",
      "explanation": "属性描述"
    }
  ],
  "summary": "已识别: 年龄段(查询条件:25到35岁), 事件:线上购买, 属性:购买金额",
  "total_results": 3,
  "confidence_score": 0.87,
  "has_ambiguity": false,
  "ambiguous_options": [],
  "execution_time": 1.23
}
```

## 测试

运行测试用例:

```bash
uv run pytest test/test_langgraph_agent.py -v
```

## 测试用例

设计文档中定义的测试用例:

### 测试用例 1: 纯人属性查询
```
查询: "用户的年龄和性别信息"
预期意图: profile
```

### 测试用例 2: 纯事件查询
```
查询: "购买相关的事件"
预期意图: event
```

### 测试用例 3: 混合查询
```
查询: "25到35岁的男性用户,过去90天内购买过商品,查询购买金额"
预期意图: mixed
```

## 性能指标

根据设计文档的预期性能:
- LLM 调用: ~1-2s
- 向量嵌入: ~0.1-0.3s (批量)
- Milvus 查询: ~0.1-0.2s (每次)
- **总耗时**: ~2-3s (端到端)

## 优化策略

系统已实现以下优化:
1. **向量缓存**: EmbeddingManager 使用缓存功能
2. **并行查询**: LangGraph 支持并行节点执行
3. **批量嵌入**: 一次性生成多个查询的向量
4. **连接池**: MilvusClient 使用单例模式

## 错误处理

系统在以下情况提供友好的错误处理:
- LLM 调用失败
- Milvus 连接失败
- 向量嵌入失败
- 查询结果为空

错误信息会包含在 `final_result["error"]` 字段中。

## 扩展功能

系统设计支持以下扩展(参见 LANGGRAPH_AGENT_DESIGN.md):
1. **多轮对话**: 支持上下文记忆
2. **查询改写**: 如果结果不满意,自动改写查询
3. **解释生成**: 用 LLM 生成自然语言解释
4. **查询建议**: 推荐相关的查询
5. **可视化**: 使用 LangGraph Studio 可视化执行过程

## 文件结构

```
src/
├── langgraph_agent/
│   ├── __init__.py              # 模块导出
│   ├── state.py                 # State Schema 定义
│   ├── graph.py                 # LangGraph 图定义
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── intent_node.py       # intent_classification 节点
│   │   ├── profile_node.py      # search_profiles 节点
│   │   ├── event_node.py        # search_events 节点
│   │   ├── event_attr_node.py   # search_event_attributes 节点
│   │   ├── aggregate_node.py    # aggregate_results 节点
│   │   └── router.py            # route_query 路由逻辑
│   └── utils.py                 # 辅助函数 (待添加)
├── config.py                    # 配置
├── milvus_client.py             # Milvus 客户端
├── embedding_manager.py         # 嵌入管理
└── llm_extractor.py             # LLM 抽取

# 主程序
langgraph_agent_cli.py           # 命令行接口

# 测试
test/
└── test_langgraph_agent.py      # 测试用例
```

## 故障排查

### 问题 1: LLM 输出格式错误
**现象**: intent_classification 节点失败
**解决方案**:
- 检查 prompt.txt 文件是否存在
- 验证 LLM 返回的 JSON 格式
- 考虑使用设计文档中的方案B (拆分节点)

### 问题 2: Milvus 连接失败
**现象**: search 节点失败
**解决方案**:
- 检查 Milvus 服务是否运行
- 验证 .env 配置
- 确认 Collection 是否存在

### 问题 3: 向量嵌入慢
**现象**: 执行时间过长
**解决方案**:
- 使用 GPU 加速
- 启用嵌入缓存
- 考虑批量处理

## 对比旧版本

相比现有的 `NaturalLanguageQueryAgent` (src/nl_query_agent.py):

### 优点
✅ 清晰的节点和边定义,易于理解
✅ 支持可视化工作流
✅ 每个节点独立测试
✅ 支持并行执行
✅ 状态管理更加透明

### 缺点
❌ 引入新依赖 langgraph
❌ 需要学习 LangGraph API

## 参考文档

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [设计文档](LANGGRAPH_AGENT_DESIGN.md)
- [Milvus 文档](https://milvus.io/docs)
- [BGE-M3 模型](https://huggingface.co/BAAI/bge-m3)

## 许可证

与项目主体相同。
