# Natural Language Query Agent for Pampers Metadata

一个基于自然语言处理的智能查询代理，能够通过自然语言查询Pampers数据库中的属性和事件信息。

## 🎯 功能特性

### 核心功能
- **三阶段查询处理**:
  1. 个人属性查询 (PROFILE类型)
  2. 事件查询 (Events)
  3. 事件属性查询 (EVENT类型，基于找到的事件)

- **智能语言理解**:
  - 中英文混合查询支持
  - 查询意图自动识别
  - 关键词提取和实体识别
  - 歧义检测和澄清建议

- **高性能搜索**:
  - BGE-M3多语言嵌入模型
  - 余弦相似度语义搜索
  - 智能相似度阈值调整
  - 结果排序和过滤

## 🏗️ 系统架构

```
自然语言查询代理
├── 配置管理 (config.py)
├── Milvus客户端 (milvus_client.py)
├── 嵌入模型管理 (embedding_manager.py)
├── 查询处理器 (query_processor.py)
├── 结果分析器 (result_analyzer.py)
├── 主代理类 (nl_query_agent.py)
├── 异常处理 (exceptions.py)
├── 日志配置 (logging_config.py)
└── 工具函数 (utils.py)
```

## 🚀 快速开始

### 1. 环境准备

确保已安装所需依赖：

```bash
# 安装所有依赖
uv sync

# 安装ML依赖 (包含BGE-M3模型)
uv sync --extra ml
```

### 2. 配置环境

创建 `.env` 文件 (可选):

```env
# Milvus配置
MILVUS_HOST=172.28.9.45
MILVUS_PORT=19530
MILVUS_DATABASE=default

# 集合配置
METADATA_COLLECTION=Pampers_metadata
EVENT_COLLECTION=Pampers_Event_metadata

# 嵌入模型配置
EMBEDDING_MODEL=BAAI/bge-m3
USE_FP16=true
BATCH_SIZE=128

# 搜索配置
SIMILARITY_THRESHOLD=0.65
SCORE_GAP_THRESHOLD=0.08

# 日志配置
LOG_LEVEL=INFO
ENABLE_CACHE=true
```

### 3. 运行代理

#### 交互模式
```bash
python natural_language_agent.py
```

#### 单次查询模式
```bash
python natural_language_agent.py -q "用户的年龄信息"
```

#### 调试模式
```bash
python natural_language_agent.py --debug
```

### 4. 测试系统
```bash
python test_agent.py
```

## 💡 使用示例

### 查询示例

1. **个人属性查询**:
   - "用户的年龄信息"
   - "会员的手机号码"
   - "客户注册时间"
   - "member_id字段"

2. **事件查询**:
   - "积分相关的事件"
   - "购买订单事件"
   - "会员绑定活动"
   - "兑换礼品记录"

3. **事件属性查询**:
   - "积分变化事件中的时间字段"
   - "订单金额相关属性"
   - "兑换渠道信息"

### 编程接口

```python
from src import NaturalLanguageQueryAgent

# 初始化代理
agent = NaturalLanguageQueryAgent()

# 处理查询
result = agent.process_query("用户的年龄信息")

# 查看结果
print(f"查询: {result.query}")
print(f"置信度: {result.confidence_score}")
print(f"结果数: {result.total_results}")

# 个人属性结果
for attr in result.profile_attributes:
    print(f"属性: {attr.source_name}.{attr.field_name}")
    print(f"相似度: {attr.score}")
    print(f"描述: {attr.explanation}")

# 事件结果
for event in result.events:
    print(f"事件: {event.event_name}")
    print(f"相似度: {event.score}")

# 事件属性结果
for attr in result.event_attributes:
    print(f"事件属性: {attr.source_name}.{attr.field_name}")
    print(f"相似度: {attr.score}")
```

## ⚙️ 配置说明

### Milvus配置
- `host`: Milvus服务器地址
- `port`: Milvus服务器端口
- `database`: 数据库名称
- `timeout`: 连接超时时间

### 嵌入模型配置
- `model_name`: BGE-M3模型路径
- `use_fp16`: 是否使用半精度加速
- `batch_size`: 批处理大小
- `max_length`: 最大文本长度

### 搜索配置
- `similarity_threshold`: 相似度阈值
- `score_gap_threshold`: 歧义检测阈值
- `max_results`: 最大结果数
- `*_search_limit`: 各阶段搜索限制

## 📊 结果解释

### 置信度等级
- 🟢 **高置信度** (相似度 > 0.8): 结果高度相关
- 🟡 **中等置信度** (相似度 0.6-0.8): 结果相关性中等
- 🔴 **低置信度** (相似度 < 0.6): 结果相关性较低

### 歧义标识
- ⚠️ **可能有歧义**: 系统检测到多个相似的结果，建议细化查询

### 结果类型
- **👤 个人属性**: 用户/会员的基本信息字段
- **🎬 事件**: 系统中的业务事件
- **🎯 事件属性**: 特定事件中的字段信息

## 🔧 开发和扩展

### 添加新的查询类型

1. 在 `query_processor.py` 中添加新的关键词和模式
2. 在 `milvus_client.py` 中添加新的搜索方法
3. 在 `result_analyzer.py` 中添加新的结果分析逻辑

### 自定义嵌入模型

```python
from src import EmbeddingConfig, EmbeddingManager

config = EmbeddingConfig(
    model_name="your-custom-model",
    use_fp16=True,
    batch_size=64
)

embedding_manager = EmbeddingManager(config)
```

### 自定义配置

```python
from src import AgentConfig, load_config

# 加载自定义配置
config = load_config()

# 修改配置
config.search.similarity_threshold = 0.7
config.milvus.host = "your-milvus-host"

# 使用自定义配置创建代理
agent = NaturalLanguageQueryAgent(config)
```

## 🐛 故障排除

### 常见问题

1. **连接失败**:
   - 检查Milvus服务器是否运行
   - 验证主机和端口配置
   - 检查网络连接

2. **模型加载失败**:
   - 确保安装了ML依赖: `uv sync --extra ml`
   - 检查模型路径是否正确
   - 验证GPU/CUDA设置

3. **搜索无结果**:
   - 检查集合是否存在数据
   - 调低相似度阈值
   - 尝试不同的查询表达

4. **性能问题**:
   - 启用缓存: `ENABLE_CACHE=true`
   - 调整批处理大小
   - 使用GPU加速

### 调试模式

启用调试模式获取详细日志：

```bash
python natural_language_agent.py --debug
```

或在代码中：

```python
from src import setup_logging

setup_logging(log_level="DEBUG", console_output=True)
```

## 📈 性能优化

### 缓存优化
- 启用嵌入缓存减少重复计算
- 定期清理缓存避免内存溢出
- 调整缓存TTL和大小限制

### 搜索优化
- 调整相似度阈值平衡准确性和召回率
- 使用批处理提高吞吐量
- 优化查询策略权重

### 硬件优化
- 使用GPU加速嵌入计算
- 增加内存提高缓存命中率
- 使用SSD提高I/O性能

## 📝 日志和监控

### 日志级别
- `DEBUG`: 详细的调试信息
- `INFO`: 一般操作信息
- `WARNING`: 警告信息
- `ERROR`: 错误信息

### 性能监控

```python
from src import performance_monitor

# 查看性能统计
metrics = performance_monitor.get_metrics()
print(metrics)
```

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证。

## 🆘 支持

如有问题或建议，请：
1. 查看故障排除部分
2. 检查现有Issue
3. 创建新Issue描述问题
4. 联系开发团队