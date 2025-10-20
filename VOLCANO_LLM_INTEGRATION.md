# 火山引擎大模型集成说明

## 概述

本项目已集成火山引擎（Volcano Engine）的大语言模型API，支持使用豆包（Doubao）系列公共模型，用于增强自然语言查询的信息抽取能力。通过调用火山引擎的大模型，系统可以更准确地理解用户意图、识别实体、提取关键信息。

## 更新说明 (2024)

- **新增公共模型支持**：可直接使用豆包系列公共模型，无需创建专属端点
- **简化配置流程**：只需API Key即可开始使用
- **多模型选择**：支持轻量版、专业版等多种规格模型
- **向后兼容**：保留原有自定义端点支持

## 主要功能

### 1. 智能信息抽取
- **实体识别**：自动识别查询中的人物、地点、时间、数值等实体
- **意图分类**：判断用户查询意图（搜索、筛选、聚合、比较等）
- **关键词提取**：从自然语言中提取关键词和短语
- **筛选条件解析**：识别并结构化查询中的筛选条件
- **时间信息提取**：解析时间范围和日期信息
- **数值信息提取**：识别数值、单位及其上下文

### 2. 灵活配置
- 支持自定义系统提示词
- 可配置的信息抽取模板
- 可调整的模型参数（温度、最大令牌数等）
- 可选的启用/禁用开关

## 配置步骤

### 1. 安装依赖

项目已在 `pyproject.toml` 中添加了必要的依赖：
```bash
uv sync
```

### 2. 配置环境变量

复制 `.env.example` 文件并创建 `.env`：
```bash
cp .env.example .env
```

编辑 `.env` 文件，选择以下配置方式之一：

#### 方式1：使用公共模型（推荐）

```env
# 启用火山引擎LLM
VOLCANO_ENABLED=true

# API密钥（必填）
VOLCANO_API_KEY=your_api_key_here

# 使用公共模型
VOLCANO_USE_PUBLIC_MODEL=true

# 选择模型（以下选其一）
VOLCANO_MODEL=doubao-lite-32k      # 轻量版，成本低，适合开发测试
# VOLCANO_MODEL=doubao-pro-32k     # 专业版，性能均衡
# VOLCANO_MODEL=doubao-pro-128k    # 专业版，长文本
# VOLCANO_MODEL=doubao-1.5-pro-32k # 最新1.5版，推荐生产使用

# 可选配置
VOLCANO_SYSTEM_PROMPT=              # 系统提示词
VOLCANO_EXTRACTION_PROMPT=          # 抽取提示词模板
VOLCANO_MAX_TOKENS=1024            # 最大令牌数
VOLCANO_TEMPERATURE=0.1            # 生成温度
```

#### 方式2：使用自定义端点（高级）

```env
# 启用火山引擎LLM
VOLCANO_ENABLED=true

# API密钥（必填）
VOLCANO_API_KEY=your_api_key_here

# 使用自定义端点
VOLCANO_USE_PUBLIC_MODEL=false
VOLCANO_ENDPOINT_ID=ep-xxxxx-xxxxx  # 您的端点ID
```

### 3. 获取API凭证

1. 访问[火山引擎控制台](https://console.volcengine.com/ark)
2. 创建或选择一个项目
3. 获取API Key
4. （可选）如需使用自定义模型，创建模型端点（Endpoint），获取端点ID

## 可用的公共模型

### 轻量版模型（成本低，速度快）
- `doubao-lite-32k`：32K上下文窗口
- `doubao-lite-128k`：128K上下文窗口
- `doubao-1.5-lite-32k`：1.5版轻量模型

### 专业版模型（性能好，准确度高）
- `doubao-pro-32k`：32K上下文窗口
- `doubao-pro-128k`：128K上下文窗口
- `doubao-pro-256k`：256K上下文窗口
- `doubao-1.5-pro-32k`：1.5版专业模型，推荐

### 选择建议
- **开发测试**：`doubao-lite-32k`
- **生产环境**：`doubao-pro-32k` 或 `doubao-1.5-pro-32k`
- **长文本处理**：`doubao-pro-128k` 或 `doubao-pro-256k`

## 使用方法

### 1. 在代码中直接使用LLM提取器

#### 使用公共模型

```python
from src import VolcanoLLMExtractor

# 使用公共模型（推荐）
extractor = VolcanoLLMExtractor(
    api_key="your_api_key",
    model="doubao-pro-32k"  # 直接使用模型名称
)

# 抽取信息
query = "查找年龄大于30岁的会员信息"
result = extractor.extract(query)

# 使用抽取结果
print(f"意图: {result.intent_type}")
print(f"实体: {result.entities}")
print(f"关键词: {result.key_terms}")
```

#### 使用自定义端点

```python
# 使用自定义端点（如果您创建了专属模型）
extractor = VolcanoLLMExtractor(
    api_key="your_api_key",
    model="ep-xxxxx-xxxxx"  # 端点ID
)
```

### 2. 通过Agent使用（自动集成）

当配置正确且启用后，`NaturalLanguageQueryAgent` 会自动使用LLM进行信息抽取：

```python
from src import NaturalLanguageQueryAgent

# Agent会自动加载配置并初始化LLM提取器
agent = NaturalLanguageQueryAgent()

# 处理查询时会自动使用LLM增强
result = agent.process_query("用户的年龄信息")
```

### 3. 选择合适的模型

```python
from src.volcano_models import recommend_model, get_model_info

# 自动推荐模型
model = recommend_model(
    task_type="extraction",
    context_length=1000,
    cost_sensitive=True
)
print(f"推荐模型: {model}")

# 获取模型信息
info = get_model_info("doubao-pro-32k")
print(f"模型: {info.display_name}")
print(f"上下文: {info.context_window} tokens")
print(f"成本级别: {info.cost_level}")
```

### 4. 自定义提示词

您可以通过环境变量或代码设置自定义提示词：

```python
# 方法1: 通过环境变量设置
# 在.env文件中设置 VOLCANO_EXTRACTION_PROMPT

# 方法2: 在代码中设置
extractor = VolcanoLLMExtractor(
    api_key="your_api_key",
    model="doubao-pro-32k",
    extraction_prompt_template="""
    请从查询中提取以下信息：
    查询: {query}

    返回JSON格式，包含：
    - entities: 实体列表
    - intent_type: 查询意图
    - key_terms: 关键词
    """
)
```

## 测试

运行测试脚本验证集成：

```bash
# 测试公共模型
uv run python test_public_models.py

# 测试LLM提取器
uv run python test_llm_extractor.py

# 测试完整的Agent（包含LLM）
uv run python natural_language_agent.py -q "查找VIP会员的消费记录"
```

## 架构说明

### 新增模块

1. **src/llm_extractor.py**
   - `VolcanoLLMExtractor`: 火山引擎LLM提取器类
   - `ExtractedInfo`: 提取信息数据类
   - 支持公共模型和自定义端点

2. **src/volcano_models.py**
   - 公共模型常量定义
   - 模型信息和推荐功能
   - 模型验证工具

3. **src/config.py**
   - `VolcanoConfig`: 火山引擎配置类
   - 支持公共模型配置

### 修改的模块

1. **src/query_processor.py**
   - 集成LLM提取器
   - 增强查询意图分类
   - 合并LLM提取的实体和关键词

2. **src/nl_query_agent.py**
   - 初始化LLM提取器
   - 在查询处理流程中使用LLM

## 降级处理

系统实现了优雅的降级机制：

1. **LLM不可用时**：自动回退到基于规则的提取
2. **API调用失败时**：使用缓存或基础提取方法
3. **未配置时**：系统正常运行，仅不使用LLM功能

## 性能和成本考虑

### 性能
- LLM调用会增加延迟：
  - 轻量版模型：0.5-2秒
  - 专业版模型：1-3秒
- 建议为频繁查询启用缓存
- 可通过 `VOLCANO_ENABLED=false` 快速禁用LLM功能

### 成本
- 免费额度：50万 tokens
- 计费示例（输入价格/千tokens）：
  - `doubao-lite-32k`：¥0.0003
  - `doubao-pro-32k`：¥0.0008
  - `doubao-1.5-pro-32k`：¥0.0008

### 优化建议
- 开发阶段使用 `doubao-lite-32k`
- 生产环境根据准确度要求选择合适模型
- 温度设置为0.1以获得更一致的结果

## 常见问题

### Q: 如何判断LLM是否正在工作？
A: 查看日志，成功时会显示 "LLM extractor initialized successfully"

### Q: 可以使用哪些模型？
A: 支持所有豆包公共模型和自定义端点：
- 公共模型：doubao-lite/pro系列，无需创建端点
- 自定义模型：通过端点ID（ep-xxxxx）访问

### Q: 如何优化抽取效果？
A:
1. 选择合适的模型（任务复杂度vs成本）
2. 调整系统提示词和抽取模板
3. 调整温度参数（更低的温度产生更确定的结果）
4. 对于简单任务使用轻量模型，复杂任务使用专业模型

### Q: API调用失败怎么办？
A: 系统会自动降级到规则基础提取，不影响基本功能

## 支持

如有问题，请检查：
1. API Key 是否正确
2. 模型名称或Endpoint ID 是否有效
3. 网络连接是否正常
4. 查看日志中的错误信息
5. 运行 `test_public_models.py` 进行诊断

## 参考链接

- [火山引擎控制台](https://console.volcengine.com/ark)
- [火山方舟文档中心](https://www.volcengine.com/docs/82379)
- [豆包大模型介绍](https://www.volcengine.com/product/doubao)