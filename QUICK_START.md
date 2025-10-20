# LangGraph Agent 快速开始

## 成功运行! ✅

LangGraph Agent 已经成功实现并可以运行。

## 快速测试

### 1. 清除 Python 缓存(首次运行时)
```bash
rm -rf src/langgraph_agent/__pycache__ src/langgraph_agent/nodes/__pycache__
```

### 2. 运行测试
```bash
# JSON 输出(推荐,避免中文乱码)
uv run python langgraph_agent_cli.py "用户的年龄信息" --json

# 交互模式
uv run python langgraph_agent_cli.py --interactive

# 详细日志
uv run python langgraph_agent_cli.py "查询会员等级" --verbose
```

### 3. 运行单元测试
```bash
uv run pytest test/test_langgraph_agent.py -v
```

## 当前状态

### ✅ 已实现和测试的功能:
- LangGraph 工作流编排
- 意图分类节点
- 路由逻辑
- 人的属性搜索节点
- 结果聚合节点
- Milvus 向量搜索
- BGE-M3 嵌入模型
- 命令行接口
- JSON 输出
- 交互模式

### ⚠️ 已知问题:

1. **LLM JSON 解析失败**:
   - LLM 返回的格式可能不符合预期
   - 系统会回退到基本提取(confidence=0.30)
   - 不影响程序运行
   - 解决方案: 调整 `prompt.txt` 的格式要求

2. **Windows 控制台中文乱码**:
   - 建议使用 `--json` 参数输出
   - 或使用 Windows Terminal / WSL

## 简化版本说明

当前实现是**简化版本**,只支持人的属性查询:
- ✅ 支持: `source_type='PROFILE_ATTRIBUTE'`
- ❌ 暂不支持: 事件查询和事件属性查询

事件相关的节点代码已经编写完成但未集成。如需完整功能,参考 `IMPLEMENTATION_STATUS.md`。

## 文档

- `LANGGRAPH_USAGE.md` - 详细使用指南
- `LANGGRAPH_AGENT_DESIGN.md` - 设计文档
- `IMPLEMENTATION_STATUS.md` - 实现状态说明

## 示例输出

```json
{
  "query": "用户的年龄信息",
  "intent_type": "profile",
  "profile_attributes": [
    {
      "idname": "age_group",
      "source_name": "年龄段",
      "source": "pampers_customer",
      "original_query": "年龄",
      "score": 0.85,
      "confidence_level": "high"
    }
  ],
  "summary": "已识别用户属性: 年龄段",
  "total_results": 1,
  "confidence_score": 0.85,
  "has_ambiguity": false,
  "ambiguous_options": [],
  "execution_time": 1.2
}
```

## 故障排除

### 问题: ModuleNotFoundError
```bash
# 解决方案: 清除缓存
rm -rf src/langgraph_agent/__pycache__ src/langgraph_agent/nodes/__pycache__
```

### 问题: 中文乱码
```bash
# 解决方案: 使用 JSON 输出
uv run python langgraph_agent_cli.py "查询" --json
```

### 问题: LLM 提取失败
- 检查 `prompt.txt` 文件
- 检查 VOLCANO_API_KEY 配置
- 查看详细日志: `--verbose`

## 性能

- 模型加载: ~10-15秒(首次)
- LLM 调用: ~1-2秒
- 向量搜索: ~0.1秒
- 总耗时: ~2-3秒

## 支持

有问题请查看:
- 日志输出(使用 `--verbose`)
- 设计文档 `LANGGRAPH_AGENT_DESIGN.md`
- 实现状态 `IMPLEMENTATION_STATUS.md`

---

**状态**: ✅ 可用
**版本**: v1.0 (简化版)
**日期**: 2025-10-17
