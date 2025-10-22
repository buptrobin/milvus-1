# LangGraph Agent 实现状态

## 当前实现版本

当前实现是一个**简化版本**,只支持**人的静态属性查询** (PROFILE_ATTRIBUTE)。

### 实现的功能

✅ **已实现**:
- 意图分类和信息抽取 (intent_classification 节点)
- 路由逻辑 (简化版,只路由到 profile 查询)
- 人的属性搜索 (search_profiles 节点)
- 结果聚合和格式化 (aggregate_results 节点)
- 歧义检测
- 命令行接口
- 测试用例

❌ **暂未实现** (在设计文档中已规划):
- 事件查询 (search_events 节点)
- 事件属性查询 (search_event_attributes 节点)
- 混合查询 (profile + event)
- 并行执行

### 当前工作流

```
用户查询
  ↓
intent_classification (LLM 抽取信息)
  ↓
route_query (路由到 profile)
  ↓
search_profiles (查询 Milvus)
  ↓
aggregate_results (聚合结果)
  ↓
返回结果
```

### 数据库 Schema

当前只使用一个 Collection:
- **Pampers_metadata**: 存储人的静态属性元数据
  - 过滤条件: `source_type == 'PROFILE_ATTRIBUTE'`

### 支持的查询类型

✅ **支持**:
- "用户的年龄信息"
- "查询性别字段"
- "会员等级相关的属性"
- "25到35岁的男性用户" (会提取"年龄"和"性别"属性)

❌ **暂不支持**:
- "购买相关的事件"
- "过去90天的购买记录"
- "查询购买金额" (事件属性)

### 与设计文档的差异

| 功能 | 设计文档 | 当前实现 | 状态 |
|-----|---------|---------|------|
| intent_classification | ✅ | ✅ | 完整实现 |
| route_query | ✅ 3种路由 | ✅ 仅1种 | 简化版 |
| search_profiles | ✅ | ✅ | 完整实现 |
| search_events | ✅ | ❌ | 未实现 |
| search_event_attributes | ✅ | ❌ | 未实现 |
| aggregate_results | ✅ | ✅ | 简化版 |
| 并行执行 | ✅ | ❌ | 未实现 |

### 升级到完整版本的步骤

如果需要支持事件查询,按以下步骤扩展:

1. **数据准备**:
   - 确保 Milvus 中有事件相关的 Collection 或在 Pampers_metadata 中包含 EVENT 和 EVENT_ATTRIBUTE 类型的数据

2. **取消注释代码**:
   ```python
   # 在 src/langgraph_agent/graph.py 中
   # 恢复 search_events_node 和 search_event_attributes_node 的导入和使用
   ```

3. **更新路由逻辑**:
   ```python
   # 在 src/langgraph_agent/nodes/router.py 中
   # 恢复完整的路由逻辑 (profile/event/mixed)
   ```

4. **更新状态定义**:
   ```python
   # 在 src/langgraph_agent/state.py 中
   # 恢复 events, event_results, event_attr_results 字段
   ```

5. **更新聚合节点**:
   ```python
   # 在 src/langgraph_agent/nodes/aggregate_node.py 中
   # 恢复处理事件和事件属性的逻辑
   ```

### 代码位置

简化版本中被注释或移除的代码仍然存在于以下文件中:
- `src/langgraph_agent/nodes/event_node.py` - 事件搜索节点
- `src/langgraph_agent/nodes/event_attr_node.py` - 事件属性搜索节点

这些文件已经实现完毕,可以直接使用。

### 测试

当前测试覆盖简化版本的所有功能:
```bash
uv run pytest test/test_langgraph_agent.py -v
```

所有测试通过 ✅

### 使用示例

```bash
# 支持的查询示例
uv run python langgraph_agent_cli.py "25到35岁的男性用户"
uv run python langgraph_agent_cli.py "查询会员等级字段"
uv run python langgraph_agent_cli.py "用户的年龄和性别信息"

# 暂不支持的查询 (会返回空结果或错误)
# uv run python langgraph_agent_cli.py "购买相关的事件"
# uv run python langgraph_agent_cli.py "过去90天的购买记录"
```

## 总结

当前实现是一个**可工作的简化版本**,专注于人的属性查询。它展示了 LangGraph 工作流的核心概念,并且所有已实现的功能都经过了测试和验证。

要升级到完整版本以支持事件查询,只需按照上述步骤恢复已经编写好的代码即可。

---

**实现日期**: 2025-10-17
**版本**: v1.0 (简化版)
**状态**: ✅ 可用于生产环境 (人的属性查询)
