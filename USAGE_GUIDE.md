# Milvus Collection 创建工具使用指南

## 功能概述

`create_milvus_collection.py` 现在支持创建两种类型的 Collection：
1. **Metadata Collection** (`Pampers_metadata`) - 用于存储档案和事件属性的元数据
2. **Event Collection** (`Event_metadata`) - 专门用于存储事件属性

## 使用方法

### 1. 交互式菜单（推荐）
```bash
uv run python create_milvus_collection.py
```
运行后会显示菜单，让您选择要创建的 Collection。

### 2. 命令行参数
```bash
# 只创建 Metadata Collection
uv run python create_milvus_collection.py metadata

# 只创建 Event Collection
uv run python create_milvus_collection.py event

# 创建两个 Collection
uv run python create_milvus_collection.py both
```

## Collection Schema 对比

### Metadata Collection (Pampers_metadata)
| 字段名 | 数据类型 | 主键 | 描述 |
|--------|----------|------|------|
| concept_id | VARCHAR(1024) | ✓ | 概念属性的全局唯一ID (如: PROFILE_UserProfile_country) |
| source_type | VARCHAR(64) | × | 来源类型: 'PROFILE' 或 'EVENT' |
| source_name | VARCHAR(256) | × | 来源实体名称 |
| field_name | VARCHAR(256) | × | 属性字段本身的名称 |
| raw_metadata | JSON | × | 存储属性的原始元数据 |
| concept_embedding | FLOAT_VECTOR(768) | × | 由属性描述文本生成的语义向量 |

### Event Collection (Event_metadata)
| 字段名 | 数据类型 | 主键 | 描述 |
|--------|----------|------|------|
| concept_id | VARCHAR(1024) | ✓ | 属性的唯一ID (如: EVENT_UserLoginEvent_device_type) |
| source_type | VARCHAR(64) | × | 固定为 'EVENT' |
| source_name | VARCHAR(256) | × | 事件名 (如: UserLoginEvent) |
| field_name | VARCHAR(256) | × | 属性名 (如: device_type) |
| raw_metadata | JSON | × | 存储该属性的原始元数据 |
| concept_embedding | FLOAT_VECTOR(768) | × | 核心向量，由详细描述生成 |

## 配置参数

在 `create_milvus_collection.py` 文件顶部可以修改以下配置：

```python
MILVUS_HOST = "172.28.9.45"        # Milvus 服务器地址
MILVUS_PORT = "19530"               # Milvus 端口
MILVUS_DATABASE = "default"         # 数据库名称
COLLECTION_NAME = "Pampers_metadata"     # Metadata Collection 名称
EVENT_COLLECTION_NAME = "Event_metadata" # Event Collection 名称
VECTOR_DIMENSION = 768              # 向量维度
```

## 索引配置

两个 Collection 都使用相同的索引配置：
- **索引类型**: HNSW (高性能图索引)
- **度量类型**: IP (内积，适用于归一化的向量)
- **参数**: M=16, efConstruction=256

## 功能特点

1. **自动重建**: 如果 Collection 已存在，会先删除再重建
2. **连接管理**: 使用辅助函数统一管理连接和断开
3. **错误处理**: 完善的异常处理和日志输出
4. **灵活选择**: 可以单独创建任一 Collection 或同时创建两个

## 注意事项

1. 确保 Milvus 服务器在指定地址运行
2. 向量维度必须与实际使用的 Embedding 模型匹配
3. Collection 创建后，索引会在数据插入时自动构建
4. 两个 Collection 的 schema 相似但用途不同：
   - Metadata Collection: 通用元数据存储
   - Event Collection: 专门用于事件相关数据

## 测试脚本

运行测试脚本验证配置：
```bash
uv run python test_collections.py
```

## 数据库管理

查看和管理数据库：
```bash
uv run python milvus_database_utils.py
```