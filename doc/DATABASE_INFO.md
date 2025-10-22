# Milvus 数据库说明

## 当前配置

根据 `create_milvus_collection.py` 的配置：

- **服务器**: 172.28.9.45:19530
- **数据库**: `default` (默认数据库)
- **Collection**: `Pampers_metadata`

## Milvus 数据库结构

```
Milvus Server
├── Database: default (默认)
│   ├── Collection: Pampers_metadata
│   └── Collection: ...
├── Database: custom_db1
│   └── Collection: ...
└── Database: custom_db2
    └── Collection: ...
```

## 如何指定数据库

### 1. 在连接时指定（推荐）
```python
connections.connect(
    alias="default",
    host="172.28.9.45",
    port="19530",
    db_name="your_database"  # 指定数据库名
)
```

### 2. 连接后切换数据库
```python
from pymilvus import db

# 先连接
connections.connect(host="172.28.9.45", port="19530")

# 切换数据库
db.using_database("your_database")
```

## 数据库管理命令

### 使用 milvus_database_utils.py

```bash
# 交互式模式
uv run python milvus_database_utils.py

# 命令行模式
uv run python milvus_database_utils.py list              # 列出所有数据库
uv run python milvus_database_utils.py create mydb       # 创建数据库
uv run python milvus_database_utils.py use mydb          # 切换到数据库
uv run python milvus_database_utils.py drop mydb         # 删除数据库
```

### 使用 Python 代码

```python
from pymilvus import db

# 列出所有数据库
databases = db.list_database()

# 创建数据库
db.create_database("my_database")

# 使用数据库
db.using_database("my_database")

# 删除数据库
db.drop_database("my_database")
```

## 修改 create_milvus_collection.py 使用不同数据库

如果您想在不同的数据库中创建 Collection，只需修改配置：

```python
# 修改第 8 行
MILVUS_DATABASE = "your_database_name"  # 改为您想要的数据库名
```

## 注意事项

1. **默认数据库**: 如果不指定，Collection 会创建在 `default` 数据库中
2. **数据库隔离**: 不同数据库中的 Collection 是完全隔离的
3. **同名 Collection**: 不同数据库可以有同名的 Collection
4. **版本要求**: Milvus 2.3.0+ 才支持多数据库功能
5. **default 数据库**: 不能删除，始终存在

## 查看 Collection 所在数据库

连接到特定数据库后，使用：

```python
from pymilvus import utility

# 列出当前数据库的所有 Collection
collections = utility.list_collections()
print(f"当前数据库的 Collections: {collections}")
```

## 最佳实践

1. **开发环境**: 使用 `dev` 或 `development` 数据库
2. **测试环境**: 使用 `test` 或 `staging` 数据库  
3. **生产环境**: 使用 `prod` 或 `production` 数据库
4. **项目隔离**: 为不同项目创建不同的数据库

例如：
- `pampers_dev` - Pampers 项目开发库
- `pampers_prod` - Pampers 项目生产库
- `experiment` - 实验和测试用库