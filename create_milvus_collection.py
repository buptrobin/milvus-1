from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection

# --- 1. 配置信息 (Configuration) ---
# 请在此处配置您的 Milvus 服务信息和 Collection 定义
MILVUS_ALIAS = "ipinyou_milvus"  # 为连接起一个别名，方便管理
MILVUS_HOST = "172.28.9.45"
MILVUS_PORT = "19530"
MILVUS_DATABASE = "default"  # 指定数据库名称，默认为 "default"

COLLECTION_NAME = "Pampers_metadata"
EVENT_COLLECTION_NAME = "Pampers_Event_metadata"  # 事件专用的 Collection
VECTOR_DIMENSION = 1024  # BGE-M3 模型的输出维度

# --- 2. 辅助函数 (Helper Functions) ---
def connect_to_milvus():
    """连接到 Milvus 服务器"""
    print(f"[*] 正在尝试连接到 Milvus 服务器: {MILVUS_HOST}:{MILVUS_PORT}...")
    print(f"    数据库: {MILVUS_DATABASE}")
    connections.connect(
        alias=MILVUS_ALIAS,
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        db_name=MILVUS_DATABASE
    )
    print(f"[+] 成功连接到 Milvus 数据库 '{MILVUS_DATABASE}'！")

def disconnect_from_milvus():
    """断开与 Milvus 的连接"""
    if MILVUS_ALIAS in connections.list_connections():
        connections.disconnect(MILVUS_ALIAS)
        print("\n[*] 与 Milvus 的连接已断开。")

# --- 3. Collection 创建函数 (Collection Creation Functions) ---
def create_metadata_collection():
    """
    连接到指定的 Milvus 服务器，并创建用于概念对齐的 Collection。
    如果 Collection 已存在，则会先删除再重建。
    """
    print("--- 创建 Metadata Collection ---")

    try:
        # 步骤 1: 连接到 Milvus 服务器
        connect_to_milvus()

        # 步骤 2: 检查 Collection 是否已存在，如果存在则删除
        if utility.has_collection(COLLECTION_NAME, using=MILVUS_ALIAS):
            print(f"[!] Collection '{COLLECTION_NAME}' 已存在。为了确保环境干净，将执行删除操作。")
            utility.drop_collection(COLLECTION_NAME, using=MILVUS_ALIAS)
            print(f"[+] 旧的 Collection '{COLLECTION_NAME}' 已成功删除。")

        # 步骤 3: 定义 Collection 的 Schema
        print("[*] 正在定义 Collection Schema...")
        fields = [
            # 主键字段
            FieldSchema(name="concept_id", dtype=DataType.VARCHAR, is_primary=True, max_length=1024,
                        description="概念属性的全局唯一ID, 例如: PROFILE_UserProfile_country"),

            # 标量元数据字段
            FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=64,
                        description="来源类型, 'PROFILE' 或 'EVENT'"),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256,
                        description="来源信息，指明是哪个档案或者哪个事件"),
            FieldSchema(name="source_name", dtype=DataType.VARCHAR, max_length=256,
                        description="来源实体名称, 例如档案名或事件名"),
            FieldSchema(name="idname", dtype=DataType.VARCHAR, max_length=256,
                        description="属性字段本身的名称"),
            FieldSchema(name="raw_metadata", dtype=DataType.JSON,
                        description="存储属性的原始元数据 (描述, 类型, 枚举等)"),

            # 核心向量字段
            FieldSchema(name="concept_embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION,
                        description="由属性描述文本生成的语义向量")
        ]

        schema = CollectionSchema(
            fields=fields,
            description="用于存储档案和事件属性的元数据",
            enable_dynamic_field=False # 建议关闭动态字段以保持 Schema 严格
        )
        print("[+] Schema 定义完成。")

        # 步骤 4: 创建 Collection
        print(f"[*] 正在创建 Collection '{COLLECTION_NAME}'...")
        collection = Collection(
            name=COLLECTION_NAME,
            schema=schema,
            using=MILVUS_ALIAS
        )
        print(f"[+] Collection '{COLLECTION_NAME}' 创建成功！")
        print(f"    - 主键字段: {collection.primary_field.name}")
        print(f"    - 字段总数: {len(collection.schema.fields)}")

        # 步骤 5: 为向量字段创建索引
        print(f"[*] 正在为向量字段 'concept_embedding' 创建索引...")
        index_params = {
            "metric_type": "COSINE",      # Inner Product (内积), 适用于归一化的 Embedding 向量
            "index_type": "HNSW",     # 高性能的图索引
            "params": {"M": 16, "efConstruction": 256}  # HNSW 索引的构建参数
        }

        collection.create_index(
            field_name="concept_embedding",
            index_params=index_params
        )
        print("[+] 索引创建任务已提交。数据插入后，索引将会在后台自动构建。")

        print("\n--- 所有操作已成功完成！ ---")

    except Exception as e:
        print(f"\n[X] 操作失败，发生错误: {e}")

    finally:
        # 步骤 6: 断开连接
        disconnect_from_milvus()

def create_event_collection():
    """
    创建用于存储事件属性的 Collection。
    如果 Collection 已存在，则会先删除再重建。
    """
    print("--- 创建 Event Collection ---")

    try:
        # 步骤 1: 连接到 Milvus 服务器
        connect_to_milvus()

        # 步骤 2: 检查 Collection 是否已存在，如果存在则删除
        if utility.has_collection(EVENT_COLLECTION_NAME, using=MILVUS_ALIAS):
            print(f"[!] Collection '{EVENT_COLLECTION_NAME}' 已存在。为了确保环境干净，将执行删除操作。")
            utility.drop_collection(EVENT_COLLECTION_NAME, using=MILVUS_ALIAS)
            print(f"[+] 旧的 Collection '{EVENT_COLLECTION_NAME}' 已成功删除。")

        # 步骤 3: 定义 Collection 的 Schema
        print("[*] 正在定义 Event Collection Schema...")
        fields = [
            # 主键字段
            FieldSchema(name="concept_id", dtype=DataType.VARCHAR, is_primary=True, max_length=1024,
                        description="属性的唯一ID, 例如: EVENT_UserLoginEvent_device_type"),

            # 标量元数据字段
            FieldSchema(name="profile_type", dtype=DataType.VARCHAR, max_length=64,
                        description="固定为 'EVENT'"),
            FieldSchema(name="event_name", dtype=DataType.VARCHAR, max_length=256,
                        description="链接字段。存储该属性所属的事件名，如 UserLoginEvent"),
            FieldSchema(name="event_idname", dtype=DataType.VARCHAR, max_length=256,
                        description="属性名，如 device_type"),
            FieldSchema(name="raw_metadata", dtype=DataType.JSON,
                        description="存储该属性的原始元数据"),

            # 核心向量字段
            FieldSchema(name="concept_embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION,
                        description="核心向量。由该属性的详细描述（包括枚举值）生成")
        ]

        schema = CollectionSchema(
            fields=fields,
            description="用于存储事件属性的元数据",
            enable_dynamic_field=False  # 建议关闭动态字段以保持 Schema 严格
        )
        print("[+] Event Schema 定义完成。")

        # 步骤 4: 创建 Collection
        print(f"[*] 正在创建 Collection '{EVENT_COLLECTION_NAME}'...")
        collection = Collection(
            name=EVENT_COLLECTION_NAME,
            schema=schema,
            using=MILVUS_ALIAS
        )
        print(f"[+] Collection '{EVENT_COLLECTION_NAME}' 创建成功！")
        print(f"    - 主键字段: {collection.primary_field.name}")
        print(f"    - 字段总数: {len(collection.schema.fields)}")

        # 步骤 5: 为向量字段创建索引
        print(f"[*] 正在为向量字段 'concept_embedding' 创建索引...")
        index_params = {
            "metric_type": "COSINE",      # Inner Product (内积), 适用于归一化的 Embedding 向量
            "index_type": "HNSW",     # 高性能的图索引
            "params": {"M": 16, "efConstruction": 256}  # HNSW 索引的构建参数
        }

        collection.create_index(
            field_name="concept_embedding",
            index_params=index_params
        )
        print("[+] 索引创建任务已提交。数据插入后，索引将会在后台自动构建。")

        print("\n--- Event Collection 创建成功！ ---")

    except Exception as e:
        print(f"\n[X] 操作失败，发生错误: {e}")

    finally:
        # 步骤 6: 断开连接
        disconnect_from_milvus()

# --- 4. 主函数 (Main Function) ---
def main():
    """主函数，提供菜单选择"""
    print("=" * 60)
    print("Milvus Collection 创建工具")
    print("=" * 60)
    print(f"服务器: {MILVUS_HOST}:{MILVUS_PORT}")
    print(f"数据库: {MILVUS_DATABASE}")
    print("=" * 60)
    print("\n请选择要创建的 Collection:")
    print("  1. 创建 Metadata Collection (Pampers_metadata)")
    print("  2. 创建 Event Collection (Event_metadata)")
    print("  3. 创建两个 Collection")
    print("  0. 退出")

    choice = input("\n请输入选项 (0-3): ").strip()

    if choice == "1":
        print("\n")
        create_metadata_collection()
    elif choice == "2":
        print("\n")
        create_event_collection()
    elif choice == "3":
        print("\n[*] 将创建两个 Collection...\n")
        create_metadata_collection()
        print("\n" + "=" * 60 + "\n")
        create_event_collection()
    elif choice == "0":
        print("\n[*] 退出程序")
    else:
        print("\n[!] 无效选项，请重新运行程序")

# --- 脚本执行入口 ---
if __name__ == "__main__":
    import sys

    # 如果有命令行参数，直接执行对应的功能
    if len(sys.argv) > 1:
        if sys.argv[1] == "metadata":
            create_metadata_collection()
        elif sys.argv[1] == "event":
            create_event_collection()
        elif sys.argv[1] == "both":
            create_metadata_collection()
            print("\n" + "=" * 60 + "\n")
            create_event_collection()
        else:
            print("用法:")
            print("  python create_milvus_collection.py          # 交互式菜单")
            print("  python create_milvus_collection.py metadata # 创建 metadata collection")
            print("  python create_milvus_collection.py event    # 创建 event collection")
            print("  python create_milvus_collection.py both     # 创建两个 collection")
    else:
        # 交互式菜单
        main()