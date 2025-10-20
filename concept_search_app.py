import json
import random
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
from FlagEmbedding import BGEM3FlagModel

# --- 1. 全局配置 ---
COLLECTION_NAME = "Pampers_metadata"
EMBEDDING_MODEL = 'BAAI/bge-m3'
DIMENSION = 1024  # BGE-M3模型的维度


MILVUS_ALIAS = "ipinyou_milvus"  # 为连接起一个别名，方便管理
MILVUS_HOST = "172.28.9.45"
MILVUS_PORT = "19530"
MILVUS_DATABASE = "default"  # 指定数据库名称，默认为 "default"

# 定义歧义判断的阈值
SIMILARITY_THRESHOLD = 0.65  # 认为是强相关的最低分数
SCORE_GAP_THRESHOLD = 0.08   # 最高分和第二高分之间的差距如果小于此值，则认为模糊

# --- 2. 模拟元数据 ---
# 包含一个“档案”和一个“事件”，它们都有一个'country'字段，用于演示歧义
MOCK_METADATA = {
    "archives": [
        {
            "name": "UserProfile",
            "desc": "用户个人档案，记录用户的基本信息。",
            "fields": {
                "user_id": {"desc": "用户唯一ID", "type": "string"},
                "age": {"desc": "用户年龄", "type": "int"},
                "country": {"desc": "用户注册时所在的国家", "type": "enum", "enums": ["US", "CN", "JP", "DE", "GB"]},
            }
        }
    ],
    "events": [
        {
            "name": "ShipmentTracking",
            "desc": "包裹物流追踪事件。",
            "fields": {
                "tracking_id": {"desc": "物流追踪号", "type": "string"},
                "status": {"desc": "物流状态", "type": "enum", "enums": ["in_transit", "delivered", "exception"]},
                "destination_country": {"desc": "包裹的目的地国家", "type": "enum", "enums": ["US", "CN", "JP", "DE", "GB"]},
                "origin_country": {"desc": "包裹的始发地国家", "type": "enum", "enums": ["US", "CN", "JP", "DE", "GB"]}
            }
        }
    ]
}

# --- 3. 核心功能函数 ---

def setup_milvus():
    """连接Milvus并创建Collection，如果不存在的话。"""
    print("正在连接 Milvus...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    if utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' 已存在，将删除重建。")
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="concept_id", dtype=DataType.VARCHAR, is_primary=True, max_length=1024),
        FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="field_name", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="concept_embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="raw_metadata", dtype=DataType.JSON)
    ]
    schema = CollectionSchema(fields, description="概念对齐引擎")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    print("正在创建索引...")
    index_params = {
        "metric_type": "COSINE", # 内积，等价于余弦相似度
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 200}
    }
    collection.create_index("concept_embedding", index_params)
    return collection

def get_embedding_model():
    """加载BGE-M3文本嵌入模型"""
    print("正在加载 BGE-M3 Embedding 模型...")
    return BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=True)

def prepare_and_insert_data(collection, model, metadata):
    """处理元数据并插入Milvus"""
    print("正在准备和插入数据...")
    data_to_insert = []

    # 处理档案
    for archive in metadata['archives']:
        for field_name, details in archive['fields'].items():
            data_to_insert.append(
                process_single_field("ARCHIVE", archive['name'], field_name, details)
            )
    # 处理事件
    for event in metadata['events']:
        for field_name, details in event['fields'].items():
            data_to_insert.append(
                process_single_field("EVENT", event['name'], field_name, details)
            )

    # 生成向量 - 使用BGE-M3的dense embeddings
    texts = [item['text_for_embedding'] for item in data_to_insert]
    embeddings = model.encode(texts)['dense_vecs']

    # 准备插入Milvus的数据
    entities = [
        [item['concept_id'] for item in data_to_insert],
        [item['source_type'] for item in data_to_insert],
        [item['source_name'] for item in data_to_insert],
        [item['field_name'] for item in data_to_insert],
        list(embeddings),
        [item['raw_metadata'] for item in data_to_insert]
    ]

    collection.insert(entities)
    collection.flush()
    print(f"成功插入 {len(data_to_insert)} 条概念属性数据。")

def process_single_field(source_type, source_name, field_name, details):
    """处理单个字段，生成用于向量化和存储的数据"""
    concept_id = f"{source_type}_{source_name}_{field_name}"
    source_type_cn = "档案" if source_type == "ARCHIVE" else "事件"

    text = (f"这是一个来源为'{source_type}'，名为'{source_name}'的{source_type_cn}下的属性字段，"
            f"名为'{field_name}'。它的含义是'{details['desc']}'。")

    if 'enums' in details:
        enums_text = ", ".join(details['enums'])
        text += f" 它是一个枚举类型，可能的取值包括：{enums_text}。"

    return {
        "concept_id": concept_id,
        "source_type": source_type,
        "source_name": source_name,
        "field_name": field_name,
        "text_for_embedding": text,
        "raw_metadata": details
    }

def search_and_analyze(collection, model, query):
    """执行搜索并分析结果"""
    query_vector = model.encode([query])['dense_vecs'][0].tolist()

    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": 128},
    }

    results = collection.search(
        data=[query_vector],
        anns_field="concept_embedding",
        param=search_params,
        limit=5,
        output_fields=["source_type", "source_name", "field_name", "raw_metadata"]
    )

    hits = results[0]
    if not hits:
        print("\n>> 分析结果：抱歉，没有找到相关的概念。")
        return

    top1 = hits[0]

    # 检查是否存在歧义
    is_ambiguous = False
    if len(hits) > 1:
        top2 = hits[1]
        if top1.distance > SIMILARITY_THRESHOLD and \
           top2.distance > SIMILARITY_THRESHOLD and \
           (top1.distance - top2.distance) < SCORE_GAP_THRESHOLD:
            is_ambiguous = True

    if is_ambiguous:
        print("\n>> 分析结果：您的问题可能存在多种解释，请您澄清：")
        for i, hit in enumerate(hits[:2]):
            entity = hit.entity
            source_type_cn = "档案" if entity.get('source_type') == "ARCHIVE" else "事件"
            print(
                f"  {i+1}. 您是指 [{source_type_cn}] '{entity.get('source_name')}' "
                f"中的属性 '{entity.get('field_name')}' "
                f"({entity.get('raw_metadata').get('desc')}) 吗？"
            )
    else:
        print("\n>> 分析结果：已为您精准对齐到以下概念：")
        entity = top1.entity
        source_type_cn = "档案" if entity.get('source_type') == "ARCHIVE" else "事件"
        metadata = entity.get('raw_metadata')

        print(f"   - 来源类型: {source_type_cn} ({entity.get('source_type')})")
        print(f"   - 来源名称: {entity.get('source_name')}")
        print(f"   - 属性字段: {entity.get('field_name')}")
        print(f"   - 属性描述: {metadata.get('desc')}")
        if 'enums' in metadata:
            print(f"   - 枚举值: {', '.join(metadata['enums'])}")
        print(f"   - (匹配分数: {top1.distance:.4f})")

# --- 4. 主程序入口 ---
if __name__ == "__main__":
    # 初始化
    collection = setup_milvus()
    model = get_embedding_model()
    prepare_and_insert_data(collection, model, MOCK_METADATA)

    # 加载 collection 到内存
    print("正在加载 Collection 到内存...")
    collection.load()

    print("\n" + "="*50)
    print(" 概念对齐引擎已就绪，请输入您的自然语言查询。")
    print(" 输入 'exit' 或 'quit' 退出。")
    print("="*50)
    print("您可以尝试以下查询：")
    print("  - '用户在哪个国家'")
    print("  - '发货到中国'")
    print("  - '包裹从哪里发货'")
    print("  - 'country' (这个会触发歧义)")
    print("-" * 50)

    while True:
        try:
            user_query = input("\n[您想查询什么?]: ")
            if user_query.lower() in ['exit', 'quit']:
                break
            if not user_query:
                continue

            search_and_analyze(collection, model, user_query)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"发生错误: {e}")

    # 清理资源
    connections.disconnect("default")
    print("\n程序已退出。")