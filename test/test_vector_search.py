import json
import torch
from pymilvus import connections, Collection, utility
from FlagEmbedding import BGEM3FlagModel
from typing import List, Dict, Any
import pandas as pd

# --- 配置信息 (与 etl_csv_to_milvus.py 保持一致) ---
MILVUS_ALIAS = "ipinyou_milvus"
MILVUS_HOST = "172.28.9.45"
MILVUS_PORT = "19530"
MILVUS_DATABASE = "default"

COLLECTION_NAME = "Pampers_metadata"
EMBEDDING_MODEL = 'BAAI/bge-m3'
VECTOR_FIELD_NAME = "concept_embedding"  # 向量字段名称

# 搜索参数
DEFAULT_TOP_K = 10  # 返回最相似的前 K 个结果
DEFAULT_METRIC_TYPE ="COSINE"   # 距离度量类型 (L2, IP, COSINE)


class VectorSearcher:
    """向量搜索器类"""

    def __init__(self):
        self.collection = None
        self.model = None
        self.device = None
        self.metric_type = None

    def connect(self):
        """连接到 Milvus 并加载模型"""
        try:
            # 连接 Milvus
            print(f"[*] 正在连接到 Milvus: {MILVUS_HOST}:{MILVUS_PORT}...")
            connections.connect(alias=MILVUS_ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)

            # 检查 Collection 是否存在
            if not utility.has_collection(COLLECTION_NAME, using=MILVUS_ALIAS):
                print(f"[X] 错误: Collection '{COLLECTION_NAME}' 不存在")
                return False

            # 获取 Collection
            self.collection = Collection(name=COLLECTION_NAME, using=MILVUS_ALIAS)
            print(f"[+] 成功连接到 Collection '{COLLECTION_NAME}'")
            print(f"[*] Collection 中有 {self.collection.num_entities} 条数据")

            # 读取索引的度量类型以避免搜索度量不匹配
            try:
                detected_metric = None
                for index in getattr(self.collection, "indexes", []):
                    field_name = getattr(index, "field_name", None)
                    if field_name == VECTOR_FIELD_NAME:
                        params = getattr(index, "params", {})
                        if isinstance(params, str):
                            try:
                                params = json.loads(params)
                            except Exception:
                                params = {}
                        detected_metric = (
                            params.get("metric_type")
                            or (isinstance(params.get("index_param"), dict) and params["index_param"].get("metric_type"))
                            or (isinstance(params.get("params"), dict) and params["params"].get("metric_type"))
                        )
                        if detected_metric:
                            break
                self.metric_type = (detected_metric or DEFAULT_METRIC_TYPE).upper()
            except Exception as e:
                print(f"[!] 读取索引参数失败: {e}")
                self.metric_type = DEFAULT_METRIC_TYPE

            print(f"[*] 使用度量类型: {self.metric_type}")

            # 加载 Collection 到内存
            print("[*] 正在加载 Collection 到内存...")
            self.collection.load()
            print("[+] Collection 已加载")

            # 加载 BGE-M3 Embedding 模型
            print(f"[*] 正在加载 BGE-M3 Embedding 模型: '{EMBEDDING_MODEL}'...")
            use_fp16 = torch.cuda.is_available()
            print(f"[*] 使用FP16: {use_fp16}")
            self.model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=use_fp16)
            print("[+] BGE-M3 Embedding 模型加载成功")

            return True

        except Exception as e:
            print(f"[X] 连接失败: {e}")
            return False

    def search(self, query_text: str, top_k: int = DEFAULT_TOP_K,
               filter_expr: str = None) -> List[Dict[str, Any]]:
        """
        执行向量相似度搜索

        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            filter_expr: 过滤表达式 (可选，如: "source_type == 'web'")

        Returns:
            搜索结果列表
        """
        if not self.collection or not self.model:
            print("[X] 错误: 请先调用 connect() 方法")
            return []

        try:
            # 将查询文本转换为向量 - 使用BGE-M3的dense embeddings
            print(f"\n[*] 查询文本: '{query_text}'")
            print("[*] 正在生成查询向量...")
            query_embedding = self.model.encode([query_text])['dense_vecs']

            # 设置搜索参数
            search_params = {
                "metric_type": (self.metric_type or DEFAULT_METRIC_TYPE),
                "params": {"nprobe": 10}
            }

            # 指定要返回的字段
            output_fields = ["concept_id", "source_type", "source_name",
                           "field_name", "raw_metadata"]

            # 执行搜索
            print(f"[*] 正在搜索最相似的 {top_k} 条记录...")
            results = self.collection.search(
                data=query_embedding.tolist(),
                anns_field=VECTOR_FIELD_NAME,
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=output_fields
            )

            # 处理搜索结果
            search_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        "score": hit.distance,  # 相似度分数
                        "concept_id": hit.entity.get("concept_id"),
                        "source_type": hit.entity.get("source_type"),
                        "source_name": hit.entity.get("source_name"),
                        "field_name": hit.entity.get("field_name"),
                        "raw_metadata": hit.entity.get("raw_metadata")
                    }
                    search_results.append(result)

            print(f"[+] 找到 {len(search_results)} 条相似结果")
            return search_results

        except Exception as e:
            print(f"[X] 搜索失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def batch_search(self, query_texts: List[str], top_k: int = DEFAULT_TOP_K) -> Dict[str, List]:
        """
        批量执行向量搜索

        Args:
            query_texts: 查询文本列表
            top_k: 每个查询返回的结果数量

        Returns:
            字典，键为查询文本，值为搜索结果
        """
        if not self.collection or not self.model:
            print("[X] 错误: 请先调用 connect() 方法")
            return {}

        try:
            print(f"\n[*] 批量查询 {len(query_texts)} 个文本...")

            # 批量生成向量 - 使用BGE-M3的dense embeddings
            query_embeddings = self.model.encode(query_texts)['dense_vecs']

            # 设置搜索参数
            search_params = {
                "metric_type": (self.metric_type or DEFAULT_METRIC_TYPE),
                "params": {"nprobe": 10}
            }

            # 指定要返回的字段
            output_fields = ["concept_id", "source_type", "source_name",
                           "field_name", "raw_metadata"]

            # 执行批量搜索
            results = self.collection.search(
                data=query_embeddings.tolist(),
                anns_field=VECTOR_FIELD_NAME,
                param=search_params,
                limit=top_k,
                output_fields=output_fields
            )

            # 整理结果
            batch_results = {}
            for i, query_text in enumerate(query_texts):
                search_results = []
                for hit in results[i]:
                    result = {
                        "score": hit.distance,
                        "concept_id": hit.entity.get("concept_id"),
                        "source_type": hit.entity.get("source_type"),
                        "source_name": hit.entity.get("source_name"),
                        "field_name": hit.entity.get("field_name"),
                        "raw_metadata": hit.entity.get("raw_metadata")
                    }
                    search_results.append(result)
                batch_results[query_text] = search_results

            print(f"[+] 批量搜索完成")
            return batch_results

        except Exception as e:
            print(f"[X] 批量搜索失败: {e}")
            return {}

    def disconnect(self):
        """断开连接并清理资源"""
        try:
            # 释放 Collection
            if self.collection:
                self.collection.release()
                print("[*] Collection 已从内存中释放")

            # 释放模型
            if self.model:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[*] 模型内存已释放")

            # 断开连接
            if connections.has_connection(MILVUS_ALIAS):
                connections.disconnect(MILVUS_ALIAS)
                print("[*] Milvus 连接已断开")

        except Exception as e:
            print(f"[!] 断开连接时出现问题: {e}")


def print_search_results(results: List[Dict[str, Any]], max_display: int = 5):
    """
    格式化打印搜索结果

    Args:
        results: 搜索结果列表
        max_display: 最多显示的结果数量
    """
    if not results:
        print("\n没有找到相似的结果")
        return

    print(f"\n======= 搜索结果 (显示前 {min(len(results), max_display)} 条) =======")

    for i, result in enumerate(results[:max_display], 1):
        print(f"\n--- 结果 {i} ---")
        print(f"相似度分数: {result['score']:.4f}")
        print(f"Concept ID: {result['concept_id']}")
        print(f"来源类型: {result['source_type']}")
        print(f"来源名称: {result['source_name']}")
        print(f"字段名称: {result['field_name']}")

        # 打印元数据（如果存在）
        if result['raw_metadata']:
            print("元数据:")
            try:
                # 如果是字典，格式化打印
                if isinstance(result['raw_metadata'], dict):
                    for key, value in result['raw_metadata'].items():
                        print(f"  - {key}: {value}")
                else:
                    print(f"  {result['raw_metadata']}")
            except:
                print(f"  {result['raw_metadata']}")


def main():
    """主函数：演示向量搜索功能"""

    # 创建搜索器实例
    searcher = VectorSearcher()

    # 连接到 Milvus
    if not searcher.connect():
        return

    try:
        # ========== 示例 1: 单个查询 ==========
        print("\n" + "="*50)
        print("示例 1: 单个文本查询")
        print("="*50)

        query_text = "baby diaper comfort"  # 修改为您想搜索的内容
        results = searcher.search(query_text, top_k=10)
        print_search_results(results)

        # ========== 示例 2: 带过滤条件的查询 ==========
        print("\n" + "="*50)
        print("示例 2: 带过滤条件的查询")
        print("="*50)

        # 只搜索特定来源类型的数据
        # filter_expr = "source_type == 'web'"  # 根据实际数据调整
        # results = searcher.search(query_text, top_k=5, filter_expr=filter_expr)
        # print_search_results(results)

        # ========== 示例 3: 批量查询 ==========
        print("\n" + "="*50)
        print("示例 3: 批量查询")
        print("="*50)

        query_texts = [
            "baby diaper",
            "soft and comfortable",
            "newborn care"
        ]

        batch_results = searcher.batch_search(query_texts, top_k=5)

        for query, results in batch_results.items():
            print(f"\n查询: '{query}'")
            print(f"找到 {len(results)} 条结果")
            if results:
                print(f"最相似结果的分数: {results[0]['score']:.4f}")

        # ========== 示例 4: 交互式查询 ==========
        print("\n" + "="*50)
        print("示例 4: 交互式查询 (输入 'quit' 退出)")
        print("="*50)

        while True:
            user_query = input("\n请输入查询文本 (或 'quit' 退出): ").strip()
            if user_query.lower() == 'quit':
                break
            if not user_query:
                continue

            results = searcher.search(user_query, top_k=5)
            print_search_results(results, max_display=3)

    finally:
        # 断开连接
        searcher.disconnect()
        print("\n[*] 程序结束")


if __name__ == "__main__":
    main()