
import json
import os
import traceback
from typing import List, Union, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

import requests
import weaviate
from openai import OpenAI
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
import time
import numpy
from weaviate.auth import AuthApiKey
from weaviate.client import WeaviateClient

# configurations
if_overwrite_existed_collection = False
if_benchmark_in_concurrent_mode = False
if_include_query_embedding_in_time_measuring = False
collection_name = "benz_20250521"
query_list = [
    # "30万左右的车"
    # "E级什么配置有车载导航，能语音唤醒",
    # "E级顶配是哪款",
    "有安全气囊的车有哪些？"
]

def contains_chinese(text: str = "") -> bool:
    """检查字符串是否包含中文字符"""
    for char in text:
        # 中文字符的 Unicode 范围
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def split_json_if_possible(obj: Optional[Dict]) -> List[str]:
    if obj is None:
        return list()
    if len(json.dumps(obj)) < 8192:
        return [json.dumps(obj)]
    else:
        __rs_ls = list()
        __en_keys: List[str] = list(filter(lambda x: not contains_chinese(text=x), obj.keys()))
        __ch_keys: List[str] = list(filter(lambda x: contains_chinese(text=x), obj.keys()))
        __n = int(len(json.dumps(obj)) / 8192.0) + 1
        __b = int(len(__ch_keys) / __n) if int(len(__ch_keys) / __n) > 0 else 2
        for __i in range(__n):
            __tmp_ch_keys = __ch_keys[__i * __b : (__i + 1) * __b]
            if len(__tmp_ch_keys) == 0:
                continue
            __tmp_obj = dict()
            for __k in __en_keys:
                __tmp_obj[__k] = obj.get(__k)
            for __k in __tmp_ch_keys:
                __tmp_obj[__k] = obj.get(__k)
            __rs_ls.append(__tmp_obj)
        __rs_ls = list(map(lambda x: json.dumps(x), __rs_ls))
        return __rs_ls

def get_text_embedding(text: Union[str, List[str]]) -> List[List[float]]:
    client = OpenAI(
        api_key="sk-08110e9a6b3b46d4893cdb54bb43930c",  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
    )

    completion = client.embeddings.create(
        model="text-embedding-v3",
        input=text,
        dimensions=1024,
        encoding_format="float"
    )

    ls = json.loads(completion.model_dump_json()).get("data")
    return list(map(lambda x: x["embedding"], ls))

class MilvusWorker:
    __default_connection_name: str = "default"

    # 连接到Milvus服务
    @staticmethod
    def connect_to_milvus():
        try:
            # 连接到本地Milvus服务，默认端口为19530
            connections.connect(
                alias=MilvusWorker.__default_connection_name,
                host='localhost',
                port='19530'
            )
            print("成功连接到Milvus服务")
        except Exception as e:
            print(f"连接失败: {e}")


    # 查看所有collection
    @staticmethod
    def list_all_collections():
        try:
            # 获取所有collection的名称
            collections = utility.list_collections()
            print("所有存在的collection:", collections)
            return collections
        except Exception as e:
            print(f"获取collection列表失败: {e}")
            return []


    # 检查特定collection是否存在
    @staticmethod
    def check_collection_exists(collection_name):
        try:
            # 检查指定名称的collection是否存在
            exists = utility.has_collection(collection_name)
            print(f"collection '{collection_name}' 是否存在:", exists)
            return exists
        except Exception as e:
            print(f"检查collection存在性失败: {e}")
            return False


    # 获取collection信息
    @staticmethod
    def get_collection_info(collection_name):
        try:
            if not MilvusWorker.check_collection_exists(collection_name):
                print(f"collection '{collection_name}' 不存在")
                return None

            # 获取collection对象
            collection = Collection(name=collection_name)

            # 加载collection到内存
            collection.load()

            # 获取collection的schema
            schema = collection.schema
            print(f"collection '{collection_name}' 的schema:")
            for field in schema.fields:
                print(f"- 字段名: {field.name}, 类型: {field.dtype}, 是否主键: {field.is_primary}")

            # 获取collection的行数
            num_entities = collection.num_entities
            print(f"collection '{collection_name}' 的行数:", num_entities)

            return {
                "schema": schema,
                "num_entities": num_entities
            }
        except Exception as e:
            print(f"获取collection信息失败: {e}")
            return None

    @staticmethod
    def query_collection_with_vectors_no_check(
            collection: Collection = None,
            limit: int = 10,
            query_vectors: Optional[List[float]] = ""
    ):
        try:

            # 定义搜索参数
            search_params = {
                "metric_type": "L2",  # 距离度量方式，可选L2、IP等
                "params": {"nprobe": 10},  # 搜索参数
            }

            # 执行向量搜索
            return collection.search(
                data=[query_vectors],  # 搜索向量
                anns_field="embedding",  # 向量字段名，需要根据实际情况修改
                param=search_params,
                limit=limit,  # 返回结果数量
                output_fields=["json_str"]  # 需要返回的其他字段
            )
        except Exception as e:
            print(f"查询collection失败: {e}")
            return None

    @staticmethod
    def query_collection_with_vectors(collection_name: str = "",
                                      limit: int = 10,
                                      query_vectors: Optional[List[float]] = ""):
        try:
            if not MilvusWorker.check_collection_exists(collection_name):
                print(f"collection '{collection_name}' 不存在")
                return None

            # 获取collection对象
            collection = Collection(name=collection_name, using=MilvusWorker.__default_connection_name)

            # 加载collection到内存
            collection.load(timeout=5)

            # 定义搜索参数
            search_params = {
                "metric_type": "L2",  # 距离度量方式，可选L2、IP等
                "params": {"nprobe": 10},  # 搜索参数
            }

            # 执行向量搜索
            results = collection.search(
                data=[query_vectors],  # 搜索向量
                anns_field="embedding",  # 向量字段名，需要根据实际情况修改
                param=search_params,
                limit=limit,  # 返回结果数量
                output_fields=["json_str"]  # 需要返回的其他字段
            )

            # 处理搜索结果
            print(f"在collection '{collection_name}' 中查询的结果:")
            for hits in results:
                for hit in hits:
                    print(f"- ID: {hit.id}, 距离: {hit.distance}, "
                          f"其他字段: {json.loads(hit.entity.get('json_str')).get('sub_car_name')}, "
                          f"{json.loads(hit.entity.get('json_str')).get('sub_car_id')}")

            return results
        except Exception as e:
            print(f"查询collection失败: {e}")
            return None

    # 查询collection中的数据
    @staticmethod
    def query_collection(collection_name: str = "", limit: int = 10, queries: Union[str, List[str]] = ""):
        # 生成随机向量作为查询向量（假设collection中有向量字段）
        # 这里需要根据实际的向量维度进行调整
        if isinstance(queries, str) is True:
            queries = [queries]
        query_vectors = get_text_embedding(text=queries)
        return list(map(lambda x: MilvusWorker.query_collection_with_vectors(
            collection_name=collection_name,
            limit=limit,
            query_vectors=x),
                        query_vectors))

    @staticmethod
    def create_collection_and_insert_data(collection_name: str = ""):
        # 创建 collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="json_str", dtype=DataType.VARCHAR, max_length=65535),  # 存储 JSON 字符串
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)  # 向量维度根据模型调整
        ]
        schema = CollectionSchema(fields=fields)
        collection = Collection(name=collection_name, schema=schema)

        # 创建索引
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 8, "ef_construction": 64}
        }
        collection.create_index(field_name="embedding", index_params=index_params)

        __target_files = list()
        __output_json_directory = "./source_json_files"
        for path, dirs, files in os.walk(__output_json_directory):
            for f in files:
                __file = os.path.join(path, f)
                if (os.path.isfile(__file) is True
                        and (str(__file).endswith(".json"))):
                    __target_files.append(__file)

        print(__target_files)
        print(f'found json file number: {len(__target_files)}')

        for local_rs_file in __target_files:
            with open(local_rs_file, "r", encoding="utf-8") as f:
                __j = json.load(f)

            print(f'got {len(__j)} rows data from file {local_rs_file}.')
            __new_j = list()
            __embedding_ls = list()
            __embedding_batch_size = 9
            __embedding_batch_index = 0
            while True:
                __tmp_j = __j[__embedding_batch_index * __embedding_batch_size :
                              (__embedding_batch_index + 1) * __embedding_batch_size]
                if len(__tmp_j) == 0:
                    break
                for __tmp_obj in __tmp_j:
                    __ls = split_json_if_possible(obj=__tmp_obj)
                    __new_j += __ls
                    for __tmp_obj_str in __ls:
                        __embedding_ls += get_text_embedding(text=__tmp_obj_str)
                __embedding_batch_index += 1
                print(f'succeeded got {len(__embedding_ls)} embedding rows.')
            __rs = collection.insert(
                [
                    __new_j,
                    __embedding_ls
                ]
            )
            print(f'succeeded inserting {__rs.insert_count} rows.')

    @staticmethod
    def __concurrent_bench_mark_worker(number: int,
                                       query_vector: Optional[List[float]],
                                       collection: Collection,
                                       query_str: str) -> Tuple[List[List[str]], List[float]]:
        print(f'the {number} number of concurrent bench mark worker has been started.')
        __time_total_ls = list()
        __query_rs_ls = list()
        try:
            __collection = collection
            __test_size: int = 100

            for i in range(__test_size):
                __time1 = time.time()
                if if_include_query_embedding_in_time_measuring is True:
                    query_vector = get_text_embedding(text=query_str)[0]
                __query_rs = MilvusWorker.query_collection_with_vectors_no_check(
                    collection=__collection,
                    query_vectors=query_vector
                )
                if __query_rs is None:
                    raise Exception('got none result from milvus instance.')
                __time2 = time.time()
                __time_total_ls.append(__time2 - __time1)
                __query_rs_ls.append(
                    [[
                         f'{json.loads(__query_rs[j][i].entity.get("json_str")).get("sub_car_name")}@{json.loads(__query_rs[j][i].entity.get("json_str")).get("sub_car_id")}'
                         for i in range(len(__query_rs[j]))]
                     for j in range(len(__query_rs))])
        except Exception as __e:
            print(''.join(traceback.format_exception(__e)))
            print(''.join(traceback.format_stack()))
        print(f'the {number} number of concurrent bench mark worker has been finished.')
        return __query_rs_ls, __time_total_ls

    @staticmethod
    def concurrent_bench_mark() -> Tuple[List[List[str]], List[float]]:
        __time_total_ls = list()
        __query_rs_ls = list()
        try:
            # 连接到Milvus
            MilvusWorker.connect_to_milvus()
            if MilvusWorker.check_collection_exists(collection_name=collection_name) is False:
                MilvusWorker.create_collection_and_insert_data(collection_name=collection_name)
            elif if_overwrite_existed_collection is True:
                Collection(name=collection_name).drop()
                if utility.has_collection(collection_name=collection_name) is False:
                    print(f'the collection {collection_name} has already been dropped.')
                MilvusWorker.create_collection_and_insert_data(collection_name=collection_name)

            query_vectors = get_text_embedding(text=query_list)[0]
            __collection = Collection(name=collection_name)
            __collection.load(timeout=5)

            with ThreadPoolExecutor(max_workers=5) as executor:
                # 提交任务
                futures = [executor.submit(MilvusWorker.__concurrent_bench_mark_worker, i, query_vectors, __collection, query_list[0]) for i in range(5)]

                # 获取结果
                for future in futures:
                    __tmp_query_rs_ls, __tmp_time_total_ls = future.result()
                    __query_rs_ls += __tmp_query_rs_ls
                    __time_total_ls += __tmp_time_total_ls

            connections.remove_connection(alias=MilvusWorker.__default_connection_name)
            connections.disconnect(alias=MilvusWorker.__default_connection_name)
            print(f'the connection has already been removed.')
        except Exception as __e:
            print(''.join(traceback.format_exception(__e)))
            print(''.join(traceback.format_stack()))
            connections.remove_connection(alias=MilvusWorker.__default_connection_name)
            connections.disconnect(alias=MilvusWorker.__default_connection_name)
            print(f'the connection has already been removed.')
        return __query_rs_ls, __time_total_ls

    @staticmethod
    def bench_mark() -> Tuple[List[List[str]], List[float]]:
        __time_total_ls = list()
        __query_rs_ls = list()
        try:
            # 连接到Milvus
            MilvusWorker.connect_to_milvus()
            if MilvusWorker.check_collection_exists(collection_name=collection_name) is False:
                MilvusWorker.create_collection_and_insert_data(collection_name=collection_name)
            elif if_overwrite_existed_collection is True:
                Collection(name=collection_name).drop()
                if utility.has_collection(collection_name=collection_name) is False:
                    print(f'the collection {collection_name} has already been dropped.')
                MilvusWorker.create_collection_and_insert_data(collection_name=collection_name)

            query_vectors = get_text_embedding(text=query_list)[0]
            __collection = Collection(name=collection_name)
            __collection.load(timeout=5)

            __test_size: int = 100

            for i in range(__test_size):
                __time1 = time.time()
                if if_include_query_embedding_in_time_measuring is True:
                    query_vectors = get_text_embedding(text=query_list)[0]
                __query_rs = MilvusWorker.query_collection_with_vectors_no_check(
                    collection=__collection,
                    query_vectors=query_vectors
                )
                if __query_rs is None:
                    raise Exception('got none result from milvus instance.')
                __time2 = time.time()
                __time_total_ls.append(__time2 - __time1)
                __query_rs_ls.append(
                    [[
                         f'{json.loads(__query_rs[j][i].entity.get("json_str")).get("sub_car_name")}@{json.loads(__query_rs[j][i].entity.get("json_str")).get("sub_car_id")}'
                         for i in range(len(__query_rs[j]))]
                     for j in range(len(__query_rs))])
            connections.remove_connection(alias=MilvusWorker.__default_connection_name)
            connections.disconnect(alias=MilvusWorker.__default_connection_name)
            print(f'the connection has already been removed.')
        except Exception as __e:
            print(''.join(traceback.format_exception(__e)))
            print(''.join(traceback.format_stack()))
            connections.remove_connection(alias=MilvusWorker.__default_connection_name)
            connections.disconnect(alias=MilvusWorker.__default_connection_name)
            print(f'the connection has already been removed.')
        return __query_rs_ls, __time_total_ls


class WeaviateWorker:
    __client: Optional[WeaviateClient] = None

    @staticmethod
    def connect_to_weaviate():
        WeaviateWorker.__client = weaviate.connect_to_local(auth_credentials=AuthApiKey("test-secret-key"))

    @staticmethod
    def check_collection_exists(collection_name: str) -> bool:
        """
        检查集合是否存在
        :param collection_name: 集合名称
        :return: True 或 False
        """
        try:
            collections = list(WeaviateWorker.__client.collections.list_all().keys())
            print(f'found all collections: {collections}')
            collections = list(map(lambda x: str(x).lower(), collections))
            return collection_name in collections
        except Exception as e:
            print(f"检查集合异常: {e}")
            return False

    @staticmethod
    def create_collection(collection_name: str):
        """
        创建集合
        :param collection_name: 集合名称
        """
        collection_obj = {
            "class": collection_name,
            "description": "A collection for product information",
            "vectorizer": "none",  # 假设你会上传自己的向量
            "vectorIndexType": "hnsw",
            "vectorIndexConfig": {
                "distance": "cosine",
                "efConstruction": 200,
                "maxConnections": 64
            },
            "properties": [
                {
                    "name": "json_str",
                    "description": "The source data content",
                    "dataType": ["text"],
                    "tokenization": "word",
                    "indexFilterable": True,
                    "indexSearchable": True
                }
            ]
        }
        try:
            WeaviateWorker.__client.collections.create_from_dict(collection_obj)
            print(f"创建集合 '{collection_name}' 成功.")
        except weaviate.exceptions.UnexpectedStatusCodeException as e:
            print(f"创建集合异常: {e}")

    @staticmethod
    def save_documents(collection_name: str,
                       documents: List[str],
                       vectors: List[List[float]]) -> List[str]:
        """
        向集合中插入数据
        :param vectors:
        :param collection_name: 集合名称
        :param documents: 文档列表
        """
        __rs_uuid_ls = list()
        collection = WeaviateWorker.__client.collections.get(collection_name)
        for i in range(len(documents)):
            content = documents[i]  # 假设文档是简单的字符串
            properties = {
                "json_str": content
            }
            try:
                uuid = collection.data.insert(properties=properties, vector=vectors[i])
                print(f"文档添加内容: {content[:30]}..., uuid: {uuid}")
                __rs_uuid_ls.append(uuid)
            except Exception as e:
                print(f"添加文档异常: {e}")
        return __rs_uuid_ls

    @staticmethod
    def query_collection(collection_name: str, query: str, limit: int = 10) -> List[str]:
        """
        从集合中查询数据
        :param collection_name: 集合名称
        :param query: 查询字符串
        :param limit: 返回的结果数量
        :return: 查询结果列表
        """
        vector = get_text_embedding(text=query)  # 假设这是你的查询向量
        return WeaviateWorker.query_vector_collection(collection_name=collection_name, vector=vector[0], limit=limit)

    @staticmethod
    def query_vector_collection(collection_name: str, vector: List[float], limit: int = 10) -> List[str]:
        """
        从集合中查询数据
        :param collection_name: 集合名称
        :param vector: 查询向量
        :param limit: 返回的结果数量
        :return: 查询结果列表
        """
        collection = WeaviateWorker.__client.collections.get(collection_name)
        response = collection.query.near_vector(
            near_vector=vector,
            limit=limit
        )
        documents = [res.properties['json_str'] for res in response.objects]
        return documents

    @staticmethod
    def get_collection(collection_name: str = "") -> Optional[weaviate.collections.collection.sync.Collection]:
        if collection_name is None or collection_name == "":
            return None
        return WeaviateWorker.__client.collections.get(collection_name)

    @staticmethod
    def query_vector_collection_with_collection(collection: weaviate.collections.collection.sync.Collection,
                                                vector: List[float],
                                                limit: int = 10) -> List[str]:
        """
        从集合中查询数据
        :param collection: 集合
        :param vector: 查询向量
        :param limit: 返回的结果数量
        :return: 查询结果列表
        """
        response = collection.query.near_vector(
            near_vector=vector,
            limit=limit
        )
        documents = [res.properties['json_str'] for res in response.objects]
        return documents

    @staticmethod
    def delete_collection(collection_name: str):
        """
        删除集合
        :param collection_name: 集合名称
        """
        try:
            WeaviateWorker.__client.collections.delete(collection_name)
            print(f"删除集合 '{collection_name}' 成功.")
        except Exception as e:
            print(f"删除集合异常: {e}")

    @staticmethod
    def create_collection_and_insert_data(collection_name: str = ""):
        if WeaviateWorker.check_collection_exists(collection_name=collection_name) is False:
            WeaviateWorker.create_collection(collection_name=collection_name)

        __target_files = list()
        __output_json_directory = "./source_json_files"
        for path, dirs, files in os.walk(__output_json_directory):
            for f in files:
                __file = os.path.join(path, f)
                if (os.path.isfile(__file) is True
                        and (str(__file).endswith(".json"))):
                    __target_files.append(__file)

        print(__target_files)
        print(f'found json file number: {len(__target_files)}')

        for local_rs_file in __target_files:
            with open(local_rs_file, "r", encoding="utf-8") as f:
                __j = json.load(f)

            print(f'got {len(__j)} rows data from file {local_rs_file}.')
            __new_j = list()
            __embedding_ls = list()
            __embedding_batch_size = 9
            __embedding_batch_index = 0
            while True:
                __tmp_j = __j[__embedding_batch_index * __embedding_batch_size:
                              (__embedding_batch_index + 1) * __embedding_batch_size]
                if len(__tmp_j) == 0:
                    break
                for __tmp_obj in __tmp_j:
                    __ls = split_json_if_possible(obj=__tmp_obj)
                    __new_j += __ls
                    for __tmp_obj_str in __ls:
                        __embedding_ls += get_text_embedding(text=__tmp_obj_str)
                __embedding_batch_index += 1
                print(f'succeeded got {len(__embedding_ls)} embedding rows.')
            __rs = WeaviateWorker.save_documents(collection_name=collection_name, documents=__new_j, vectors=__embedding_ls)
            print(f'succeeded inserting {len(__rs)} rows with input data size {len(__new_j)}.')

    @staticmethod
    def __concurrent_bench_mark_worker(number: int,
                                       query_vector: Optional[List[float]],
                                       collection: weaviate.collections.collection.sync.Collection,
                                       query_str: str) -> Tuple[List[List[str]], List[float]]:
        print(f'the {number} number of concurrent bench mark worker has been started.')
        __time_total_ls = list()
        __query_rs_ls = list()
        __test_size: int = 100
        __time_total_ls = list()
        __query_rs_ls = list()
        for i in range(__test_size):
            __time1 = time.time()
            if if_include_query_embedding_in_time_measuring is True:
                query_vector = get_text_embedding(text=query_str)[0]
            __query_rs = WeaviateWorker.query_vector_collection_with_collection(
                collection=collection,
                vector=query_vector
            )
            __time2 = time.time()
            __time_total_ls.append(__time2 - __time1)
            __query_rs_ls.append(
                [[
                    f'{json.loads(__query_rs[i]).get("sub_car_name")}@{json.loads(__query_rs[i]).get("sub_car_id")}'
                    for i in range(len(__query_rs))]])
        print(f'the {number} number of concurrent bench mark worker has been finished.')
        return __query_rs_ls, __time_total_ls

    @staticmethod
    def concurrent_bench_mark() -> Tuple[List[List[str]], List[float]]:
        # 连接到Milvus=
        WeaviateWorker.connect_to_weaviate()
        if WeaviateWorker.check_collection_exists(collection_name=collection_name) is False:
            WeaviateWorker.create_collection_and_insert_data(collection_name=collection_name)
        elif if_overwrite_existed_collection is True:
            WeaviateWorker.delete_collection(collection_name=collection_name)
            WeaviateWorker.create_collection_and_insert_data(collection_name=collection_name)

        query_vectors = get_text_embedding(text=query_list)[0]
        __collection: weaviate.collections.collection.sync.Collection = WeaviateWorker.get_collection(
            collection_name=collection_name)

        __query_rs_ls = list()
        __time_total_ls = list()
        with ThreadPoolExecutor(max_workers=5) as executor:
                # 提交任务
                futures = [executor.submit(WeaviateWorker.__concurrent_bench_mark_worker, i, query_vectors, __collection, query_list[0]) for i in range(5)]

                # 获取结果
                for future in futures:
                    __tmp_query_rs_ls, __tmp_time_total_ls = future.result()
                    __query_rs_ls += __tmp_query_rs_ls
                    __time_total_ls += __tmp_time_total_ls
        WeaviateWorker.__client.close()
        return __query_rs_ls, __time_total_ls

    @staticmethod
    def bench_mark() -> Tuple[List[List[str]], List[float]]:
        # 连接到Milvus=
        WeaviateWorker.connect_to_weaviate()
        if WeaviateWorker.check_collection_exists(collection_name=collection_name) is False:
            WeaviateWorker.create_collection_and_insert_data(collection_name=collection_name)
        elif if_overwrite_existed_collection is True:
            WeaviateWorker.delete_collection(collection_name=collection_name)
            WeaviateWorker.create_collection_and_insert_data(collection_name=collection_name)

        query_vectors = get_text_embedding(text=query_list)[0]
        __collection: weaviate.collections.collection.sync.Collection = WeaviateWorker.get_collection(
            collection_name=collection_name)

        __test_size: int = 100
        __time_total_ls: List = list()
        __query_rs_ls: List = list()
        if_print_sample: bool = False
        for i in range(__test_size):
            __time1 = time.time()
            # additional testing.
            if if_include_query_embedding_in_time_measuring is True:
                query_vectors = get_text_embedding(text=query_list)[0]

            __query_rs = WeaviateWorker.query_vector_collection_with_collection(
                collection=__collection,
                vector=query_vectors
            )
            __time2 = time.time()
            __time_total_ls.append(__time2 - __time1)
            # if if_print_sample is False:
            #     print('the beginning of sample output')
            #     print('-----')
            #     print(__query_rs)
            #     print('-----')
            #     print('the endding of sample output')
            #     if_print_sample = True
            __query_rs_ls.append(
                [[
                    f'{json.loads(__query_rs[i]).get("sub_car_name")}@{json.loads(__query_rs[i]).get("sub_car_id")}'
                    for i in range(len(__query_rs))]])
        WeaviateWorker.__client.close()
        time.sleep(3)
        return __query_rs_ls, __time_total_ls


class DifyWorker:
    __target_dataset_id: str = "9c264221-1fb8-43df-869f-6f4d44236253"

    @staticmethod
    def __request_2_dataset(query: Optional[str]) -> List[str]:
        # 替换为实际的数据集ID和API密钥
        dataset_id = DifyWorker.__target_dataset_id
        api_key = "dataset-AqaHd7mp2L48rQCNDe2CtDFW"

        # 请求的URL
        url = f'http://agentplatform.ipinyou.com/v1/datasets/{dataset_id}/retrieve'

        # 请求头
        headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
        }

        # 请求体数据
        data = {
                "query": query,
                "retrieval_model": {
                        # "search_method": "hybrid_search",
                        "search_method": "semantic_search",
                        "reranking_enable": False,
                        "reranking_mode": None,
                        "reranking_model": {
                                "reranking_provider_name": "",
                                "reranking_model_name": ""
                        },
                        "weights": None,
                        "top_k": 10,
                        "score_threshold_enabled": False,
                        "score_threshold": None
                }
        }

        # 将数据转换为JSON字符串
        json_data = json.dumps(data)

        try:
            # 发起POST请求
            response = requests.post(url, headers=headers, data=json_data)

            # 检查响应状态码
            if response.status_code == 200:
                pass
                # print("the request has already succeeded.")
            else:
                print(f"请求失败，状态码: {response.status_code}，响应内容: {response.text}")
        except requests.RequestException as e:
                print(f"请求发生错误: {e}")
        response_data = json.loads(response.text)
        # 提取所需的content字段
        content_list = []
        for record in response_data.get('records', []):
            segment = record.get('segment', {})
            content = segment.get('content', '')
            content = content.replace('\n', '')
            try:
                # 尝试将 content 解析为 Python 对象
                content_obj = json.loads(content)
                content_list.append(content_obj)
            except json.JSONDecodeError:
                # 如果解析失败，直接添加原始内容
                content_list.append(content)

        return content_list

    @staticmethod
    def __concurrent_bench_mark_worker(number: int, query: str) -> Tuple[List[List[str]], List[float]]:
        print(f'the number {number} of concurrent bench mark worker has been started.')

        __test_size: int = 100
        __time_total_ls = list()
        __query_rs_ls = list()
        for i in range(__test_size):
            __time1 = time.time()
            __query_rs = DifyWorker.__request_2_dataset(query=query)

            __time2 = time.time()
            __time_total_ls.append(__time2 - __time1)
            __query_rs_ls.append(__query_rs)
        print(f'the number {number} of concurrent bench mark worker has been finished.')
        return __query_rs_ls, __time_total_ls

    @staticmethod
    def concurrent_bench_mark() -> Tuple[List[List[str]], List[float]]:
        __time_total_ls = list()
        __query_rs_ls = list()
        with ThreadPoolExecutor(max_workers=5) as executor:
                # 提交任务
                futures = [executor.submit(DifyWorker.__concurrent_bench_mark_worker, i, query_list[0]) for i in range(5)]

                # 获取结果
                for future in futures:
                    __tmp_query_rs_ls, __tmp_time_total_ls = future.result()
                    __query_rs_ls += __tmp_query_rs_ls
                    __time_total_ls += __tmp_time_total_ls
        return __query_rs_ls, __time_total_ls

    @staticmethod
    def bench_mark() -> Tuple[List[List[str]], List[float]]:
        query: str = query_list[0]

        __test_size: int = 100
        __time_total_ls = list()
        __query_rs_ls = list()
        for i in range(__test_size):
            __time1 = time.time()
            __query_rs = DifyWorker.__request_2_dataset(query=query)
            __time2 = time.time()
            __time_total_ls.append(__time2 - __time1)
            __query_rs_ls.append(__query_rs)
        return __query_rs_ls, __time_total_ls


# 主函数
if __name__ == "__main__":
    __query_rs_ls = list()
    __time_total_ls = list()
    if if_benchmark_in_concurrent_mode is False:
        # __query_rs_ls, __time_total_ls = DifyWorker.bench_mark()
        # __query_rs_ls, __time_total_ls = MilvusWorker.bench_mark()
        __query_rs_ls, __time_total_ls = WeaviateWorker.bench_mark()
    else:
        # __query_rs_ls, __time_total_ls = DifyWorker.concurrent_bench_mark()
        # __query_rs_ls, __time_total_ls = MilvusWorker.concurrent_bench_mark()
        __query_rs_ls, __time_total_ls = WeaviateWorker.concurrent_bench_mark()


    print(type(__query_rs_ls[0]))
    print(len(__query_rs_ls[0]))
    print(__query_rs_ls[0])
    print(__query_rs_ls[1])
    print(__query_rs_ls[2])
    print(len(__query_rs_ls))
    print(f'耗时中位数是：{numpy.median(__time_total_ls)}')
    print(f'耗时平均数是：{numpy.mean(__time_total_ls)}')
    print(f'耗时最大数是：{numpy.max(__time_total_ls)}')
    print(f'耗时最小数是：{numpy.min(__time_total_ls)}')

    from itertools import chain
    if isinstance(__query_rs_ls[0][0], list) is True:
        __distinct_rs_ls = list(set(list(chain(*list(chain(*__query_rs_ls))))))
        print(f'不同结果数量：{len(__distinct_rs_ls)}')
    else:

        __distinct_rs_ls = set()
        for __q1 in __query_rs_ls:
            for __q2 in __q1:
                __distinct_rs_ls.add(__q2)
        __distinct_rs_ls = list(__distinct_rs_ls)
        print(type(__distinct_rs_ls))
        # for __d in __distinct_rs_ls:
        #     print('-----')
        #     print(__d)
        #     print('-----')
        print(type(__distinct_rs_ls[0]))
        print(f'不同结果数量：{len(__distinct_rs_ls)}')


