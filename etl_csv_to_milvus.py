import pandas as pd
import json
import torch
from create_milvus_collection import EVENT_COLLECTION_NAME
from pymilvus import connections, utility, Collection
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm
import numpy as np

# --- 1. 配置信息 (Configuration) ---
# Milvus 连接配置
MILVUS_ALIAS = "ipinyou_milvus"  # 为连接起一个别名，方便管理
MILVUS_HOST = "172.28.9.45"
MILVUS_PORT = "19530"
MILVUS_DATABASE = "default"  # 指定数据库名称，默认为 "default"

COLLECTION_NAME = "Pampers_metadata"
# EVENT_COLLECTION_NAME = "Pampers_Event"
# EVENT_METADATA_COLLECTION_NAME = "Pampers_Event_metadata"  # 事件专用的 Collection
# PROFILE_METADATA_COLLECTION_NAME = "Pampers_Profile_metadata"  # 事件专用的 Collection

VECTOR_DIMENSION = 1024  # BGE-M3 模型的输出维度
# 数据源和模型配置
CSV_FILE_PATH = "./pampers_metadata.csv"  # <-- 请确保您的 CSV 文件路径正确
EMBEDDING_MODEL = 'BAAI/bge-m3' # BGE-M3 多语言模型
BATCH_SIZE = 128  # 批量处理数据的大小，可以根据您的内存进行调整

# --- 2. 核心功能函数 ---

def extract_key_info(description, source_name, idname, raw_metadata):
    """
    从各字段中提取关键信息，生成优化的向量化文本
    """
    import re

    # 提取description中的关键信息
    key_parts = []

    # 添加source_name和idname
    if source_name and source_name != 'unknown':
        key_parts.append(source_name)

    if idname:
        key_parts.append(idname)

    # 从description中提取含义和可选值
    if description:
        # 提取"它的含义是"后面的内容
        meaning_match = re.search(r'它的含义是[""](.*?)[""。]', description)
        if meaning_match:
            key_parts.append(meaning_match.group(1))

        # 提取"可选值为"后面的内容
        values_match = re.search(r'可选值为[：:](.*?)(?:$|。)', description)
        if values_match:
            values = values_match.group(1)
            if values != '无':
                # 将可选值分开添加，增加匹配权重
                value_list = [v.strip() for v in values.split(',')]
                key_parts.extend(value_list)

    # 从raw_metadata中提取desc
    if isinstance(raw_metadata, dict) and 'desc' in raw_metadata:
        key_parts.append(raw_metadata['desc'])

    # 组合关键信息，去重
    seen = set()
    unique_parts = []
    for part in key_parts:
        if part and part not in seen:
            seen.add(part)
            unique_parts.append(part)

    # 生成优化的文本
    optimized_text = ' '.join(unique_parts)

    # 如果优化后的文本为空，使用原始description
    if not optimized_text.strip():
        optimized_text = description if description else ''

    return optimized_text

def etl_process(csv_path):
    """
    执行从 CSV 文件到 Milvus 的完整 ETL 流程。
    """
    print("--- 开始执行 CSV 到 Milvus 的 ETL 任务 ---")

    collection = None
    model = None

    # 必需的列
    required_columns = ['concept_id', 'source_type', 'source', 'source_name', 'idname', 'raw_metadata', 'description']

    try:
        # 步骤 1: 加载 BGE-M3 Embedding 模型
        print(f"[*] 正在加载 BGE-M3 Embedding 模型: '{EMBEDDING_MODEL}'...")
        # BGE-M3 模型会自动检测并使用GPU
        use_fp16 = torch.cuda.is_available()  # 如果有GPU，使用fp16加速
        print(f"[*] 使用FP16: {use_fp16}")
        model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=use_fp16)
        print("[+] BGE-M3 Embedding 模型加载成功。")

        # 步骤 2: 连接 Milvus 并获取 Collection 对象
        print(f"[*] 正在连接到 Milvus: {MILVUS_HOST}:{MILVUS_PORT}...")
        connections.connect(alias=MILVUS_ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)

        if not utility.has_collection(COLLECTION_NAME, using=MILVUS_ALIAS):
            print(f"[X] 错误: Collection '{COLLECTION_NAME}' 不存在。请先创建它。")
            return

        collection = Collection(name=COLLECTION_NAME, using=MILVUS_ALIAS)
        print(f"[+] 成功连接到 Collection '{COLLECTION_NAME}'。")

        # 步骤 3: 使用 pandas 读取和分块处理 CSV
        print(f"[*] 正在以分块方式读取 CSV 文件: {csv_path}...")

        # 首先验证 CSV 文件的列
        first_chunk = pd.read_csv(csv_path, nrows=1, encoding='utf-8')
        missing_columns = set(required_columns) - set(first_chunk.columns)
        if missing_columns:
            print(f"[X] 错误: CSV 文件缺少必需的列: {missing_columns}")
            return
        print(f"[+] CSV 文件包含所有必需的列")

        total_rows_inserted = 0
        total_chunks = 0
        skipped_rows = 0

        # 使用 chunksize 来避免一次性将大文件读入内存
        for chunk_df in tqdm(pd.read_csv(csv_path, chunksize=BATCH_SIZE, encoding='utf-8'), desc="处理 CSV 文件块"):
            total_chunks += 1

            # 步骤 3.1: 数据清洗和验证
            # 处理缺失值
            chunk_df = chunk_df.dropna(subset=['description'])  # 删除 description 为空的行
            chunk_df['description'] = chunk_df['description'].fillna('')  # 确保没有 NaN
            chunk_df['description'] = chunk_df['description'].astype(str)  # 转换为字符串

            if chunk_df.empty:
                print(f"[!] 警告: 第 {total_chunks} 块数据在清洗后为空，跳过")
                continue

            # 步骤 3.2: 生成优化的向量化文本
            # 先解析raw_metadata以便使用
            raw_metadata_list = []
            for meta_str in chunk_df['raw_metadata'].tolist():
                try:
                    if pd.isna(meta_str) or meta_str == '':
                        raw_metadata_list.append({})
                    else:
                        raw_metadata_list.append(json.loads(meta_str))
                except (json.JSONDecodeError, TypeError):
                    raw_metadata_list.append({})

            # 生成优化的向量化文本
            optimized_texts = []
            for idx, row in chunk_df.iterrows():
                relative_idx = idx - chunk_df.index[0]  # 获取相对索引
                optimized_text = extract_key_info(
                    row['description'],
                    row['source_name'],
                    row['idname'],
                    raw_metadata_list[relative_idx]
                )
                optimized_texts.append(optimized_text)

            descriptions = optimized_texts  # 使用优化后的文本进行向量化

            # 步骤 3.3: 批量进行文本向量化 (核心转换步骤)
            # 使用BGE-M3的dense embeddings
            embeddings = model.encode(descriptions)['dense_vecs']

            # 步骤 3.4: 验证数据完整性
            # raw_metadata已经在前面解析过了，这里只需要验证
            if len(embeddings) != len(chunk_df):
                print(f"[!] 警告: 第 {total_chunks} 块的向量数量与数据行数不匹配，跳过")
                skipped_rows += len(chunk_df)
                continue

            # 步骤 3.5: 准备插入 Milvus 的数据
            # 处理其他字段的缺失值
            chunk_df['concept_id'] = chunk_df['concept_id'].fillna('').astype(str)
            chunk_df['source_type'] = chunk_df['source_type'].fillna('unknown').astype(str)
            chunk_df['source'] = chunk_df['source'].fillna('unknown').astype(str)
            chunk_df['source_name'] = chunk_df['source_name'].fillna('unknown').astype(str)
            chunk_df['idname'] = chunk_df['idname'].fillna('').astype(str)

            entities_to_insert = [
                chunk_df['concept_id'].tolist(),
                chunk_df['source_type'].tolist(),
                chunk_df['source'].tolist(),
                chunk_df['source_name'].tolist(),
                chunk_df['idname'].tolist(),
                raw_metadata_list,
                embeddings.tolist()  # 转换为列表格式
            ]

            # 步骤 3.6: 批量插入数据到 Milvus
            try:
                insert_result = collection.insert(entities_to_insert)
                total_rows_inserted += insert_result.insert_count
            except Exception as e:
                print(f"[!] 警告: 第 {total_chunks} 块插入失败: {e}")
                skipped_rows += len(chunk_df)
                continue

        # 步骤 4: 刷新 Milvus Collection
        print("\n[*] 所有数据块已处理完毕，正在刷新 Milvus 以确保数据可见...")
        if collection:
            collection.flush()
            print(f"[+] 刷新完成。总共成功插入 {total_rows_inserted} 条数据。")
            if skipped_rows > 0:
                print(f"[!] 跳过了 {skipped_rows} 条无效数据")

            # 步骤 5: 打印统计信息 (可选)
            print(f"[*] 当前 Collection '{COLLECTION_NAME}' 中共有 {collection.num_entities} 条实体。")

    except FileNotFoundError:
        print(f"[X] 错误: CSV 文件未找到，请检查路径 '{csv_path}' 是否正确。")
    except pd.errors.EmptyDataError:
        print(f"[X] 错误: CSV 文件为空或格式不正确")
    except Exception as e:
        print(f"\n[X] ETL 过程中发生严重错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 步骤 6: 清理资源
        # 释放模型占用的内存
        if model is not None:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 断开连接
        try:
            if connections.has_connection(MILVUS_ALIAS):
                connections.disconnect(MILVUS_ALIAS)
                print("\n[*] 与 Milvus 的连接已断开。")
        except Exception as e:
            print(f"[!] 断开连接时出现问题: {e}")

        print("--- ETL 任务结束 ---")

# --- 脚本执行入口 ---
if __name__ == "__main__":
    etl_process(CSV_FILE_PATH)