"""测试改进后的向量化效果"""
# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 引入优化函数
def extract_key_info(description, source_name, field_name, raw_metadata):
    """
    从各字段中提取关键信息，生成优化的向量化文本
    """
    import re
    
    # 提取description中的关键信息
    key_parts = []
    
    # 添加source_name和field_name
    if source_name and source_name != 'unknown':
        key_parts.append(source_name)
    
    if field_name:
        key_parts.append(field_name)
    
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

# 加载模型
model_name = 'paraphrase-multilingual-mpnet-base-v2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(model_name, device=device)

# 测试数据
test_data = {
    'description': '这是一个来源为EVENT，名为"购买渠道"的下的属性字段，名为purchase_channel_name。它的含义是"购买渠道"。可选值为：DOUYIN,JD,O2O,TMALL,VIP,YOUZAN',
    'source_name': '购买渠道',
    'field_name': 'purchase_channel_name',
    'raw_metadata': {'desc': '购买渠道', 'type': 'String'}
}

# 原始方式：只使用description
original_text = test_data['description']

# 优化方式：提取关键信息
optimized_text = extract_key_info(
    test_data['description'],
    test_data['source_name'],
    test_data['field_name'],
    test_data['raw_metadata']
)

print("原始文本（仅description）：")
print(f"  {original_text[:100]}...")
print()
print("优化后文本：")
print(f"  {optimized_text}")
print()

# 测试查询
queries = [
    "DOUYIN买的的",
    "抖音购买",
    "购买渠道",
    "天猫",
    "TMALL购买"
]

# 生成向量
original_embedding = model.encode(original_text)
optimized_embedding = model.encode(optimized_text)
query_embeddings = model.encode(queries)

print("="*60)
print("相似度对比测试")
print("="*60)

for i, query in enumerate(queries):
    # 计算相似度
    sim_original = cosine_similarity([query_embeddings[i]], [original_embedding])[0][0]
    sim_optimized = cosine_similarity([query_embeddings[i]], [optimized_embedding])[0][0]
    
    improvement = ((sim_optimized - sim_original) / sim_original) * 100 if sim_original > 0 else 0
    
    print(f"\n查询: {query}")
    print(f"  原始方式相似度: {sim_original:.4f}")
    print(f"  优化方式相似度: {sim_optimized:.4f}")
    print(f"  提升: {improvement:+.1f}%")