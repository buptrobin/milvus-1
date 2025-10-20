"""测试向量相似度问题"""
# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import torch
from FlagEmbedding import BGEM3FlagModel
import numpy as np

# 加载BGE-M3模型
model_name = 'BAAI/bge-m3'
use_fp16 = torch.cuda.is_available()
model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

# 测试文本
query = "DOUYIN买的的"
target = "这是一个来源为EVENT，名为\"购买渠道\"的下的属性字段，名为purchase_channel_name。它的含义是\"购买渠道\"。可选值为：DOUYIN,JD,O2O,TMALL,VIP,YOUZAN"

# 其他对比文本
texts = [
    target,
    "购买渠道：DOUYIN",
    "DOUYIN渠道购买",
    "抖音购买",
    "这是一个购买时间字段",
    "会员ID",
]

# 生成向量 - 使用BGE-M3的dense embeddings
query_embedding = model.encode([query])['dense_vecs'][0]
text_embeddings = model.encode(texts)['dense_vecs']

# 计算相似度（余弦相似度）
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity([query_embedding], text_embeddings)[0]

print(f"查询文本: {query}\n")
print("相似度排序：")
for i, (text, sim) in enumerate(sorted(zip(texts, similarities), key=lambda x: x[1], reverse=True)):
    print(f"{i+1}. 相似度: {sim:.4f}")
    print(f"   文本: {text[:80]}...")
    print()