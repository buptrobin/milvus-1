"""测试BGE-M3模型是否正常工作"""
# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
from FlagEmbedding import BGEM3FlagModel
import numpy as np

def test_bge_m3():
    """测试BGE-M3模型的基本功能"""
    print("开始测试BGE-M3模型...")
    
    # 1. 加载模型
    print("\n[1] 加载BGE-M3模型...")
    model_name = 'BAAI/bge-m3'
    use_fp16 = torch.cuda.is_available()
    print(f"    使用FP16: {use_fp16}")
    
    try:
        model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        print("    ✓ 模型加载成功")
    except Exception as e:
        print(f"    ✗ 模型加载失败: {e}")
        return False
    
    # 2. 测试单个文本编码
    print("\n[2] 测试单个文本编码...")
    test_text = "这是一个测试文本"
    try:
        result = model.encode([test_text])
        dense_vec = result['dense_vecs'][0]
        print(f"    输入文本: {test_text}")
        print(f"    向量维度: {len(dense_vec)}")
        print(f"    向量类型: {type(dense_vec)}")
        print(f"    前5个值: {dense_vec[:5]}")
        print("    ✓ 单文本编码成功")
    except Exception as e:
        print(f"    ✗ 单文本编码失败: {e}")
        return False
    
    # 3. 测试批量文本编码
    print("\n[3] 测试批量文本编码...")
    test_texts = [
        "用户在哪个国家",
        "发货到中国",
        "包裹从哪里发货",
        "country"
    ]
    try:
        results = model.encode(test_texts)
        dense_vecs = results['dense_vecs']
        print(f"    输入文本数量: {len(test_texts)}")
        print(f"    输出向量数量: {len(dense_vecs)}")
        print(f"    每个向量维度: {len(dense_vecs[0])}")
        print("    ✓ 批量编码成功")
    except Exception as e:
        print(f"    ✗ 批量编码失败: {e}")
        return False
    
    # 4. 测试相似度计算
    print("\n[4] 测试相似度计算...")
    try:
        # 计算余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        
        query = "国家"
        query_vec = model.encode([query])['dense_vecs']
        
        similarities = cosine_similarity(query_vec, dense_vecs)[0]
        
        print(f"    查询: '{query}'")
        print("    相似度结果:")
        for text, sim in zip(test_texts, similarities):
            print(f"      - '{text}': {sim:.4f}")
        print("    ✓ 相似度计算成功")
    except Exception as e:
        print(f"    ✗ 相似度计算失败: {e}")
        return False
    
    # 5. 验证向量维度
    print("\n[5] 验证向量维度...")
    expected_dim = 1024
    actual_dim = len(dense_vec)
    if actual_dim == expected_dim:
        print(f"    ✓ 向量维度正确: {actual_dim}")
    else:
        print(f"    ✗ 向量维度错误: 期望 {expected_dim}, 实际 {actual_dim}")
        return False
    
    print("\n" + "="*50)
    print("✅ BGE-M3模型测试全部通过!")
    print("="*50)
    return True

if __name__ == "__main__":
    import sys
    success = test_bge_m3()
    sys.exit(0 if success else 1)