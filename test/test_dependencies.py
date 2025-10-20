#!/usr/bin/env python
"""
测试依赖包是否正确安装
"""
import sys

def test_dependencies():
    """测试所有必需的依赖包"""
    print("=" * 60)
    print("检查依赖包安装情况")
    print("=" * 60)
    
    dependencies = [
        ("pandas", "数据处理"),
        ("pymilvus", "Milvus 客户端"),
        ("sentence_transformers", "文本嵌入模型"),
        ("tqdm", "进度条显示"),
        ("numpy", "数值计算"),
        ("torch", "深度学习框架"),
    ]
    
    success_count = 0
    failed = []
    
    for module_name, description in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "版本未知")
            print(f"[OK] {module_name:25} {version:15} {description}")
            success_count += 1
        except ImportError as e:
            print(f"[X] {module_name:25} 未安装           {description}")
            failed.append(module_name)
    
    print("=" * 60)
    print(f"成功: {success_count}/{len(dependencies)}")
    
    if failed:
        print(f"失败: {', '.join(failed)}")
        print("\n安装缺失的包:")
        print("  使用 uv:")
        print("    uv sync")
        print("  或使用 pip:")
        print(f"    pip install {' '.join(failed)}")
        return False
    else:
        print("\n[SUCCESS] 所有依赖包已正确安装！")
        return True

def test_sentence_transformers():
    """测试 sentence-transformers 功能"""
    print("\n" + "=" * 60)
    print("测试 Sentence Transformers")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        print("正在加载模型（首次加载可能需要下载）...")
        
        # 使用一个小型模型进行测试
        model_name = 'paraphrase-MiniLM-L6-v2'
        model = SentenceTransformer(model_name)
        
        # 测试编码
        sentences = ['这是一个测试句子', 'This is a test sentence']
        embeddings = model.encode(sentences)
        
        print(f"[OK] 成功加载模型: {model_name}")
        print(f"[OK] 生成嵌入向量维度: {embeddings.shape}")
        print(f"  - 句子数量: {embeddings.shape[0]}")
        print(f"  - 向量维度: {embeddings.shape[1]}")
        return True
        
    except Exception as e:
        print(f"[FAIL] Sentence Transformers 测试失败: {e}")
        return False

def test_pymilvus_import():
    """测试 pymilvus 导入"""
    print("\n" + "=" * 60)
    print("测试 PyMilvus 导入")
    print("=" * 60)
    
    try:
        from pymilvus import (
            connections,
            utility,
            Collection,
            FieldSchema,
            CollectionSchema,
            DataType
        )
        print("[OK] PyMilvus 核心模块导入成功")
        print("  可用的主要类和函数:")
        print("  - connections: 连接管理")
        print("  - utility: 工具函数")
        print("  - Collection: 集合操作")
        print("  - FieldSchema: 字段定义")
        print("  - CollectionSchema: 模式定义")
        print("  - DataType: 数据类型")
        return True
        
    except ImportError as e:
        print(f"[FAIL] PyMilvus 导入失败: {e}")
        return False

def main():
    """主函数"""
    print("\n[*] 开始检查项目依赖...\n")
    
    # 测试基础依赖
    deps_ok = test_dependencies()
    
    # 测试 pymilvus
    pymilvus_ok = test_pymilvus_import()
    
    # 可选：测试 sentence-transformers（会下载模型）
    if deps_ok:
        print("\n是否要测试 Sentence Transformers（可能需要下载模型）？")
        choice = input("输入 y 测试，其他键跳过: ").strip().lower()
        if choice == 'y':
            test_sentence_transformers()
    
    print("\n" + "=" * 60)
    if deps_ok and pymilvus_ok:
        print("[SUCCESS] 环境检查完成！所有必需的依赖都已就绪。")
        print("\n您现在可以运行:")
        print("  uv run python create_milvus_collection.py")
    else:
        print("[ERROR] 环境检查失败，请安装缺失的依赖。")
        sys.exit(1)

if __name__ == "__main__":
    main()