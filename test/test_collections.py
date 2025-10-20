#!/usr/bin/env python
"""
测试脚本 - 验证 create_milvus_collection.py 的语法和导入
"""
import sys

def test_imports():
    """测试导入和基本语法"""
    print("测试 create_milvus_collection.py...")
    
    try:
        # 测试导入
        from create_milvus_collection import (
            connect_to_milvus,
            disconnect_from_milvus,
            create_metadata_collection,
            create_event_collection,
            main,
            COLLECTION_NAME,
            EVENT_COLLECTION_NAME,
            VECTOR_DIMENSION
        )
        print("[OK] 所有函数和变量导入成功")
        
        # 显示配置
        print("\n配置信息:")
        print(f"  - Metadata Collection: {COLLECTION_NAME}")
        print(f"  - Event Collection: {EVENT_COLLECTION_NAME}")
        print(f"  - Vector Dimension: {VECTOR_DIMENSION}")
        
        # 验证函数存在
        print("\n可用函数:")
        print("  - connect_to_milvus()")
        print("  - disconnect_from_milvus()")
        print("  - create_metadata_collection()")
        print("  - create_event_collection()")
        print("  - main()")
        
        print("\n[成功] 文件结构正确，可以使用以下命令运行:")
        print("  uv run python create_milvus_collection.py          # 交互式菜单")
        print("  uv run python create_milvus_collection.py metadata # 创建 metadata collection")
        print("  uv run python create_milvus_collection.py event    # 创建 event collection")
        print("  uv run python create_milvus_collection.py both     # 创建两个 collection")
        
        return True
        
    except ImportError as e:
        print(f"[错误] 导入失败: {e}")
        return False
    except SyntaxError as e:
        print(f"[错误] 语法错误: {e}")
        return False

def show_schema_comparison():
    """显示两个 Collection 的 Schema 对比"""
    print("\n" + "=" * 70)
    print("Schema 对比")
    print("=" * 70)
    
    print("\nMetadata Collection (Pampers_metadata) Schema:")
    print("-" * 40)
    metadata_fields = [
        ("concept_id", "VARCHAR(1024)", "True", "概念属性的全局唯一ID"),
        ("source_type", "VARCHAR(64)", "False", "'PROFILE' 或 'EVENT'"),
        ("source_name", "VARCHAR(256)", "False", "来源实体名称"),
        ("field_name", "VARCHAR(256)", "False", "属性字段本身的名称"),
        ("raw_metadata", "JSON", "False", "原始元数据"),
        ("concept_embedding", "FLOAT_VECTOR(768)", "False", "语义向量")
    ]
    
    for field in metadata_fields:
        print(f"  {field[0]:20} {field[1]:20} 主键:{field[2]:5} {field[3]}")
    
    print("\nEvent Collection (Event_metadata) Schema:")
    print("-" * 40)
    event_fields = [
        ("concept_id", "VARCHAR(1024)", "True", "属性的唯一ID"),
        ("source_type", "VARCHAR(64)", "False", "固定为 'EVENT'"),
        ("source_name", "VARCHAR(256)", "False", "事件名"),
        ("field_name", "VARCHAR(256)", "False", "属性名"),
        ("raw_metadata", "JSON", "False", "原始元数据"),
        ("concept_embedding", "FLOAT_VECTOR(768)", "False", "核心向量")
    ]
    
    for field in event_fields:
        print(f"  {field[0]:20} {field[1]:20} 主键:{field[2]:5} {field[3]}")
    
    print("\n主要区别:")
    print("  1. source_type: Metadata 可以是 'PROFILE' 或 'EVENT'，Event 固定为 'EVENT'")
    print("  2. 描述措辞: Event 版本更强调事件相关的术语")
    print("  3. Collection 名称: 'Pampers_metadata' vs 'Event_metadata'")

if __name__ == "__main__":
    print("=" * 70)
    print("Collection 创建脚本测试")
    print("=" * 70)
    
    # 测试导入
    if test_imports():
        # 显示 Schema 对比
        show_schema_comparison()
        
        print("\n" + "=" * 70)
        print("测试完成！")
    else:
        print("\n请检查 create_milvus_collection.py 文件")
        sys.exit(1)