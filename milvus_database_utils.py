#!/usr/bin/env python
"""
Milvus 数据库管理工具
用于创建、列出和管理 Milvus 数据库
"""
from pymilvus import connections, db

# 配置信息
MILVUS_HOST = "172.28.9.45"
MILVUS_PORT = "19530"

def connect_to_milvus():
    """连接到 Milvus 服务器"""
    try:
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        print(f"[+] 成功连接到 Milvus 服务器: {MILVUS_HOST}:{MILVUS_PORT}")
        return True
    except Exception as e:
        print(f"[-] 连接失败: {e}")
        return False

def list_databases():
    """列出所有数据库"""
    try:
        databases = db.list_database()
        print("\n现有数据库列表:")
        for idx, database in enumerate(databases, 1):
            print(f"  {idx}. {database}")
        return databases
    except Exception as e:
        print(f"[-] 列出数据库失败: {e}")
        return []

def create_database(db_name):
    """创建新数据库"""
    try:
        # 检查数据库是否已存在
        databases = db.list_database()
        if db_name in databases:
            print(f"[!] 数据库 '{db_name}' 已存在")
            return False
        
        # 创建数据库
        db.create_database(db_name)
        print(f"[+] 成功创建数据库: {db_name}")
        return True
    except Exception as e:
        print(f"[-] 创建数据库失败: {e}")
        return False

def drop_database(db_name):
    """删除数据库"""
    try:
        # 不允许删除 default 数据库
        if db_name == "default":
            print("[!] 不能删除 default 数据库")
            return False
        
        # 检查数据库是否存在
        databases = db.list_database()
        if db_name not in databases:
            print(f"[!] 数据库 '{db_name}' 不存在")
            return False
        
        # 删除数据库
        db.drop_database(db_name)
        print(f"[+] 成功删除数据库: {db_name}")
        return True
    except Exception as e:
        print(f"[-] 删除数据库失败: {e}")
        return False

def use_database(db_name):
    """切换到指定数据库"""
    try:
        db.using_database(db_name)
        print(f"[+] 已切换到数据库: {db_name}")
        return True
    except Exception as e:
        print(f"[-] 切换数据库失败: {e}")
        return False

def get_database_info():
    """获取当前数据库信息"""
    try:
        # 获取当前使用的数据库
        # 注意：pymilvus 可能没有直接获取当前数据库的方法
        # 这里我们列出所有数据库作为参考
        databases = db.list_database()
        print("\n数据库信息:")
        print(f"  可用数据库: {databases}")
        print(f"  默认数据库: default")
        print("\n提示: Milvus 2.3+ 支持多数据库")
        print("  - 每个数据库可以包含多个 Collection")
        print("  - Collection 名称在数据库内必须唯一")
        print("  - 不同数据库可以有同名的 Collection")
        return True
    except Exception as e:
        print(f"[-] 获取数据库信息失败: {e}")
        return False

def main():
    """主函数 - 交互式数据库管理"""
    print("="*60)
    print("Milvus 数据库管理工具")
    print("="*60)
    
    # 连接到 Milvus
    if not connect_to_milvus():
        print("\n无法连接到 Milvus 服务器")
        return
    
    while True:
        print("\n" + "-"*40)
        print("选择操作:")
        print("  1. 列出所有数据库")
        print("  2. 创建新数据库")
        print("  3. 删除数据库")
        print("  4. 切换数据库")
        print("  5. 数据库信息")
        print("  0. 退出")
        
        choice = input("\n请输入选项 (0-5): ").strip()
        
        if choice == "0":
            print("\n退出程序")
            break
        elif choice == "1":
            list_databases()
        elif choice == "2":
            db_name = input("输入新数据库名称: ").strip()
            if db_name:
                create_database(db_name)
        elif choice == "3":
            db_name = input("输入要删除的数据库名称: ").strip()
            if db_name:
                confirm = input(f"确定要删除数据库 '{db_name}' 吗？(y/n): ").strip().lower()
                if confirm == 'y':
                    drop_database(db_name)
        elif choice == "4":
            db_name = input("输入要切换到的数据库名称: ").strip()
            if db_name:
                use_database(db_name)
        elif choice == "5":
            get_database_info()
        else:
            print("[!] 无效选项")
    
    # 断开连接
    connections.disconnect("default")
    print("\n[*] 已断开与 Milvus 的连接")

if __name__ == "__main__":
    # 也可以直接调用特定功能
    import sys
    if len(sys.argv) > 1:
        if connect_to_milvus():
            if sys.argv[1] == "list":
                list_databases()
            elif sys.argv[1] == "create" and len(sys.argv) > 2:
                create_database(sys.argv[2])
            elif sys.argv[1] == "drop" and len(sys.argv) > 2:
                drop_database(sys.argv[2])
            elif sys.argv[1] == "use" and len(sys.argv) > 2:
                use_database(sys.argv[2])
            else:
                print("用法:")
                print("  python milvus_database_utils.py          # 交互式模式")
                print("  python milvus_database_utils.py list     # 列出数据库")
                print("  python milvus_database_utils.py create <name>  # 创建数据库")
                print("  python milvus_database_utils.py drop <name>    # 删除数据库")
                print("  python milvus_database_utils.py use <name>     # 切换数据库")
            connections.disconnect("default")
    else:
        main()