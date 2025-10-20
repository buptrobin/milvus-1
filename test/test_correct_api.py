#!/usr/bin/env python
"""
Test script to verify correct pymilvus API usage
This can run without a Milvus server to check imports
"""

def test_imports():
    """Test that all imports work correctly"""
    print("Testing pymilvus imports...")
    
    try:
        # Test correct imports
        from pymilvus import (
            connections,
            utility,
            Collection,
            FieldSchema,
            CollectionSchema,
            DataType,
            Partition,
        )
        print("[OK] All imports successful")
        
        # Show what's available in utility module
        print("\nAvailable utility functions:")
        utility_methods = [m for m in dir(utility) if not m.startswith('_')]
        for method in utility_methods[:10]:  # Show first 10
            print(f"  - utility.{method}")
        print("  ... and more")
        
        return True
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False

def test_wrong_usage():
    """Demonstrate what NOT to do"""
    print("\n" + "="*50)
    print("Common mistakes to avoid:")
    print("="*50)
    
    from pymilvus import utility
    
    # This will fail - demonstrate the error
    print("\n1. Trying to call utility.collection():")
    try:
        utility.collection("test")
    except AttributeError as e:
        print(f"   [Expected Error] {e}")
        print("   [Solution] Use: from pymilvus import Collection; Collection('name')")
    
    print("\n2. Trying to call utility.Collection():")
    try:
        utility.Collection("test")
    except AttributeError as e:
        print(f"   [Expected Error] {e}")
        print("   [Solution] Import Collection separately")

def test_correct_usage():
    """Demonstrate correct usage patterns"""
    print("\n" + "="*50)
    print("Correct usage patterns:")
    print("="*50)
    
    from pymilvus import Collection, utility, connections
    
    print("\n1. Correct way to work with collections:")
    print("   from pymilvus import Collection")
    print("   collection = Collection('name')  # Get existing")
    print("   collection = Collection(name='new', schema=schema)  # Create new")
    
    print("\n2. Correct way to use utility functions:")
    print("   utility.list_collections()  # List all collections")
    print("   utility.has_collection('name')  # Check existence")
    print("   utility.drop_collection('name')  # Drop collection")
    
    print("\n3. Correct way to manage connections:")
    print("   connections.connect(host='localhost', port='19530')")
    print("   connections.disconnect('default')")
    print("   connections.list_connections()")

def main():
    """Main test function"""
    print("="*60)
    print("PyMilvus API Verification Test")
    print("="*60)
    
    # Test imports
    if test_imports():
        # Show what NOT to do
        test_wrong_usage()
        
        # Show correct usage
        test_correct_usage()
        
        print("\n" + "="*60)
        print("Test completed successfully!")
        print("\nRefer to pymilvus_api_reference.md for complete API guide")
        print("Run milvus_example.py for a full working example")
    else:
        print("\nPlease ensure pymilvus is installed:")
        print("  uv sync")
        print("  or")
        print("  pip install pymilvus")
    
    print("="*60)

if __name__ == "__main__":
    main()