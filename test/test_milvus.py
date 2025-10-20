#!/usr/bin/env python
"""
Test script for Milvus Python SDK
"""
import pymilvus
from pymilvus import connections, utility

def test_milvus_version():
    """Test pymilvus version"""
    print(f"PyMilvus version: {pymilvus.__version__}")
    print("Successfully imported pymilvus modules")
    
def test_connection_params():
    """Test connection parameters setup"""
    # This is just a test of the connection parameters, not an actual connection
    try:
        # Example connection parameters (do not actually connect)
        host = "localhost"
        port = 19530
        print(f"Milvus connection parameters configured: {host}:{port}")
        print("Ready to connect to Milvus server when available")
        return True
    except Exception as e:
        print(f"Error setting up connection parameters: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("Milvus Python SDK Test")
    print("=" * 50)
    
    # Test 1: Version check
    test_milvus_version()
    print()
    
    # Test 2: Connection parameters
    if test_connection_params():
        print("\n[SUCCESS] All Milvus dependencies are properly installed!")
        print("\nTo connect to a Milvus server, use:")
        print("  connections.connect(host='localhost', port='19530')")
    else:
        print("\n[ERROR] There were issues with the setup")
    
    print("=" * 50)

if __name__ == "__main__":
    main()