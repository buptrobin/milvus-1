# PyMilvus API Reference Guide

## Common Import Structure

```python
from pymilvus import (
    connections,      # Connection management
    utility,         # Utility functions
    Collection,      # Collection class
    FieldSchema,     # Field definition
    CollectionSchema,# Schema definition
    DataType,        # Data types
    Partition,       # Partition management
)
```

## Correct API Usage

### ❌ WRONG - These attributes don't exist:
```python
# These will cause AttributeError:
utility.collection()  # WRONG - no such method
utility.Collection()  # WRONG - Collection is a separate import
```

### ✅ CORRECT - Use these instead:

## 1. Connection Management

```python
# Connect to Milvus
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# Disconnect
connections.disconnect("default")

# List connections
connections.list_connections()
```

## 2. Utility Functions (utility module)

```python
# List all collections
collections = utility.list_collections()

# Check if collection exists
exists = utility.has_collection("collection_name")

# Drop collection
utility.drop_collection("collection_name")

# Get server version
version = utility.get_server_version()

# Flush all collections
utility.flush_all()

# Load collection
utility.load_collection("collection_name")

# Release collection
utility.release_collection("collection_name")

# Get query segment info
info = utility.get_query_segment_info("collection_name")

# Index building progress
progress = utility.index_building_progress("collection_name")

# Wait for index building
utility.wait_for_index_building_complete("collection_name")

# Load balance
utility.load_balance("collection_name")

# Bulk insert
utility.do_bulk_insert(
    collection_name="collection_name",
    files=["file1.json", "file2.json"]
)

# Get bulk insert state
state = utility.get_bulk_insert_state(task_id)

# List bulk insert tasks
tasks = utility.list_bulk_insert_tasks()
```

## 3. Collection Operations

```python
# Create collection with schema
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

# Define fields
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500),
]

# Create schema
schema = CollectionSchema(fields, description="My collection")

# Create collection
collection = Collection(name="my_collection", schema=schema)

# Get existing collection
collection = Collection("existing_collection")

# Collection properties
collection.name           # Get name
collection.schema        # Get schema
collection.description   # Get description
collection.num_entities  # Get entity count
collection.is_empty     # Check if empty
collection.partitions   # Get partitions

# Collection operations
collection.load()       # Load to memory
collection.release()    # Release from memory
collection.insert(data) # Insert data
collection.delete(expr) # Delete by expression
collection.upsert(data) # Upsert data
collection.search(...)  # Search vectors
collection.query(expr)  # Query with expression
collection.flush()      # Flush data
collection.drop()       # Drop collection
collection.compact()    # Compact segments
collection.get_replicas() # Get replica info
```

## 4. Index Management

```python
# Create index
collection.create_index(
    field_name="embedding",
    index_params={
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
)

# Drop index
collection.drop_index()

# Get index info
collection.index()
collection.indexes

# Check if index exists
collection.has_index()
```

## 5. Partition Management

```python
# Create partition
partition = collection.create_partition("partition_name")

# Get partition
partition = collection.partition("partition_name")

# List partitions
partitions = collection.partitions

# Drop partition
collection.drop_partition("partition_name")

# Check if partition exists
collection.has_partition("partition_name")

# Partition operations
partition.load()
partition.release()
partition.insert(data)
partition.delete(expr)
partition.search(...)
partition.query(expr)
```

## 6. Data Types

```python
from pymilvus import DataType

# Available data types
DataType.BOOL
DataType.INT8
DataType.INT16
DataType.INT32
DataType.INT64
DataType.FLOAT
DataType.DOUBLE
DataType.STRING        # Deprecated, use VARCHAR
DataType.VARCHAR
DataType.JSON
DataType.ARRAY
DataType.BINARY_VECTOR
DataType.FLOAT_VECTOR
DataType.FLOAT16_VECTOR
DataType.BFLOAT16_VECTOR
DataType.SPARSE_FLOAT_VECTOR
```

## 7. Search and Query

```python
# Vector search
results = collection.search(
    data=[[0.1, 0.2, ...]],  # Query vectors
    anns_field="embedding",   # Vector field name
    param={
        "metric_type": "L2",
        "params": {"nprobe": 10}
    },
    limit=10,
    expr="id > 0",           # Filter expression
    output_fields=["id", "text"],
    consistency_level="Strong"
)

# Scalar query
results = collection.query(
    expr="id in [1, 2, 3]",
    output_fields=["id", "text"],
    consistency_level="Strong"
)

# Hybrid search
from pymilvus import AnnSearchRequest, RRFRanker

req1 = AnnSearchRequest(
    data=[[0.1, 0.2, ...]],
    anns_field="embedding1",
    param={"metric_type": "L2"},
    limit=10
)

req2 = AnnSearchRequest(
    data=[[0.3, 0.4, ...]],
    anns_field="embedding2",
    param={"metric_type": "COSINE"},
    limit=10
)

results = collection.hybrid_search(
    reqs=[req1, req2],
    rerank=RRFRanker(),
    limit=10
)
```

## 8. Consistency Levels

```python
from pymilvus import ConsistencyLevel

ConsistencyLevel.Strong     # Strong consistency
ConsistencyLevel.Session    # Session consistency
ConsistencyLevel.Bounded    # Bounded staleness
ConsistencyLevel.Eventually # Eventual consistency
ConsistencyLevel.Customized # User-defined timestamp
```

## Common Errors and Solutions

### Error: `module 'pymilvus.orm.utility' has no attribute 'collection'`
**Solution**: `utility` module contains utility functions, not classes. Use `Collection` class directly:
```python
# Wrong
collection = utility.collection("name")

# Correct
from pymilvus import Collection
collection = Collection("name")
```

### Error: `Connection not established`
**Solution**: Always connect first:
```python
connections.connect(host="localhost", port="19530")
```

### Error: `Collection not loaded`
**Solution**: Load collection before search:
```python
collection.load()
```

## Quick Test Script

```python
from pymilvus import connections, utility

# Test connection
try:
    connections.connect(host="localhost", port="19530")
    print("Connected to Milvus")

    # List collections (correct usage)
    collections = utility.list_collections()
    print(f"Collections: {collections}")

except Exception as e:
    print(f"Error: {e}")
```