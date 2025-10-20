"""
Milvus client wrapper with connection management
"""
import logging
from typing import Any

from pymilvus import Collection, connections, utility

from .config import CollectionConfig, MilvusConfig

logger = logging.getLogger(__name__)


class MilvusClient:
    """Milvus client wrapper with singleton pattern and connection management"""

    _instance = None
    _initialized = False

    def __new__(cls, milvus_config: MilvusConfig, collection_config: CollectionConfig):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, milvus_config: MilvusConfig, collection_config: CollectionConfig):
        if self._initialized:
            return

        self.milvus_config = milvus_config
        self.collection_config = collection_config
        self._collections = {}
        self._connected = False
        self._initialized = True

        self.connect()

    def connect(self) -> bool:
        """Connect to Milvus server"""
        try:
            if self._connected:
                return True

            connections.connect(
                alias=self.milvus_config.alias,
                host=self.milvus_config.host,
                port=self.milvus_config.port,
                timeout=self.milvus_config.timeout
            )

            self._connected = True
            logger.info(f"Connected to Milvus: {self.milvus_config.host}:{self.milvus_config.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Milvus server"""
        try:
            if self._connected:
                connections.disconnect(self.milvus_config.alias)
                self._connected = False
                self._collections.clear()
                logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Error disconnecting from Milvus: {e}")

    def get_collection(self, collection_name: str) -> Collection | None:
        """Get collection instance with caching"""
        if not self._connected:
            if not self.connect():
                return None

        if collection_name in self._collections:
            return self._collections[collection_name]

        try:
            if not utility.has_collection(collection_name):
                logger.error(f"Collection '{collection_name}' does not exist")
                return None

            collection = Collection(collection_name)
            self._collections[collection_name] = collection

            # Load collection to memory if not already loaded
            if not collection.has_index():
                logger.warning(f"Collection '{collection_name}' has no index")
            else:
                collection.load()
                logger.info(f"Collection '{collection_name}' loaded to memory")

            return collection

        except Exception as e:
            logger.error(f"Failed to get collection '{collection_name}': {e}")
            return None

    def get_metadata_collection(self) -> Collection | None:
        """Get metadata collection"""
        return self.get_collection(self.collection_config.metadata_collection)


    def search_collection(
        self,
        collection_name: str,
        query_vector: list[float],
        search_params: dict[str, Any],
        output_fields: list[str],
        limit: int = 10,
        expr: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Search in a collection and return formatted results

        Args:
            collection_name: Name of the collection to search
            query_vector: Query vector for similarity search
            search_params: Search parameters for Milvus
            output_fields: Fields to return in results
            limit: Maximum number of results
            expr: Filter expression

        Returns:
            List of search results with scores and metadata
        """
        collection = self.get_collection(collection_name)
        if not collection:
            return []

        try:
            # Log query parameters
            logger.debug(f"[Milvus] Searching collection: {collection_name}")
            logger.debug(f"[Milvus] Query vector (dim={len(query_vector)}): [{query_vector[0]:.4f}, {query_vector[1]:.4f}, ..., {query_vector[-1]:.4f}]")
            logger.debug(f"[Milvus] Search params: {search_params}")
            logger.debug(f"[Milvus] Output fields: {output_fields}")
            logger.debug(f"[Milvus] Limit: {limit}")
            logger.debug(f"[Milvus] Filter expression: {expr}")

            results = collection.search(
                data=[query_vector],
                anns_field="concept_embedding",
                param=search_params,
                limit=limit,
                output_fields=output_fields,
                expr=expr
            )

            formatted_results = []
            if results and len(results) > 0:
                hits = results[0]
                logger.debug(f"[Milvus] Found {len(hits)} results")

                for idx, hit in enumerate(hits):
                    result = {
                        'id': hit.id,
                        'distance': hit.distance,
                        'score': hit.distance  # For COSINE, distance is the similarity score
                    }

                    # Add output fields
                    entity = hit.entity
                    for field in output_fields:
                        result[field] = entity.get(field)

                    formatted_results.append(result)

                    # Log each result
                    key_fields = {
                        'id': hit.id,
                        'score': round(hit.distance, 4),
                        'source_name': entity.get('source_name', 'N/A'),
                        'idname': entity.get('idname', 'N/A')
                    }
                    logger.debug(f"[Milvus] Result #{idx+1}: {key_fields}")
            else:
                logger.debug(f"[Milvus] No results found")

            return formatted_results

        except Exception as e:
            logger.error(f"[Milvus] Search failed in collection '{collection_name}': {e}")
            return []

    def search_profile_attributes(
        self,
        query_vector: list[float],
        limit: int = 5
    ) -> list[dict[str, Any]]:
        """Search for profile attributes (source_type = 'PROFILE')"""
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 128}
        }

        output_fields = [
            "source_type", "source_name", "idname",
            "raw_metadata"
        ]

        # Filter for PROFILE type
        expr = "source_type == 'PROFILE_ATTRIBUTE'"

        return self.search_collection(
            collection_name=self.collection_config.metadata_collection,
            query_vector=query_vector,
            search_params=search_params,
            output_fields=output_fields,
            limit=limit,
            expr=expr
        )



    def list_collections(self) -> list[str]:
        """List all available collections"""
        if not self._connected:
            if not self.connect():
                return []

        try:
            return utility.list_collections()
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def collection_info(self, collection_name: str) -> dict[str, Any]:
        """Get collection information"""
        collection = self.get_collection(collection_name)
        if not collection:
            return {}

        try:
            info = {
                "name": collection_name,
                "description": collection.description,
                "num_entities": collection.num_entities,
                "has_index": collection.has_index(),
                "is_loaded": utility.load_state(collection_name).state.name == "Loaded"
            }
            return info
        except Exception as e:
            logger.error(f"Failed to get collection info for '{collection_name}': {e}")
            return {}

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, '_connected') and self._connected:
            self.disconnect()
