"""
Configuration management for Natural Language Query Agent
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class MilvusConfig:
    """Milvus database configuration"""
    host: str = "172.28.9.45"
    port: str = "19530"
    database: str = "default"
    alias: str = "default"
    timeout: float = 30.0


@dataclass
class CollectionConfig:
    """Collection configuration"""
    metadata_collection: str = "Pampers_metadata"
    vector_dimension: int = 1024


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = "BAAI/bge-m3"
    use_fp16: bool = True
    batch_size: int = 128
    max_length: int = 512


@dataclass
class SearchConfig:
    """Search configuration"""
    similarity_threshold: float = 0.65
    score_gap_threshold: float = 0.08
    max_results: int = 10
    profile_search_limit: int = 5
    event_search_limit: int = 5
    event_attr_search_limit: int = 10


@dataclass
class VolcanoConfig:
    """Volcano Engine LLM configuration"""
    api_key: str = ""
    model: str = ""  # Public model name or endpoint ID
    endpoint_id: str = ""  # Deprecated, use 'model' instead
    use_public_model: bool = True  # True for public models, False for custom endpoints
    base_url: str = ""  # API base URL (auto-set for public models)
    system_prompt: str = ""
    extraction_prompt_template: str = ""
    prompt_file_path: str = ""  # Path to prompt template file
    max_tokens: int = 1024
    temperature: float = 0.1
    timeout: int = 30
    enabled: bool = False


@dataclass
class AgentConfig:
    """Main agent configuration"""
    milvus: MilvusConfig
    collections: CollectionConfig
    embedding: EmbeddingConfig
    search: SearchConfig
    volcano: VolcanoConfig
    log_level: str = "INFO"
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1 hour


def load_config() -> AgentConfig:
    """Load configuration from environment variables and defaults"""

    # Milvus configuration
    milvus_config = MilvusConfig(
        host=os.getenv("MILVUS_HOST", "172.28.9.45"),
        port=os.getenv("MILVUS_PORT", "19530"),
        database=os.getenv("MILVUS_DATABASE", "default"),
        alias=os.getenv("MILVUS_ALIAS", "default"),
        timeout=float(os.getenv("MILVUS_TIMEOUT", "30.0"))
    )

    # Collection configuration
    collection_config = CollectionConfig(
        metadata_collection=os.getenv("METADATA_COLLECTION", "Pampers_metadata"),
        vector_dimension=int(os.getenv("VECTOR_DIMENSION", "1024"))
    )

    # Embedding configuration
    embedding_config = EmbeddingConfig(
        model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        use_fp16=os.getenv("USE_FP16", "true").lower() == "true",
        batch_size=int(os.getenv("BATCH_SIZE", "128")),
        max_length=int(os.getenv("MAX_LENGTH", "512"))
    )

    # Search configuration
    search_config = SearchConfig(
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.65")),
        score_gap_threshold=float(os.getenv("SCORE_GAP_THRESHOLD", "0.08")),
        max_results=int(os.getenv("MAX_RESULTS", "10")),
        profile_search_limit=int(os.getenv("PROFILE_SEARCH_LIMIT", "5")),
        event_search_limit=int(os.getenv("EVENT_SEARCH_LIMIT", "5")),
        event_attr_search_limit=int(os.getenv("EVENT_ATTR_SEARCH_LIMIT", "10"))
    )

    # Volcano Engine configuration
    # Support both new 'model' parameter and deprecated 'endpoint_id'
    model_name = os.getenv("VOLCANO_MODEL", os.getenv("VOLCANO_MODEL_NAME", ""))
    endpoint_id = os.getenv("VOLCANO_ENDPOINT_ID", "")

    # Use model if provided, otherwise fall back to endpoint_id
    model = model_name if model_name else endpoint_id

    volcano_config = VolcanoConfig(
        api_key=os.getenv("VOLCANO_API_KEY", "7b9a3304-daa2-43fe-af5c-bbade3432252"),
        model=model,
        endpoint_id=endpoint_id,  # Keep for backward compatibility
        use_public_model=os.getenv("VOLCANO_USE_PUBLIC_MODEL", "true").lower() == "true",
        base_url=os.getenv("VOLCANO_BASE_URL", ""),
        system_prompt=os.getenv("VOLCANO_SYSTEM_PROMPT", ""),
        extraction_prompt_template=os.getenv("VOLCANO_EXTRACTION_PROMPT", ""),
        prompt_file_path=os.getenv("VOLCANO_PROMPT_FILE", "prompt.txt"),  # Default to prompt.txt
        max_tokens=int(os.getenv("VOLCANO_MAX_TOKENS", "1024")),
        temperature=float(os.getenv("VOLCANO_TEMPERATURE", "0.1")),
        timeout=int(os.getenv("VOLCANO_TIMEOUT", "30")),
        enabled=os.getenv("VOLCANO_ENABLED", "false").lower() == "true"
    )

    return AgentConfig(
        milvus=milvus_config,
        collections=collection_config,
        embedding=embedding_config,
        search=search_config,
        volcano=volcano_config,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        enable_cache=os.getenv("ENABLE_CACHE", "true").lower() == "true",
        cache_ttl=int(os.getenv("CACHE_TTL", "3600"))
    )


# Global configuration instance
CONFIG = load_config()
