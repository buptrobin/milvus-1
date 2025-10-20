"""
Natural Language Query Agent Package
"""
from .config import CONFIG, AgentConfig, VolcanoConfig, load_config
from .embedding_manager import EmbeddingManager
from .exceptions import (
    AgentError,
    AgentInitializationError,
    CacheError,
    ConfigurationError,
    EmbeddingModelError,
    MilvusConnectionError,
    QueryProcessingError,
    SearchError,
)
from .llm_extractor import ExtractedInfo, VolcanoLLMExtractor
from .logging_config import LoggerMixin, get_logger, setup_logging
from .milvus_client import MilvusClient
from .nl_query_agent import NaturalLanguageQueryAgent
from .query_processor import QueryIntent, QueryProcessor
from .result_analyzer import AnalysisResult, AnalyzedResult, ResultAnalyzer
from .utils import PerformanceMonitor, Timer, performance_monitor
from .volcano_models import (
    ModelCategory,
    ModelInfo,
    VOLCANO_PUBLIC_MODELS,
    CHAT_MODELS,
    EMBEDDING_MODELS,
    DEFAULT_MODEL,
    PRODUCTION_MODEL,
    LONG_CONTEXT_MODEL,
    VOLCANO_BASE_URL,
    get_model_info,
    recommend_model,
    validate_model_name,
    is_public_model,
    get_model_description,
    list_available_models,
)

__version__ = "1.0.0"

__all__ = [
    "CONFIG",
    "AgentConfig",
    "VolcanoConfig",
    "load_config",
    "NaturalLanguageQueryAgent",
    "MilvusClient",
    "EmbeddingManager",
    "VolcanoLLMExtractor",
    "ExtractedInfo",
    "QueryProcessor",
    "QueryIntent",
    "ResultAnalyzer",
    "AnalysisResult",
    "AnalyzedResult",
    "AgentError",
    "AgentInitializationError",
    "MilvusConnectionError",
    "EmbeddingModelError",
    "QueryProcessingError",
    "SearchError",
    "ConfigurationError",
    "CacheError",
    "setup_logging",
    "get_logger",
    "LoggerMixin",
    "Timer",
    "PerformanceMonitor",
    "performance_monitor",
    # Volcano models
    "ModelCategory",
    "ModelInfo",
    "VOLCANO_PUBLIC_MODELS",
    "CHAT_MODELS",
    "EMBEDDING_MODELS",
    "DEFAULT_MODEL",
    "PRODUCTION_MODEL",
    "LONG_CONTEXT_MODEL",
    "VOLCANO_BASE_URL",
    "get_model_info",
    "recommend_model",
    "validate_model_name",
    "is_public_model",
    "get_model_description",
    "list_available_models",
]
