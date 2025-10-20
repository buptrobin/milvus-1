"""
Custom exceptions for the Natural Language Query Agent
"""


class AgentError(Exception):
    """Base exception class for agent-related errors"""
    pass


class AgentInitializationError(AgentError):
    """Raised when agent initialization fails"""
    pass


class MilvusConnectionError(AgentError):
    """Raised when Milvus connection fails"""
    pass


class EmbeddingModelError(AgentError):
    """Raised when embedding model operations fail"""
    pass


class QueryProcessingError(AgentError):
    """Raised when query processing fails"""
    pass


class SearchError(AgentError):
    """Raised when search operations fail"""
    pass


class ConfigurationError(AgentError):
    """Raised when configuration is invalid"""
    pass


class CacheError(AgentError):
    """Raised when cache operations fail"""
    pass
