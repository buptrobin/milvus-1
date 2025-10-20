"""
Volcano Engine Public Models Constants and Utilities
火山引擎公共模型常量和工具函数
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ModelCategory(Enum):
    """Model categories"""
    PROFESSIONAL = "professional"
    LITE = "lite"
    VISION = "vision"
    THINKING = "thinking"
    EMBEDDING = "embedding"


@dataclass
class ModelInfo:
    """Model information"""
    name: str
    display_name: str
    category: ModelCategory
    context_window: int
    description: str
    cost_level: str  # low, medium, high
    recommended_for: list[str]


# Volcano Engine Public Models
VOLCANO_PUBLIC_MODELS = {
    # Doubao Professional Models (专业版模型)
    "doubao-pro-32k": ModelInfo(
        name="doubao-pro-32k",
        display_name="豆包 Pro 32K",
        category=ModelCategory.PROFESSIONAL,
        context_window=32768,
        description="Professional model with 32K context window, balanced performance",
        cost_level="medium",
        recommended_for=["general_qa", "document_analysis", "information_extraction"]
    ),
    "doubao-pro-128k": ModelInfo(
        name="doubao-pro-128k",
        display_name="豆包 Pro 128K",
        category=ModelCategory.PROFESSIONAL,
        context_window=131072,
        description="Professional model with 128K context window, for long documents",
        cost_level="high",
        recommended_for=["long_document", "multi_file_analysis", "complex_reasoning"]
    ),
    "doubao-pro-256k": ModelInfo(
        name="doubao-pro-256k",
        display_name="豆包 Pro 256K",
        category=ModelCategory.PROFESSIONAL,
        context_window=262144,
        description="Professional model with 256K context window, for very long documents",
        cost_level="high",
        recommended_for=["very_long_document", "book_analysis", "code_repository"]
    ),

    # Doubao Lite Models (轻量版模型)
    "doubao-lite-32k": ModelInfo(
        name="doubao-lite-32k",
        display_name="豆包 Lite 32K",
        category=ModelCategory.LITE,
        context_window=32768,
        description="Lightweight model with 32K context, faster and cheaper",
        cost_level="low",
        recommended_for=["simple_qa", "quick_extraction", "testing"]
    ),
    "doubao-lite-128k": ModelInfo(
        name="doubao-lite-128k",
        display_name="豆包 Lite 128K",
        category=ModelCategory.LITE,
        context_window=131072,
        description="Lightweight model with 128K context",
        cost_level="low",
        recommended_for=["long_text_simple_task", "batch_processing"]
    ),

    # Doubao 1.5 Models (新一代模型)
    "doubao-1.5-pro-32k": ModelInfo(
        name="doubao-1.5-pro-32k",
        display_name="豆包 1.5 Pro 32K",
        category=ModelCategory.PROFESSIONAL,
        context_window=32768,
        description="Latest Doubao 1.5 professional model with enhanced capabilities",
        cost_level="medium",
        recommended_for=["advanced_reasoning", "complex_extraction", "high_accuracy"]
    ),
    "doubao-1.5-lite-32k": ModelInfo(
        name="doubao-1.5-lite-32k",
        display_name="豆包 1.5 Lite 32K",
        category=ModelCategory.LITE,
        context_window=32768,
        description="Latest Doubao 1.5 lightweight model",
        cost_level="low",
        recommended_for=["quick_tasks", "development", "prototyping"]
    ),

    # Embedding Models (嵌入模型)
    "doubao-embedding": ModelInfo(
        name="doubao-embedding",
        display_name="豆包嵌入模型",
        category=ModelCategory.EMBEDDING,
        context_window=2048,
        description="Text embedding model with 2560 dimensions",
        cost_level="low",
        recommended_for=["text_embedding", "similarity_search", "rag"]
    ),
    "doubao-embedding-large": ModelInfo(
        name="doubao-embedding-large",
        display_name="豆包大型嵌入模型",
        category=ModelCategory.EMBEDDING,
        context_window=2048,
        description="Large text embedding model with 2048 dimensions",
        cost_level="low",
        recommended_for=["high_quality_embedding", "semantic_search"]
    ),
}

# Default model recommendations
DEFAULT_MODEL = "doubao-lite-32k"  # Default for testing and development
PRODUCTION_MODEL = "doubao-pro-32k"  # Recommended for production
LONG_CONTEXT_MODEL = "doubao-pro-128k"  # For long documents

# Model name lists for convenience
CHAT_MODELS = [
    "doubao-pro-32k",
    "doubao-pro-128k",
    "doubao-pro-256k",
    "doubao-lite-32k",
    "doubao-lite-128k",
    "doubao-1.5-pro-32k",
    "doubao-1.5-lite-32k",
]

EMBEDDING_MODELS = [
    "doubao-embedding",
    "doubao-embedding-large",
    "doubao-embedding-text-240715",
    "doubao-embedding-large-text-240915",
]

# API Configuration
VOLCANO_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


def get_model_info(model_name: str) -> Optional[ModelInfo]:
    """
    Get model information by name

    Args:
        model_name: Model name

    Returns:
        ModelInfo object or None if not found
    """
    return VOLCANO_PUBLIC_MODELS.get(model_name)


def recommend_model(
    task_type: str = "general",
    context_length: int = 0,
    cost_sensitive: bool = True
) -> str:
    """
    Recommend a model based on task requirements

    Args:
        task_type: Type of task (general, extraction, embedding, etc.)
        context_length: Required context length in tokens
        cost_sensitive: Whether to prioritize lower cost

    Returns:
        Recommended model name
    """
    # For embedding tasks
    if task_type == "embedding":
        return "doubao-embedding"

    # Based on context length
    if context_length > 131072:
        return "doubao-pro-256k"
    elif context_length > 32768:
        if cost_sensitive:
            return "doubao-lite-128k"
        else:
            return "doubao-pro-128k"

    # Based on task type and cost sensitivity
    if task_type in ["extraction", "analysis", "reasoning"]:
        if cost_sensitive:
            return "doubao-lite-32k"
        else:
            return "doubao-1.5-pro-32k"

    # Default recommendation
    return DEFAULT_MODEL if cost_sensitive else PRODUCTION_MODEL


def validate_model_name(model_name: str) -> bool:
    """
    Validate if a model name is a valid public model

    Args:
        model_name: Model name to validate

    Returns:
        True if valid, False otherwise
    """
    return model_name in VOLCANO_PUBLIC_MODELS or model_name in EMBEDDING_MODELS


def is_public_model(model_identifier: str) -> bool:
    """
    Check if an identifier is a public model name or an endpoint ID

    Args:
        model_identifier: Model name or endpoint ID

    Returns:
        True if it's a public model name, False if it's an endpoint ID
    """
    # Endpoint IDs typically start with "ep-"
    if model_identifier.startswith("ep-"):
        return False

    # Check if it's a known public model
    return validate_model_name(model_identifier)


def get_model_description(model_name: str) -> str:
    """
    Get a human-readable description of a model

    Args:
        model_name: Model name

    Returns:
        Model description string
    """
    info = get_model_info(model_name)
    if info:
        return f"{info.display_name}: {info.description} (Context: {info.context_window:,} tokens, Cost: {info.cost_level})"
    return f"Unknown model: {model_name}"


def list_available_models(category: Optional[ModelCategory] = None) -> list[str]:
    """
    List available models, optionally filtered by category

    Args:
        category: Optional category filter

    Returns:
        List of model names
    """
    if category is None:
        return list(VOLCANO_PUBLIC_MODELS.keys())

    return [
        name for name, info in VOLCANO_PUBLIC_MODELS.items()
        if info.category == category
    ]