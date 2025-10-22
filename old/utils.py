"""
Utility functions for the Natural Language Query Agent
"""
import hashlib
import json
import time
from functools import wraps
from typing import Any

from .exceptions import AgentError
from .logging_config import get_logger

logger = get_logger(__name__)


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry function calls on failure

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each failure
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        break

                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )

                    if attempt < max_retries:
                        time.sleep(current_delay)
                        current_delay *= backoff

            logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
            raise last_exception

        return wrapper
    return decorator


def validate_config(config: dict[str, Any], required_fields: list[str]) -> None:
    """
    Validate configuration dictionary

    Args:
        config: Configuration dictionary to validate
        required_fields: List of required field names

    Raises:
        AgentError: If validation fails
    """
    missing_fields = []
    for field in required_fields:
        if field not in config:
            missing_fields.append(field)

    if missing_fields:
        raise AgentError(f"Missing required configuration fields: {missing_fields}")


def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """
    Safely load JSON string with fallback

    Args:
        json_string: JSON string to parse
        default: Default value if parsing fails

    Returns:
        Parsed JSON object or default value
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    Safely dump object to JSON string with fallback

    Args:
        obj: Object to serialize
        default: Default JSON string if serialization fails

    Returns:
        JSON string or default value
    """
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize to JSON: {e}")
        return default


def normalize_score(score: float, min_score: float = 0.0, max_score: float = 1.0) -> float:
    """
    Normalize score to a specific range

    Args:
        score: Score to normalize
        min_score: Minimum value in target range
        max_score: Maximum value in target range

    Returns:
        Normalized score
    """
    if score < min_score:
        return min_score
    elif score > max_score:
        return max_score
    else:
        return score


def calculate_text_hash(text: str) -> str:
    """
    Calculate MD5 hash of text for caching

    Args:
        text: Input text

    Returns:
        MD5 hash string
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def merge_dicts(dict1: dict[str, Any], dict2: dict[str, Any]) -> dict[str, Any]:
    """
    Merge two dictionaries recursively

    Args:
        dict1: First dictionary
        dict2: Second dictionary

    Returns:
        Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def extract_field_value(data: dict[str, Any], field_path: str, default: Any = None) -> Any:
    """
    Extract value from nested dictionary using dot notation

    Args:
        data: Dictionary to extract from
        field_path: Dot-separated field path (e.g., "user.profile.name")
        default: Default value if field not found

    Returns:
        Extracted value or default
    """
    try:
        current = data
        for field in field_path.split('.'):
            if isinstance(current, dict) and field in current:
                current = current[field]
            else:
                return default
        return current
    except (TypeError, KeyError):
        return default


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def batch_process(items: list[Any], batch_size: int = 100):
    """
    Generator that yields batches of items

    Args:
        items: List of items to process
        batch_size: Size of each batch

    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def clean_text(text: str) -> str:
    """
    Clean and normalize text

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove excessive whitespace
    import re
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def calculate_similarity_threshold(scores: list[float], percentile: float = 0.8) -> float:
    """
    Calculate dynamic similarity threshold based on score distribution

    Args:
        scores: List of similarity scores
        percentile: Percentile to use as threshold

    Returns:
        Calculated threshold
    """
    if not scores:
        return 0.5

    sorted_scores = sorted(scores, reverse=True)
    index = int(len(sorted_scores) * (1 - percentile))
    return sorted_scores[min(index, len(sorted_scores) - 1)]


class Timer:
    """Context manager for timing operations"""

    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        logger.debug(f"Starting {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        if exc_type is None:
            logger.debug(f"{self.operation_name} completed in {format_duration(duration)}")
        else:
            logger.error(f"{self.operation_name} failed after {format_duration(duration)}")

    @property
    def duration(self) -> float | None:
        """Get operation duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class PerformanceMonitor:
    """Simple performance monitoring utility"""

    def __init__(self):
        self.metrics = {}

    def record_operation(self, operation: str, duration: float, success: bool = True):
        """Record operation metrics"""
        if operation not in self.metrics:
            self.metrics[operation] = {
                'count': 0,
                'total_time': 0.0,
                'successes': 0,
                'failures': 0,
                'avg_time': 0.0
            }

        metrics = self.metrics[operation]
        metrics['count'] += 1
        metrics['total_time'] += duration

        if success:
            metrics['successes'] += 1
        else:
            metrics['failures'] += 1

        metrics['avg_time'] = metrics['total_time'] / metrics['count']

    def get_metrics(self) -> dict[str, Any]:
        """Get all recorded metrics"""
        return self.metrics.copy()

    def get_operation_metrics(self, operation: str) -> dict[str, Any] | None:
        """Get metrics for a specific operation"""
        return self.metrics.get(operation)

    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.clear()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
