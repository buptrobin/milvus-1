"""
Embedding manager for BGE-M3 model with caching and optimization
"""
import hashlib
import logging
import time
from typing import Any

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Simple in-memory cache for embeddings"""

    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.ttl = ttl
        self.max_size = max_size

    def _get_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        """Get embedding from cache"""
        key = self._get_key(text)
        current_time = time.time()

        if key in self.cache:
            embedding, timestamp = self.cache[key]

            # Check if expired
            if current_time - timestamp > self.ttl:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                return None

            # Update access time
            self.access_times[key] = current_time
            return embedding

        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache"""
        key = self._get_key(text)
        current_time = time.time()

        # Evict oldest if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_oldest()

        self.cache[key] = (embedding, current_time)
        self.access_times[key] = current_time

    def _evict_oldest(self) -> None:
        """Evict the least recently used item"""
        if not self.access_times:
            return

        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]

    def clear(self) -> None:
        """Clear all cached embeddings"""
        self.cache.clear()
        self.access_times.clear()

    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)


class EmbeddingManager:
    """BGE-M3 embedding model manager with caching and batch processing"""

    def __init__(self, config: EmbeddingConfig, enable_cache: bool = True):
        self.config = config
        self.model = None
        self.device = None
        self.cache = EmbeddingCache() if enable_cache else None
        self._model_loaded = False

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize BGE-M3 model"""
        try:
            logger.info(f"Loading BGE-M3 model: {self.config.model_name}")

            # Check CUDA availability
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")

            # Load model with appropriate settings
            self.model = BGEM3FlagModel(
                self.config.model_name,
                use_fp16=self.config.use_fp16 and torch.cuda.is_available()
            )

            self._model_loaded = True
            logger.info("BGE-M3 model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load BGE-M3 model: {e}")
            self._model_loaded = False
            raise

    def is_ready(self) -> bool:
        """Check if model is ready"""
        return self._model_loaded and self.model is not None

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text into embedding vector

        Args:
            text: Input text to encode

        Returns:
            Embedding vector as numpy array
        """
        if not self.is_ready():
            raise RuntimeError("Embedding model is not ready")

        # Check cache first
        if self.cache:
            cached_embedding = self.cache.get(text)
            if cached_embedding is not None:
                return cached_embedding

        try:
            # Truncate text if too long
            if len(text) > self.config.max_length:
                text = text[:self.config.max_length]

            # Get dense embeddings from BGE-M3
            result = self.model.encode([text])
            embedding = result['dense_vecs'][0]

            # Convert to numpy array if not already
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()

            # Cache the result
            if self.cache:
                self.cache.put(text, embedding)

            return embedding

        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Encode multiple texts into embedding vectors

        Args:
            texts: List of input texts to encode

        Returns:
            List of embedding vectors as numpy arrays
        """
        if not self.is_ready():
            raise RuntimeError("Embedding model is not ready")

        if not texts:
            return []

        # Check cache for existing embeddings
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []

        if self.cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text)
                if cached is not None:
                    cached_embeddings[i] = cached
                else:
                    uncached_texts.append(text[:self.config.max_length])  # Truncate if needed
                    uncached_indices.append(i)
        else:
            uncached_texts = [text[:self.config.max_length] for text in texts]
            uncached_indices = list(range(len(texts)))

        # Process uncached texts in batches
        new_embeddings = {}
        if uncached_texts:
            try:
                for i in range(0, len(uncached_texts), self.config.batch_size):
                    batch_texts = uncached_texts[i:i + self.config.batch_size]
                    batch_indices = uncached_indices[i:i + self.config.batch_size]

                    # Get embeddings for batch
                    result = self.model.encode(batch_texts)
                    batch_embeddings = result['dense_vecs']

                    # Convert to numpy arrays and cache
                    for j, embedding in enumerate(batch_embeddings):
                        if isinstance(embedding, torch.Tensor):
                            embedding = embedding.cpu().numpy()

                        original_idx = batch_indices[j]
                        new_embeddings[original_idx] = embedding

                        # Cache the result
                        if self.cache:
                            self.cache.put(texts[original_idx], embedding)

            except Exception as e:
                logger.error(f"Failed to encode batch: {e}")
                raise

        # Combine cached and new embeddings in original order
        result_embeddings = []
        for i in range(len(texts)):
            if i in cached_embeddings:
                result_embeddings.append(cached_embeddings[i])
            elif i in new_embeddings:
                result_embeddings.append(new_embeddings[i])
            else:
                # Fallback: encode single text
                result_embeddings.append(self.encode_single(texts[i]))

        return result_embeddings

    def encode(self, texts: str | list[str]) -> np.ndarray | list[np.ndarray]:
        """
        Encode text(s) into embedding vector(s)

        Args:
            texts: Single text string or list of texts

        Returns:
            Single embedding array or list of embedding arrays
        """
        if isinstance(texts, str):
            return self.encode_single(texts)
        else:
            return self.encode_batch(texts)

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score between 0 and 1
        """
        emb1 = self.encode_single(text1)
        emb2 = self.encode_single(text2)

        # Calculate cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        if not self.cache:
            return {"cache_enabled": False}

        return {
            "cache_enabled": True,
            "cache_size": self.cache.size(),
            "cache_max_size": self.cache.max_size,
            "cache_ttl": self.cache.ttl
        }

    def clear_cache(self) -> None:
        """Clear embedding cache"""
        if self.cache:
            self.cache.clear()
            logger.info("Embedding cache cleared")

    def get_model_info(self) -> dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "use_fp16": self.config.use_fp16,
            "batch_size": self.config.batch_size,
            "max_length": self.config.max_length,
            "model_loaded": self._model_loaded,
            "cuda_available": torch.cuda.is_available()
        }
