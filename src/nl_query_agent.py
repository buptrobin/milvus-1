"""
Natural Language Query Agent with three-stage processing
"""
import logging
import time
from typing import Any

from .config import CONFIG, AgentConfig
from .embedding_manager import EmbeddingManager
from .llm_extractor import VolcanoLLMExtractor
from .milvus_client import MilvusClient
from .query_processor import QueryProcessor
from .result_analyzer import AnalysisResult, ResultAnalyzer

logger = logging.getLogger(__name__)


class NaturalLanguageQueryAgent:
    """
    Main agent class that processes natural language queries through three stages:
    1. Profile attributes search (PROFILE type in Pampers_metadata)
    2. Event search (in Pampers_Event_metadata)
    3. Event-specific attributes search (EVENT type with matching source_name)
    """

    def __init__(self, config: AgentConfig = None):
        self.config = config or CONFIG
        self.milvus_client = None
        self.embedding_manager = None
        self.query_processor = None
        self.result_analyzer = None
        self.llm_extractor = None
        self._initialized = False

        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize all agent components"""
        try:
            logger.info("Initializing Natural Language Query Agent...")

            # Initialize Milvus client
            logger.info("Connecting to Milvus...")
            self.milvus_client = MilvusClient(
                self.config.milvus,
                self.config.collections
            )

            # Initialize embedding manager
            logger.info("Loading embedding model...")
            self.embedding_manager = EmbeddingManager(
                self.config.embedding,
                enable_cache=self.config.enable_cache
            )

            # Initialize LLM extractor if enabled
            if self.config.volcano.enabled and self.config.volcano.api_key:
                logger.info("Initializing Volcano LLM extractor...")
                try:
                    # Use model parameter (supports both public models and endpoints)
                    model = self.config.volcano.model or self.config.volcano.endpoint_id
                    if not model:
                        logger.warning("No model or endpoint_id configured for LLM extractor")
                        self.llm_extractor = None
                    else:
                        # Log prompt template configuration
                        if self.config.volcano.prompt_file_path:
                            logger.info(f"Prompt file configured: {self.config.volcano.prompt_file_path}")
                        elif self.config.volcano.extraction_prompt_template:
                            logger.info(
                                f"Using custom extraction prompt template "
                                f"(length: {len(self.config.volcano.extraction_prompt_template)} chars)"
                            )
                        else:
                            logger.info("Using default extraction prompt template")

                        self.llm_extractor = VolcanoLLMExtractor(
                            api_key=self.config.volcano.api_key,
                            model=model,
                            base_url=self.config.volcano.base_url if self.config.volcano.base_url else None,
                            system_prompt=self.config.volcano.system_prompt if self.config.volcano.system_prompt else None,
                            extraction_prompt_template=self.config.volcano.extraction_prompt_template if self.config.volcano.extraction_prompt_template else None,
                            prompt_file_path=self.config.volcano.prompt_file_path if self.config.volcano.prompt_file_path else None,
                            max_tokens=self.config.volcano.max_tokens,
                            temperature=self.config.volcano.temperature,
                            timeout=self.config.volcano.timeout
                        )
                        logger.info("LLM extractor initialized successfully with structured extraction support")
                except Exception as e:
                    logger.warning(f"Failed to initialize LLM extractor: {e}. Continuing without LLM support.")
                    self.llm_extractor = None
            else:
                logger.info("LLM extractor is disabled or not configured")
                self.llm_extractor = None

            # Initialize query processor with optional LLM extractor
            self.query_processor = QueryProcessor(llm_extractor=self.llm_extractor)

            # Initialize result analyzer
            self.result_analyzer = ResultAnalyzer(self.config.search)

            self._initialized = True
            logger.info("Agent initialization completed successfully")

        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            self._initialized = False
            raise

    def is_ready(self) -> bool:
        """Check if agent is ready to process queries"""
        return (
            self._initialized and
            self.milvus_client and
            self.embedding_manager and
            self.embedding_manager.is_ready()
        )

    def process_query(self, query: str) -> AnalysisResult:
        """
        Process a natural language query through all three stages

        Args:
            query: User's natural language query

        Returns:
            Complete analysis result with findings from all stages
        """
        if not self.is_ready():
            raise RuntimeError("Agent is not ready. Please check initialization.")

        start_time = time.time()

        try:
            logger.info(f"Processing query: {query}")

            # Stage 0: Process and understand the query
            query_intent = self.query_processor.classify_intent(query)
            logger.info(f"Query intent: {query_intent.intent_type} (confidence: {query_intent.confidence:.2f})")

            # NEW: 如果有结构化抽取结果，优先使用定向查询
            if query_intent.structured_profile_attributes or query_intent.structured_events:
                logger.info("Using structured extraction results for targeted querying")
                return self._process_structured_query(query, query_intent)

            # 否则走原有增强搜索流程
            logger.info("Using keyword-based enhanced search strategy")
            # Generate enhanced queries for better search
            enhanced_queries = self.query_processor.enhance_query_for_search(query_intent)
            search_strategy = self.query_processor.get_search_strategy(query_intent)

            # Generate embeddings for all query variations
            embeddings = self.embedding_manager.encode(enhanced_queries)
            primary_embedding = embeddings[0]  # Use the first (original) query embedding as primary

            # Stage 1: Search profile attributes (PROFILE type)
            profile_results = []
            if search_strategy['search_profiles']:
                logger.info("Stage 1: Searching profile attributes...")
                profile_results = self._search_profile_attributes(
                    primary_embedding,
                    search_strategy,
                    enhanced_queries,
                    embeddings
                )
                logger.info(f"Found {len(profile_results)} profile attributes")

            # Stage 2: Search events
            event_results = []
            if search_strategy['search_events']:
                logger.info("Stage 2: Searching events...")
                event_results = self._search_events(
                    primary_embedding,
                    search_strategy,
                    enhanced_queries,
                    embeddings
                )
                logger.info(f"Found {len(event_results)} events")

            # Stage 3: Search event-specific attributes
            event_attr_results = []
            if search_strategy['search_event_attributes'] and event_results:
                logger.info("Stage 3: Searching event-specific attributes...")
                event_attr_results = self._search_event_attributes(
                    primary_embedding,
                    event_results,
                    search_strategy,
                    enhanced_queries,
                    embeddings
                )
                logger.info(f"Found {len(event_attr_results)} event attributes")

            # Analyze and format results
            execution_time = time.time() - start_time
            analysis_result = self.result_analyzer.analyze_search_results(
                query=query,
                profile_results=profile_results,
                event_results=event_results,
                event_attr_results=event_attr_results,
                execution_time=execution_time
            )

            logger.info(f"Query processing completed in {execution_time:.2f}s")
            return analysis_result

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            execution_time = time.time() - start_time

            # Return error result
            return AnalysisResult(
                query=query,
                profile_attributes=[],
                events=[],
                event_attributes=[],
                has_ambiguity=False,
                confidence_score=0.0,
                total_results=0,
                execution_time=execution_time,
                summary=f"查询处理失败: {str(e)}"
            )

    def _search_profile_attributes(
        self,
        primary_embedding: list[float],
        search_strategy: dict[str, Any],
        enhanced_queries: list[str],
        embeddings: list[list[float]]
    ) -> list[dict[str, Any]]:
        """Search for profile attributes (PROFILE type)"""
        try:
            limit = self.config.search.profile_search_limit
            results = self.milvus_client.search_profile_attributes(
                query_vector=primary_embedding,
                limit=limit
            )

            # If using enhanced queries and primary search didn't yield good results
            if len(results) < 3 and search_strategy.get('use_enhanced_queries', False):
                results = self._search_with_enhanced_queries(
                    'profile',
                    enhanced_queries[1:],  # Skip the first one (already used)
                    embeddings[1:],
                    limit
                )

            # Apply threshold filtering
            filtered_results = [
                r for r in results
                if r['score'] >= search_strategy.get('similarity_threshold', self.config.search.similarity_threshold)
            ]

            return filtered_results

        except Exception as e:
            logger.error(f"Error searching profile attributes: {e}")
            return []

    def _search_events(
        self,
        primary_embedding: list[float],
        search_strategy: dict[str, Any],
        enhanced_queries: list[str],
        embeddings: list[list[float]]
    ) -> list[dict[str, Any]]:
        """Search for events"""
        try:
            limit = self.config.search.event_search_limit
            results = self.milvus_client.search_events(
                query_vector=primary_embedding,
                limit=limit
            )

            # If using enhanced queries and primary search didn't yield good results
            if len(results) < 3 and search_strategy.get('use_enhanced_queries', False):
                results = self._search_with_enhanced_queries(
                    'events',
                    enhanced_queries[1:],
                    embeddings[1:],
                    limit
                )

            # Apply threshold filtering
            filtered_results = [
                r for r in results
                if r['score'] >= search_strategy.get('similarity_threshold', self.config.search.similarity_threshold)
            ]

            return filtered_results

        except Exception as e:
            logger.error(f"Error searching events: {e}")
            return []

    def _search_event_attributes(
        self,
        primary_embedding: list[float],
        event_results: list[dict[str, Any]],
        search_strategy: dict[str, Any],
        enhanced_queries: list[str],
        embeddings: list[list[float]]
    ) -> list[dict[str, Any]]:
        """Search for attributes within specific events"""
        try:
            # Extract event names from event results
            event_names = []
            for event in event_results:
                event_idname = event.get('event_idname', '')
                if event_idname:
                    event_names.append(event_idname)

            if not event_names:
                return []

            limit = self.config.search.event_attr_search_limit
            results = self.milvus_client.search_event_attributes(
                query_vector=primary_embedding,
                event_names=event_names,
                limit=limit
            )

            # Apply threshold filtering
            filtered_results = [
                r for r in results
                if r['score'] >= search_strategy.get('similarity_threshold', self.config.search.similarity_threshold)
            ]

            return filtered_results

        except Exception as e:
            logger.error(f"Error searching event attributes: {e}")
            return []

    def _search_with_enhanced_queries(
        self,
        search_type: str,
        enhanced_queries: list[str],
        embeddings: list[list[float]],
        limit: int
    ) -> list[dict[str, Any]]:
        """Search using enhanced query variations"""
        all_results = []

        for _query, embedding in zip(enhanced_queries, embeddings, strict=False):
            try:
                if search_type == 'profile':
                    results = self.milvus_client.search_profile_attributes(
                        query_vector=embedding,
                        limit=limit // 2
                    )
                elif search_type == 'events':
                    results = self.milvus_client.search_events(
                        query_vector=embedding,
                        limit=limit // 2
                    )
                else:
                    continue

                all_results.extend(results)

            except Exception as e:
                logger.warning(f"Error in enhanced query search: {e}")
                continue

        # Remove duplicates based on ID and sort by score
        seen_ids = set()
        unique_results = []

        for result in sorted(all_results, key=lambda x: x['score'], reverse=True):
            result_id = result.get('id', '')
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)

        return unique_results[:limit]

    def _process_structured_query(
        self,
        query: str,
        query_intent: 'QueryIntent'
    ) -> AnalysisResult:
        """
        使用LLM抽取的结构化信息进行定向查询

        处理流程:
        1. 如果有人属性 -> 查 Pampers_metadata (source_type='PROFILE')
        2. 如果有事件 -> 先查 Pampers_Event_metadata 获取 event_idname
                      -> 再查 Pampers_metadata (source_type='EVENT' + event_idname)
        """
        from .query_processor import QueryIntent
        start_time = time.time()
        profile_results = []
        event_results = []
        event_attr_results = []

        try:
            # 1. 处理人静态属性查询
            if query_intent.structured_profile_attributes:
                profile_texts = [p.query_text for p in query_intent.structured_profile_attributes]
                logger.info(f"Structured query: Searching {len(profile_texts)} profile attributes: {profile_texts}")

                # 批量生成embeddings
                profile_embeddings = self.embedding_manager.encode(profile_texts)

                for i, embedding in enumerate(profile_embeddings):
                    try:
                        results = self.milvus_client.search_profile_attributes(
                            query_vector=embedding,
                            limit=3
                        )
                        profile_results.extend(results)
                        logger.debug(f"Profile attribute '{profile_texts[i]}' found {len(results)} results")
                    except Exception as e:
                        logger.warning(f"Error searching profile attribute '{profile_texts[i]}': {e}")

            # 2. 处理事件查询（两步流程）
            if query_intent.structured_events:
                logger.info(f"Structured query: Processing {len(query_intent.structured_events)} events")

                for event_info in query_intent.structured_events:
                    try:
                        # Step 1: 查询事件元数据获取 event_idname
                        logger.debug(f"Searching event metadata for: {event_info.event_description}")
                        event_embedding = self.embedding_manager.encode([event_info.event_description])[0]
                        event_meta_results = self.milvus_client.search_events(
                            query_vector=event_embedding,
                            limit=2
                        )
                        event_results.extend(event_meta_results)
                        logger.debug(f"Event '{event_info.event_description}' found {len(event_meta_results)} metadata results")

                        # Step 2: 查询事件属性
                        if event_info.event_attributes and event_meta_results:
                            # 提取 event_idname
                            event_idnames = [e.get('event_idname', '') for e in event_meta_results if e.get('event_idname')]

                            if event_idnames:
                                logger.debug(
                                    f"Searching event attributes for event_idnames={event_idnames}, "
                                    f"attributes={event_info.event_attributes}"
                                )

                                # 为每个事件属性生成embedding
                                attr_embeddings = self.embedding_manager.encode(event_info.event_attributes)

                                for j, attr_embedding in enumerate(attr_embeddings):
                                    try:
                                        attr_results = self.milvus_client.search_event_attributes(
                                            query_vector=attr_embedding,
                                            event_idnames=event_idnames,  # 使用新参数
                                            limit=5
                                        )
                                        event_attr_results.extend(attr_results)
                                        logger.debug(
                                            f"Event attribute '{event_info.event_attributes[j]}' "
                                            f"found {len(attr_results)} results"
                                        )
                                    except Exception as e:
                                        logger.warning(f"Error searching event attribute '{event_info.event_attributes[j]}': {e}")
                            else:
                                logger.warning(f"No event_idname found for event: {event_info.event_description}")

                    except Exception as e:
                        logger.warning(f"Error processing event '{event_info.event_description}': {e}")

            # 去重
            profile_results = self._deduplicate_results(profile_results)
            event_results = self._deduplicate_results(event_results)
            event_attr_results = self._deduplicate_results(event_attr_results)

            logger.info(
                f"Structured query completed: {len(profile_results)} profiles, "
                f"{len(event_results)} events, {len(event_attr_results)} event attributes"
            )

        except Exception as e:
            logger.error(f"Structured query failed: {e}, falling back to keyword search")
            return self._fallback_query(query, query_intent)

        # 分析结果
        execution_time = time.time() - start_time
        return self.result_analyzer.analyze_search_results(
            query=query,
            profile_results=profile_results,
            event_results=event_results,
            event_attr_results=event_attr_results,
            execution_time=execution_time
        )

    def _deduplicate_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """去重结果（根据ID）并保持顺序"""
        seen = set()
        unique = []
        for r in results:
            rid = r.get('id')
            if rid not in seen:
                seen.add(rid)
                unique.append(r)
        return unique

    def _fallback_query(
        self,
        query: str,
        query_intent: 'QueryIntent'
    ) -> AnalysisResult:
        """LLM抽取失败或结构化查询出错时的回退逻辑"""
        from .query_processor import QueryIntent
        logger.warning("Using fallback keyword-based search strategy")
        start_time = time.time()

        try:
            # 使用原有的关键词增强策略
            enhanced_queries = self.query_processor.enhance_query_for_search(query_intent, use_llm_info=False)
            search_strategy = self.query_processor.get_search_strategy(query_intent)

            # 生成embeddings
            embeddings = self.embedding_manager.encode(enhanced_queries)
            primary_embedding = embeddings[0]

            # 执行三阶段搜索
            profile_results = []
            event_results = []
            event_attr_results = []

            if search_strategy['search_profiles']:
                profile_results = self._search_profile_attributes(
                    primary_embedding,
                    search_strategy,
                    enhanced_queries,
                    embeddings
                )

            if search_strategy['search_events']:
                event_results = self._search_events(
                    primary_embedding,
                    search_strategy,
                    enhanced_queries,
                    embeddings
                )

            if search_strategy['search_event_attributes'] and event_results:
                event_attr_results = self._search_event_attributes(
                    primary_embedding,
                    event_results,
                    search_strategy,
                    enhanced_queries,
                    embeddings
                )

            execution_time = time.time() - start_time
            return self.result_analyzer.analyze_search_results(
                query=query,
                profile_results=profile_results,
                event_results=event_results,
                event_attr_results=event_attr_results,
                execution_time=execution_time
            )

        except Exception as e:
            logger.error(f"Fallback query also failed: {e}")
            execution_time = time.time() - start_time
            return AnalysisResult(
                query=query,
                profile_attributes=[],
                events=[],
                event_attributes=[],
                has_ambiguity=False,
                confidence_score=0.0,
                total_results=0,
                execution_time=execution_time,
                summary=f"查询处理失败: {str(e)}"
            )

    def get_agent_status(self) -> dict[str, Any]:
        """Get current agent status and statistics"""
        status = {
            "initialized": self._initialized,
            "ready": self.is_ready(),
            "config": {
                "milvus_host": self.config.milvus.host,
                "milvus_port": self.config.milvus.port,
                "embedding_model": self.config.embedding.model_name,
                "cache_enabled": self.config.enable_cache
            }
        }

        if self.embedding_manager:
            status["embedding"] = self.embedding_manager.get_model_info()
            status["cache"] = self.embedding_manager.get_cache_stats()

        if self.milvus_client:
            status["collections"] = {
                "metadata": self.milvus_client.collection_info(self.config.collections.metadata_collection),
                "events": self.milvus_client.collection_info(self.config.collections.event_collection)
            }

        return status

    def clear_cache(self) -> None:
        """Clear all caches"""
        if self.embedding_manager:
            self.embedding_manager.clear_cache()
            logger.info("Agent cache cleared")

    def shutdown(self) -> None:
        """Shutdown agent and cleanup resources"""
        try:
            if self.milvus_client:
                self.milvus_client.disconnect()

            if self.embedding_manager:
                self.embedding_manager.clear_cache()

            self._initialized = False
            logger.info("Agent shutdown completed")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
