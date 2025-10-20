"""
Result analyzer with similarity scoring, ranking, and ambiguity detection
"""
import logging
from dataclasses import dataclass
from typing import Any

from .config import SearchConfig

logger = logging.getLogger(__name__)


@dataclass
class AnalyzedResult:
    """Analyzed search result with additional metadata"""
    # Original search result fields
    id: str
    score: float
    source_type: str
    source_name: str
    field_name: str
    description: str
    raw_metadata: dict[str, Any]

    # Analysis fields
    relevance_score: float
    confidence_level: str  # 'high', 'medium', 'low'
    category: str  # 'profile', 'event', 'event_attribute'
    is_ambiguous: bool
    explanation: str

    # Event-specific fields (for event attributes)
    event_name: str | None = None
    event_description: str | None = None


@dataclass
class AnalysisResult:
    """Complete analysis result for a query"""
    query: str
    profile_attributes: list[AnalyzedResult]
    events: list[AnalyzedResult]
    event_attributes: list[AnalyzedResult]
    has_ambiguity: bool
    confidence_score: float
    total_results: int
    execution_time: float
    summary: str


class ResultAnalyzer:
    """Analyzes and ranks search results with ambiguity detection"""

    def __init__(self, config: SearchConfig):
        self.config = config

    def analyze_search_results(
        self,
        query: str,
        profile_results: list[dict[str, Any]],
        event_results: list[dict[str, Any]],
        event_attr_results: list[dict[str, Any]],
        execution_time: float = 0.0
    ) -> AnalysisResult:
        """
        Analyze complete search results from all three stages

        Args:
            query: Original user query
            profile_results: Results from profile attribute search
            event_results: Results from event search
            event_attr_results: Results from event attribute search
            execution_time: Total execution time

        Returns:
            Complete analysis result
        """
        # Analyze each category of results
        analyzed_profiles = self._analyze_profile_results(profile_results)
        analyzed_events = self._analyze_event_results(event_results)
        analyzed_event_attrs = self._analyze_event_attribute_results(event_attr_results, analyzed_events)

        # Detect overall ambiguity
        has_ambiguity = self._detect_global_ambiguity(
            analyzed_profiles, analyzed_events, analyzed_event_attrs
        )

        # Calculate overall confidence
        confidence_score = self._calculate_overall_confidence(
            analyzed_profiles, analyzed_events, analyzed_event_attrs
        )

        # Generate summary
        summary = self._generate_summary(
            query, analyzed_profiles, analyzed_events, analyzed_event_attrs
        )

        total_results = len(analyzed_profiles) + len(analyzed_events) + len(analyzed_event_attrs)

        return AnalysisResult(
            query=query,
            profile_attributes=analyzed_profiles,
            events=analyzed_events,
            event_attributes=analyzed_event_attrs,
            has_ambiguity=has_ambiguity,
            confidence_score=confidence_score,
            total_results=total_results,
            execution_time=execution_time,
            summary=summary
        )

    def _analyze_profile_results(self, results: list[dict[str, Any]]) -> list[AnalyzedResult]:
        """Analyze profile attribute search results"""
        analyzed = []

        for result in results:
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(result['score'])

            # Determine confidence level
            confidence_level = self._get_confidence_level(relevance_score)

            # Check for ambiguity
            is_ambiguous = self._is_result_ambiguous(result, results)

            # Generate explanation
            explanation = self._generate_result_explanation(result, 'profile')

            analyzed_result = AnalyzedResult(
                id=str(result.get('id', '')),
                score=result['score'],
                source_type=result['source_type'],
                source_name=result['source_name'],
                field_name=result['field_name'],
                description=result.get('raw_metadata', {}).get('desc', ''),
                raw_metadata=result.get('raw_metadata', {}),
                relevance_score=relevance_score,
                confidence_level=confidence_level,
                category='profile',
                is_ambiguous=is_ambiguous,
                explanation=explanation
            )

            analyzed.append(analyzed_result)

        # Sort by relevance score
        analyzed.sort(key=lambda x: x.relevance_score, reverse=True)

        return analyzed

    def _analyze_event_results(self, results: list[dict[str, Any]]) -> list[AnalyzedResult]:
        """Analyze event search results"""
        analyzed = []

        for result in results:
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(result['score'])

            # Determine confidence level
            confidence_level = self._get_confidence_level(relevance_score)

            # Check for ambiguity
            is_ambiguous = self._is_result_ambiguous(result, results)

            # Generate explanation
            explanation = self._generate_result_explanation(result, 'event')

            analyzed_result = AnalyzedResult(
                id=str(result.get('id', '')),
                score=result['score'],
                source_type='EVENT',  # Events are always EVENT type
                source_name=result.get('event_idname', ''),
                field_name=result.get('event_name', ''),
                description=result.get('raw_metadata', {}).get('desc', ''),
                raw_metadata=result.get('raw_metadata', {}),
                relevance_score=relevance_score,
                confidence_level=confidence_level,
                category='event',
                is_ambiguous=is_ambiguous,
                explanation=explanation,
                event_name=result.get('event_name', ''),
                event_description=result.get('event_name', '')
            )

            analyzed.append(analyzed_result)

        # Sort by relevance score
        analyzed.sort(key=lambda x: x.relevance_score, reverse=True)

        return analyzed

    def _analyze_event_attribute_results(
        self,
        results: list[dict[str, Any]],
        event_results: list[AnalyzedResult]
    ) -> list[AnalyzedResult]:
        """Analyze event attribute search results"""
        analyzed = []

        # Create mapping of event names to descriptions
        event_descriptions = {
            event.source_name: event.event_description
            for event in event_results
            if event.event_description
        }

        for result in results:
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(result['score'])

            # Determine confidence level
            confidence_level = self._get_confidence_level(relevance_score)

            # Check for ambiguity
            is_ambiguous = self._is_result_ambiguous(result, results)

            # Generate explanation
            explanation = self._generate_result_explanation(result, 'event_attribute')

            # Get event information
            event_name = result.get('source_name', '')
            event_description = event_descriptions.get(event_name, '')

            analyzed_result = AnalyzedResult(
                id=str(result.get('id', '')),
                score=result['score'],
                source_type=result['source_type'],
                source_name=result['source_name'],
                field_name=result['field_name'],
                description=result.get('raw_metadata', {}).get('desc', ''),
                raw_metadata=result.get('raw_metadata', {}),
                relevance_score=relevance_score,
                confidence_level=confidence_level,
                category='event_attribute',
                is_ambiguous=is_ambiguous,
                explanation=explanation,
                event_name=event_name,
                event_description=event_description
            )

            analyzed.append(analyzed_result)

        # Sort by relevance score
        analyzed.sort(key=lambda x: x.relevance_score, reverse=True)

        return analyzed

    def _calculate_relevance_score(self, similarity_score: float) -> float:
        """Calculate relevance score from similarity score"""
        # Normalize and enhance the score
        if similarity_score >= self.config.similarity_threshold:
            # High similarity: enhance the score
            return min(similarity_score * 1.1, 1.0)
        else:
            # Low similarity: penalize the score
            return similarity_score * 0.8

    def _get_confidence_level(self, relevance_score: float) -> str:
        """Determine confidence level based on relevance score"""
        if relevance_score >= 0.8:
            return 'high'
        elif relevance_score >= 0.6:
            return 'medium'
        else:
            return 'low'

    def _is_result_ambiguous(self, result: dict[str, Any], all_results: list[dict[str, Any]]) -> bool:
        """Check if a result is ambiguous compared to others"""
        if len(all_results) < 2:
            return False

        result_score = result['score']

        # Find other results with similar scores
        similar_results = [
            r for r in all_results
            if r != result and abs(r['score'] - result_score) < self.config.score_gap_threshold
        ]

        return len(similar_results) > 0

    def _detect_global_ambiguity(
        self,
        profile_results: list[AnalyzedResult],
        event_results: list[AnalyzedResult],
        event_attr_results: list[AnalyzedResult]
    ) -> bool:
        """Detect if there's global ambiguity across all result categories"""
        # Check if there are high-scoring results in multiple categories
        high_score_categories = []

        if profile_results and profile_results[0].relevance_score >= 0.7:
            high_score_categories.append('profile')

        if event_results and event_results[0].relevance_score >= 0.7:
            high_score_categories.append('event')

        if event_attr_results and event_attr_results[0].relevance_score >= 0.7:
            high_score_categories.append('event_attribute')

        # Global ambiguity if multiple categories have high scores
        if len(high_score_categories) > 1:
            return True

        # Check for ambiguity within individual categories
        for results in [profile_results, event_results, event_attr_results]:
            if len(results) >= 2:
                if abs(results[0].relevance_score - results[1].relevance_score) < self.config.score_gap_threshold:
                    return True

        return False

    def _calculate_overall_confidence(
        self,
        profile_results: list[AnalyzedResult],
        event_results: list[AnalyzedResult],
        event_attr_results: list[AnalyzedResult]
    ) -> float:
        """Calculate overall confidence score"""
        all_scores = []

        # Collect top scores from each category
        if profile_results:
            all_scores.append(profile_results[0].relevance_score)

        if event_results:
            all_scores.append(event_results[0].relevance_score)

        if event_attr_results:
            all_scores.append(event_attr_results[0].relevance_score)

        if not all_scores:
            return 0.0

        # Use the highest score as overall confidence
        return max(all_scores)

    def _generate_result_explanation(self, result: dict[str, Any], category: str) -> str:
        """Generate explanation for a search result"""
        if category == 'profile':
            return (f"个人属性字段: {result.get('field_name', '')} - "
                    f"{result.get('raw_metadata', {}).get('desc', '')}")

        elif category == 'event':
            return f"事件: {result.get('event_name', '')} - {result.get('raw_metadata', {}).get('desc', result.get('event_name', ''))}"

        elif category == 'event_attribute':
            return (f"事件属性: {result.get('source_name', '')} 中的 {result.get('field_name', '')} - "
                    f"{result.get('raw_metadata', {}).get('desc', '')}")

        return "未知类型"

    def _generate_summary(
        self,
        query: str,
        profile_results: list[AnalyzedResult],
        event_results: list[AnalyzedResult],
        event_attr_results: list[AnalyzedResult]
    ) -> str:
        """Generate summary of analysis results"""
        summary_parts = []

        if profile_results:
            top_profile = profile_results[0]
            summary_parts.append(f"相关个人属性: {top_profile.field_name} (置信度: {top_profile.confidence_level})")

        if event_results:
            top_event = event_results[0]
            summary_parts.append(f"相关事件: {top_event.event_name} (置信度: {top_event.confidence_level})")

        if event_attr_results:
            top_attr = event_attr_results[0]
            summary_parts.append(
                f"相关事件属性: {top_attr.source_name}.{top_attr.field_name} "
                f"(置信度: {top_attr.confidence_level})"
            )

        if not summary_parts:
            return f"未找到与查询 '{query}' 相关的结果"

        return "; ".join(summary_parts)

    def filter_results_by_threshold(self, results: list[AnalyzedResult]) -> list[AnalyzedResult]:
        """Filter results by similarity threshold"""
        return [
            result for result in results
            if result.score >= self.config.similarity_threshold
        ]

    def get_top_results(
        self,
        results: list[AnalyzedResult],
        limit: int = None
    ) -> list[AnalyzedResult]:
        """Get top N results"""
        if limit is None:
            limit = self.config.max_results

        return results[:limit]

    def format_result_for_display(self, result: AnalyzedResult) -> str:
        """Format a single result for display"""
        confidence_icon = {
            'high': '🟢',
            'medium': '🟡',
            'low': '🔴'
        }.get(result.confidence_level, '⚪')

        ambiguous_flag = ' ⚠️ 可能有歧义' if result.is_ambiguous else ''

        return (
            f"{confidence_icon} [{result.category.upper()}] "
            f"{result.source_name}.{result.field_name} "
            f"(相似度: {result.score:.3f}){ambiguous_flag}\n"
            f"   描述: {result.explanation}"
        )
