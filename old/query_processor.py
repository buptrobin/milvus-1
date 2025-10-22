"""
Query processor for natural language understanding and preprocessing
"""
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .llm_extractor import ExtractedInfo, VolcanoLLMExtractor

logger = logging.getLogger(__name__)


@dataclass
class ProfileAttribute:
    """人的静态属性 (Person static attribute)"""
    attribute_name: str  # 属性名称（如"年龄"）
    query_text: str      # 用于向量检索的文本


@dataclass
class EventInfo:
    """事件信息 (Event information)"""
    event_description: str           # 事件描述（如"购买事件"）
    event_attributes: list[str] = field(default_factory=list)  # 事件相关属性列表


@dataclass
class QueryIntent:
    """Query intent classification result"""
    original_query: str
    processed_query: str
    intent_type: str  # 'profile', 'event', 'mixed', 'unknown'
    confidence: float
    keywords: list[str]
    entities: list[str]
    llm_extracted_info: Optional[ExtractedInfo] = None
    structured_profile_attributes: list[ProfileAttribute] = field(default_factory=list)
    structured_events: list[EventInfo] = field(default_factory=list)


def parse_structured_extraction(llm_response: ExtractedInfo) -> tuple[list[ProfileAttribute], list[EventInfo]]:
    """
    解析LLM响应为结构化数据

    支持两种JSON格式 (在structured_query字段中):

    格式1 (新格式 - prompt.txt):
    {
        "person_attributes": {"年龄": "25到35岁", "性别": "男性"},
        "behavioral_events": [
            {
                "event_type": "购买",
                "attributes": {"时间范围": "过去90天", "频率": "至少3次"}
            }
        ]
    }

    格式2 (旧格式 - 向后兼容):
    {
        "person_attributes": ["年龄", "性别"],
        "events": [
            {
                "event_description": "购买事件",
                "attributes": ["购买金额", "购买时间"]
            }
        ]
    }

    Args:
        llm_response: LLM抽取的信息对象

    Returns:
        (profile_attributes, events) 元组
    """
    profile_attrs: list[ProfileAttribute] = []
    events: list[EventInfo] = []

    if not llm_response or not llm_response.structured_query:
        return profile_attrs, events

    try:
        structured_data = llm_response.structured_query

        # 解析人属性 (支持对象和列表两种格式)
        if "person_attributes" in structured_data:
            person_attrs_data = structured_data["person_attributes"]

            # 格式1: 对象 {"年龄": "25到35岁", "性别": "男性"}
            if isinstance(person_attrs_data, dict):
                for key, value in person_attrs_data.items():
                    if key and str(value).strip():
                        profile_attrs.append(ProfileAttribute(
                            attribute_name=key.strip(),
                            query_text=f"{key.strip()}: {str(value).strip()}"  # 组合键值作为查询文本
                        ))

            # 格式2: 列表 ["年龄", "性别"]
            elif isinstance(person_attrs_data, list):
                for attr in person_attrs_data:
                    if attr and isinstance(attr, str) and attr.strip():
                        profile_attrs.append(ProfileAttribute(
                            attribute_name=attr.strip(),
                            query_text=attr.strip()
                        ))

        # 解析事件 (支持 behavioral_events 和 events 两种字段名)
        events_key = "behavioral_events" if "behavioral_events" in structured_data else "events"

        if events_key in structured_data and isinstance(structured_data[events_key], list):
            for event_data in structured_data[events_key]:
                if not isinstance(event_data, dict):
                    continue

                # 获取事件类型/描述 (支持 event_type 和 event_description)
                event_desc = event_data.get("event_type", "") or event_data.get("event_description", "")
                if not event_desc or not event_desc.strip():
                    continue

                event_attrs_data = event_data.get("attributes", [])

                # 处理事件属性 (支持对象和列表格式)
                event_attrs = []
                if isinstance(event_attrs_data, dict):
                    # 格式1: 对象 {"时间范围": "过去90天", "频率": "至少3次"}
                    for key, value in event_attrs_data.items():
                        if key and str(value).strip():
                            event_attrs.append(f"{key.strip()}: {str(value).strip()}")
                elif isinstance(event_attrs_data, list):
                    # 格式2: 列表 ["购买金额", "购买时间"]
                    event_attrs = [attr.strip() for attr in event_attrs_data if attr and isinstance(attr, str) and attr.strip()]

                events.append(EventInfo(
                    event_description=event_desc.strip(),
                    event_attributes=event_attrs
                ))

    except Exception as e:
        logger.warning(f"Error parsing structured extraction: {e}")
        return [], []

    return profile_attrs, events


class QueryProcessor:
    """Natural language query processor with intent detection and preprocessing"""

    def __init__(self, llm_extractor: Optional[VolcanoLLMExtractor] = None):
        self.llm_extractor = llm_extractor
        # Profile-related keywords (person attributes)
        self.profile_keywords = {
            'zh': [
                '用户', '会员', '客户', '人员', '个人', '档案', '资料',
                '年龄', '性别', '生日', '电话', '邮箱', '地址', '国家', '城市',
                '注册', '登录', '身份', '编号', 'ID', '标识', '名称', '姓名',
                '等级', '级别', '状态', '类型', '分类', '标签', '属性'
            ],
            'en': [
                'user', 'member', 'customer', 'person', 'personal', 'profile',
                'age', 'gender', 'birthday', 'phone', 'email', 'address', 'country', 'city',
                'register', 'login', 'identity', 'id', 'name', 'level', 'status',
                'type', 'category', 'tag', 'attribute'
            ]
        }

        # Event-related keywords
        self.event_keywords = {
            'zh': [
                '事件', '活动', '行为', '操作', '记录', '日志', '历史',
                '购买', '下单', '支付', '交易', '订单', '商品', '产品',
                '积分', '兑换', '核销', '赠送', '获得', '消费', '使用',
                '绑定', '关注', '分享', '参与', '签到', '抽奖', '活动',
                '时间', '日期', '频次', '次数', '金额', '数量', '渠道'
            ],
            'en': [
                'event', 'activity', 'action', 'operation', 'record', 'log', 'history',
                'purchase', 'order', 'payment', 'transaction', 'product', 'item',
                'points', 'redeem', 'exchange', 'give', 'earn', 'spend', 'use',
                'bind', 'follow', 'share', 'participate', 'checkin', 'lottery',
                'time', 'date', 'frequency', 'count', 'amount', 'quantity', 'channel'
            ]
        }

        # Common question patterns
        self.question_patterns = {
            'zh': [
                r'什么是', r'哪些', r'哪个', r'什么时候', r'怎么', r'如何',
                r'有哪些', r'包含', r'涉及', r'相关', r'关于', r'查询', r'搜索'
            ],
            'en': [
                r'what\s+is', r'which', r'what', r'when', r'how', r'where',
                r'what\s+are', r'include', r'involve', r'relate', r'about', r'search', r'find'
            ]
        }

    def preprocess_query(self, query: str) -> str:
        """Preprocess and normalize query text"""
        if not query:
            return ""

        # Basic cleaning
        processed = query.strip()

        # Remove excessive whitespace
        processed = re.sub(r'\s+', ' ', processed)

        # Remove special characters that might interfere with search
        processed = re.sub(r'[^\w\s\u4e00-\u9fff\u3400-\u4dbf]', ' ', processed)

        # Normalize whitespace again
        processed = re.sub(r'\s+', ' ', processed).strip()

        return processed

    def extract_keywords(self, query: str) -> list[str]:
        """Extract meaningful keywords from query"""
        if not query:
            return []

        # Convert to lowercase for English matching
        # query_lower = query.lower()  # Unused variable

        keywords = []

        # Extract words (both Chinese and English)
        words = re.findall(r'[\w\u4e00-\u9fff\u3400-\u4dbf]+', query)

        for word in words:
            # Skip very short words
            if len(word) < 2:
                continue

            # Skip common stop words
            if self._is_stop_word(word):
                continue

            keywords.append(word)

        return list(set(keywords))  # Remove duplicates

    def _is_stop_word(self, word: str) -> bool:
        """Check if word is a stop word"""
        zh_stop_words = {'的', '是', '在', '有', '和', '与', '或', '及', '这', '那', '了', '吗', '呢', '吧'}
        en_stop_words = {'the', 'is', 'in', 'and', 'or', 'a', 'an', 'this', 'that', 'of', 'to', 'for'}

        word_lower = word.lower()
        return word_lower in zh_stop_words or word_lower in en_stop_words

    def detect_language(self, query: str) -> str:
        """Detect query language (zh/en/mixed)"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', query))
        english_chars = len(re.findall(r'[a-zA-Z]', query))
        total_chars = chinese_chars + english_chars

        if total_chars == 0:
            return 'unknown'

        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars

        if chinese_ratio > 0.6:
            return 'zh'
        elif english_ratio > 0.6:
            return 'en'
        else:
            return 'mixed'

    def classify_intent(self, query: str) -> QueryIntent:
        """Classify query intent and extract relevant information"""
        processed_query = self.preprocess_query(query)
        keywords = self.extract_keywords(processed_query)
        # language = self.detect_language(processed_query)  # Unused variable

        # Try LLM extraction first if available
        llm_extracted_info = None
        structured_profiles: list[ProfileAttribute] = []
        structured_events: list[EventInfo] = []

        if self.llm_extractor:
            logger.info(f"llm_extractor processing query: {processed_query}")
            try:
                extraction_start = time.time()
                llm_extracted_info = self.llm_extractor.extract(query)
                extraction_time = time.time() - extraction_start
                logger.info(f"LLM extracted info: {llm_extracted_info}")
                logger.info(f"LLM extraction completed in {extraction_time:.2f}s")
                # Use LLM-extracted information to enhance classification
                if llm_extracted_info.intent_type != "unknown":
                    # Augment keywords with LLM-extracted key terms
                    if llm_extracted_info.key_terms:
                        keywords = list(set(keywords + llm_extracted_info.key_terms))

                # Parse structured extraction results
                if llm_extracted_info.structured_query:
                    try:
                        parsing_start = time.time()
                        structured_profiles, structured_events = parse_structured_extraction(llm_extracted_info)
                        parsing_time = time.time() - parsing_start
                        logger.info(
                            f"Structured extraction parsed in {parsing_time:.3f}s: "
                            f"{len(structured_profiles)} profile attributes, {len(structured_events)} events"
                        )
                    except Exception as e:
                        logger.warning(f"Structured extraction parsing failed: {e}")

            except Exception as e:
                logger.warning(f"LLM extraction failed, falling back to rule-based: {e}")

        # Count profile and event keyword matches
        profile_score = 0
        event_score = 0

        query_lower = processed_query.lower()

        # Check profile keywords
        for lang in ['zh', 'en']:
            for keyword in self.profile_keywords[lang]:
                if keyword in query_lower:
                    profile_score += 1

        # Check event keywords
        for lang in ['zh', 'en']:
            for keyword in self.event_keywords[lang]:
                if keyword in query_lower:
                    event_score += 1

        # Determine intent type
        intent_type = 'unknown'
        confidence = 0.0

        if profile_score > 0 and event_score > 0:
            intent_type = 'mixed'
            confidence = min(profile_score, event_score) / max(profile_score, event_score)
        elif profile_score > event_score:
            intent_type = 'profile'
            confidence = min(profile_score / len(keywords), 1.0) if keywords else 0.1
        elif event_score > profile_score:
            intent_type = 'event'
            confidence = min(event_score / len(keywords), 1.0) if keywords else 0.1
        else:
            intent_type = 'mixed'  # Default to mixed when uncertain
            confidence = 0.3

        # Extract entities (potential field names, values)
        entities = self._extract_entities(processed_query)

        # Merge LLM-extracted entities if available
        if llm_extracted_info and llm_extracted_info.entities:
            llm_entity_values = [e.get('value', '') for e in llm_extracted_info.entities if e.get('value')]
            entities = list(set(entities + llm_entity_values))

        # If LLM provided intent, use it with higher confidence
        if llm_extracted_info and llm_extracted_info.intent_type != "unknown":
            llm_intent_map = {
                'search': 'mixed',
                'filter': 'mixed',
                'aggregate': 'event',
                'compare': 'mixed',
                'analyze': 'mixed'
            }
            if llm_extracted_info.intent_confidence > confidence:
                intent_type = llm_intent_map.get(llm_extracted_info.intent_type, intent_type)
                confidence = max(confidence, llm_extracted_info.intent_confidence)

        return QueryIntent(
            original_query=query,
            processed_query=processed_query,
            intent_type=intent_type,
            confidence=confidence,
            keywords=keywords,
            entities=entities,
            llm_extracted_info=llm_extracted_info,
            structured_profile_attributes=structured_profiles,
            structured_events=structured_events
        )

    def _extract_entities(self, query: str) -> list[str]:
        """Extract potential entities from query"""
        entities = []

        # Extract quoted strings (likely exact field names or values)
        quoted_matches = re.findall(r'["\']([^"\']+)["\']', query)
        entities.extend(quoted_matches)

        # Extract potential field names (words followed by common field indicators)
        field_patterns = [
            r'(\w+)(?:字段|属性|信息|数据)',
            r'(\w+)(?:\s+field|\s+attribute|\s+info|\s+data)',
            r'(\w+)[_\-](\w+)',  # underscore or hyphen separated terms
        ]

        for pattern in field_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    entities.extend(match)
                else:
                    entities.append(match)

        # Extract potential values (numbers, dates, specific terms)
        value_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # dates
            r'\d+',  # numbers
            r'[A-Z_]{3,}',  # uppercase terms (often enum values)
        ]

        for pattern in value_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)

        return list(set(entities))  # Remove duplicates

    def enhance_query_for_search(self, query_intent: QueryIntent, use_llm_info: bool = True) -> list[str]:
        """
        Generate enhanced queries for better search results

        Args:
            query_intent: Classified query intent

        Returns:
            List of enhanced query strings for searching
        """
        enhanced_queries = []

        # Original processed query
        enhanced_queries.append(query_intent.processed_query)

        # Add keyword-based variations
        if query_intent.keywords:
            # Join keywords with spaces
            keyword_query = ' '.join(query_intent.keywords)
            enhanced_queries.append(keyword_query)

            # Create focused queries with most important keywords
            if len(query_intent.keywords) > 3:
                # Take first 3 most relevant keywords
                focused_query = ' '.join(query_intent.keywords[:3])
                enhanced_queries.append(focused_query)

        # Add entity-based queries
        if query_intent.entities:
            for entity in query_intent.entities:
                enhanced_queries.append(entity)

        # Add LLM-extracted information if available
        if use_llm_info and query_intent.llm_extracted_info:
            llm_info = query_intent.llm_extracted_info

            # Add key terms from LLM
            if llm_info.key_terms:
                enhanced_queries.append(' '.join(llm_info.key_terms[:5]))

            # Add filter fields if any
            if llm_info.filters:
                for filter_item in llm_info.filters[:3]:
                    if 'field' in filter_item:
                        enhanced_queries.append(filter_item['field'])

        # Add intent-specific enhancements
        if query_intent.intent_type == 'profile':
            enhanced_queries.append(f"{query_intent.processed_query} 用户 个人 属性")
            enhanced_queries.append(f"{query_intent.processed_query} user profile attribute")

        elif query_intent.intent_type == 'event':
            enhanced_queries.append(f"{query_intent.processed_query} 事件 活动 记录")
            enhanced_queries.append(f"{query_intent.processed_query} event activity record")

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for query in enhanced_queries:
            if query and query not in seen:
                seen.add(query)
                unique_queries.append(query)

        return unique_queries

    def get_search_strategy(self, query_intent: QueryIntent) -> dict[str, Any]:
        """
        Determine search strategy based on query intent

        Args:
            query_intent: Classified query intent

        Returns:
            Dictionary containing search strategy configuration
        """
        strategy = {
            'search_profiles': True,
            'search_events': True,
            'search_event_attributes': True,
            'profile_weight': 1.0,
            'event_weight': 1.0,
            'event_attr_weight': 1.0,
            'use_enhanced_queries': True,
            'similarity_threshold': 0.65
        }

        # Adjust strategy based on intent
        if query_intent.intent_type == 'profile':
            strategy['profile_weight'] = 1.5
            strategy['event_weight'] = 0.7
            strategy['event_attr_weight'] = 0.8

        elif query_intent.intent_type == 'event':
            strategy['profile_weight'] = 0.7
            strategy['event_weight'] = 1.5
            strategy['event_attr_weight'] = 1.2

        elif query_intent.intent_type == 'mixed':
            # Balanced approach for mixed intent
            strategy['profile_weight'] = 1.0
            strategy['event_weight'] = 1.0
            strategy['event_attr_weight'] = 1.0

        # Adjust similarity threshold based on confidence
        if query_intent.confidence > 0.8:
            strategy['similarity_threshold'] = 0.7  # Higher threshold for high confidence
        elif query_intent.confidence < 0.3:
            strategy['similarity_threshold'] = 0.5  # Lower threshold for low confidence

        return strategy
