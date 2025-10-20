"""
LLM-based information extraction using Volcano Engine API
"""
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from volcenginesdkarkruntime import Ark

from .volcano_models import (
    VOLCANO_BASE_URL,
    is_public_model,
    validate_model_name,
    get_model_description,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractedInfo:
    """Extracted information from natural language query"""
    entities: list[dict] = field(default_factory=list)
    intent_type: str = "unknown"
    intent_confidence: float = 0.0
    key_terms: list[str] = field(default_factory=list)
    filters: list[dict] = field(default_factory=list)
    temporal_info: Optional[dict] = None
    numerical_info: list[dict] = field(default_factory=list)
    structured_query: Optional[dict] = None
    raw_llm_response: str = ""


class VolcanoLLMExtractor:
    """Information extractor using Volcano Engine's Large Language Model"""

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,  # Can be public model name or endpoint ID
        endpoint_id: Optional[str] = None,  # Deprecated, use 'model' instead
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        extraction_prompt_template: Optional[str] = None,
        prompt_file_path: Optional[str] = None,  # NEW: Path to prompt file
        max_tokens: int = 1024,
        temperature: float = 0.1,
        timeout: int = 30
    ):
        """
        Initialize Volcano LLM Extractor

        Args:
            api_key: Volcano Engine API key
            model: Model identifier - can be:
                   - Public model name (e.g., "doubao-pro-32k", "doubao-lite-32k")
                   - Endpoint ID (e.g., "ep-20240618134341-xxxxx")
            endpoint_id: (Deprecated) Use 'model' parameter instead
            base_url: API base URL (defaults to Volcano public API)
            system_prompt: System prompt for the model
            extraction_prompt_template: Template for extraction prompt (with {query} placeholder)
            prompt_file_path: Path to prompt file (if provided, overrides extraction_prompt_template)
            max_tokens: Maximum tokens in response
            temperature: Model temperature for generation
            timeout: API timeout in seconds
        """
        if not api_key:
            raise ValueError("API key is required for Volcano LLM Extractor")

        # Handle backward compatibility
        if endpoint_id and not model:
            logger.warning("Parameter 'endpoint_id' is deprecated, use 'model' instead")
            model = endpoint_id

        if not model:
            raise ValueError("Model identifier is required (public model name or endpoint ID)")

        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        # Determine if using public model or custom endpoint
        self.is_public_model = is_public_model(model)

        # Set base URL
        if self.is_public_model:
            self.base_url = base_url or VOLCANO_BASE_URL
            if validate_model_name(model):
                logger.info(f"Using public model: {get_model_description(model)}")
            else:
                logger.info(f"Using public model: {model}")
        else:
            self.base_url = base_url  # Use provided base_url or None for default
            logger.info(f"Using custom endpoint: {model}")

        # Initialize Ark client
        try:
            client_kwargs = {"api_key": api_key, "timeout": timeout}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = Ark(**client_kwargs)
            logger.info(f"Initialized Volcano LLM Extractor successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ark client: {e}")
            raise

        # Default system prompt
        self.system_prompt = system_prompt or self._get_default_system_prompt()

        # Load extraction prompt template (priority: file > parameter > default)
        self.extraction_prompt_template = self._load_extraction_prompt(
            prompt_file_path=prompt_file_path,
            template_string=extraction_prompt_template
        )

    def _load_extraction_prompt(
        self,
        prompt_file_path: Optional[str],
        template_string: Optional[str]
    ) -> str:
        """
        Load extraction prompt template with priority: file > parameter > default

        Args:
            prompt_file_path: Path to prompt template file
            template_string: Prompt template string

        Returns:
            Loaded prompt template
        """
        # Priority 1: Load from file
        if prompt_file_path:
            try:
                file_path = Path(prompt_file_path)
                if not file_path.is_absolute():
                    # Relative path: resolve relative to project root
                    project_root = Path(__file__).parent.parent
                    file_path = project_root / prompt_file_path

                if file_path.exists():
                    prompt_content = file_path.read_text(encoding='utf-8')
                    logger.info(f"Loaded extraction prompt from file: {file_path} ({len(prompt_content)} chars)")
                    return prompt_content
                else:
                    logger.warning(f"Prompt file not found: {file_path}, falling back to template string or default")
            except Exception as e:
                logger.error(f"Error loading prompt file '{prompt_file_path}': {e}, falling back to template string or default")

        # Priority 2: Use provided template string
        if template_string:
            logger.info(f"Using provided prompt template string ({len(template_string)} chars)")
            return template_string

        # Priority 3: Use default template
        logger.info("Using default extraction prompt template")
        return self._get_default_extraction_template2()

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for information extraction"""
        return """你是一个专业的信息抽取助手。你的任务是从用户的自然语言查询中抽取结构化信息。
请严格按照JSON格式返回抽取结果，不要包含其他解释性文字。"""

    def _get_default_extraction_template2(self) -> str:
        """Get default extraction prompt template"""
        return """
# 1 角色
你是一个高度智能的信息提取引擎，能够理解自然语言并从中动态抽取出关键信息，以灵活的键值对（Key-Value）形式进行组织。

# 2 任务
你的任务是分析下方的【待处理文本】，该文本描述了一个特定人群。你需要：
1.  提取描述这个人群的**所有静态属性**。
2.  提取这个人群执行的**所有行为事件**及其相关属性。
3.  以一个结构化的JSON对象输出。

# 3 核心规则
1.  **绝对原文**: 这是最高指令。所有提取出的“值”（value）都**必须**是【待处理文本】中的**原文片段，一字不差**。
2.  **动态生成键 (Key)**: 不要使用预设的字段名。对于每一个提取到的属性，你需要根据其在文中的含义，**生成一个简明扼要、符合逻辑的“键”**（key）。例如，对于原文“年龄在25到35岁之间”，应提取为 `"年龄": "25到35岁之间"`。
3.  **清晰分类**: 必须严格区分“静态属性”（描述人群特征）和“行为事件”（描述人群动作），并将它们分别放入JSON中的 `person_attributes` 和 `behavioral_events` 部分。
4.  **格式保证**: 输出必须是严格的JSON格式。如果找不到任何信息，则返回一个包含空对象和空列表的JSON。

# 4 输出结构定义
输出的JSON应包含两个主要部分：

- `person_attributes`: 一个对象。用于存放所有静态属性。其内部是**动态生成的键值对**。
- `behavioral_events`: 一个列表。用于存放所有行为事件。列表中的每个元素都是一个独立事件对象，包含：
    - `event_type`: 对事件核心动作的概括，如 "下单", "浏览", "注册"。
    - `attributes`: 一个对象，用于存放该事件的所有相关属性，其内部同样是**动态生成的键值对**。

# 5 示例 (Few-shot Learning)
请学习并严格遵循这个示例的模式，尤其是键的动态生成方式。

---
**示例输入文本:**
```
我想筛选出所有在北京的、年龄在25到35岁之间的男性软件工程师。这批用户必须是在过去90天内，通过App端至少下过3次单，并且主要购买的是数码产品。他们最近一次登录是在昨天。
```

**示例输出JSON:**
```json
{
  "person_attributes": {
    "地理位置": "北京",
    "年龄": "25到35岁之间",
    "性别": "男性",
    "职业": "软件工程师"
  },
  "behavioral_events": [
    {
      "event_type": "下单",
      "attributes": {
        "时间范围": "过去90天内",
        "渠道": "App端",
        "频率": "至少下过3次单",
        "购买品类": "数码产品"
      }
    },
    {
      "event_type": "登录",
      "attributes": {
        "最近一次登录时间": "昨天"
      }
    }
  ]
}
```
# 2 开始处理
请根据以上所有规则和定义，处理下面的文本。
【待处理文本】:
{query}
        """


    def _get_default_extraction_template(self) -> str:
        """Get default extraction prompt template"""
        return """请从以下查询中抽取信息并返回JSON格式结果：

查询: {query}

请抽取以下信息：
1. entities: 实体列表，每个实体包含 type（类型）和 value（值）
   - 支持的实体类型: person（人物）, location（地点）, time（时间）, organization（组织）, product（产品）, event（事件）, attribute（属性）, other（其他）
2. intent_type: 查询意图，可选值：search（搜索）, filter（筛选）, aggregate（聚合）, compare（比较）, analyze（分析）, other（其他）
3. intent_confidence: 意图置信度（0-1之间的小数）
4. key_terms: 关键词列表
5. filters: 筛选条件列表，每个条件包含 field（字段）, operator（操作符）, value（值）
6. temporal_info: 时间信息，包含 start_date, end_date, time_range_type
7. numerical_info: 数值信息列表，每个包含 value（数值）, unit（单位）, context（上下文）
8. structured_query: 结构化查询信息（**必须包含以下新字段**）
   - person_attributes: 人的静态属性列表，如 ["年龄", "性别", "城市"]
   - events: 事件列表，每个事件包含：
     * event_description: 事件描述（如"购买事件"、"登录事件"）
     * attributes: 该事件的属性列表（如["购买金额", "购买时间"]）

返回JSON格式示例：
{{
    "entities": [
        {{"type": "person", "value": "张三"}},
        {{"type": "time", "value": "2024年"}}
    ],
    "intent_type": "search",
    "intent_confidence": 0.95,
    "key_terms": ["用户", "年龄", "信息"],
    "filters": [
        {{"field": "age", "operator": ">", "value": 18}}
    ],
    "temporal_info": {{
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "time_range_type": "year"
    }},
    "numerical_info": [
        {{"value": 18, "unit": "岁", "context": "年龄"}}
    ],
    "structured_query": {{
        "person_attributes": ["年龄", "性别"],
        "events": [
            {{
                "event_description": "购买事件",
                "attributes": ["购买金额", "购买时间"]
            }}
        ]
    }}
}}

注意：
- 如果查询涉及人的静态属性（如年龄、性别、地址等），请在 person_attributes 中列出
- 如果查询涉及事件（如购买、登录、注册等），请在 events 中列出，并提取该事件相关的属性
- 如果不涉及人属性或事件，对应字段可以为空列表

请只返回JSON结果，不要包含其他文字。"""

    def extract(self, query: str, custom_prompt: Optional[str] = None) -> ExtractedInfo:
        """
        Extract information from natural language query using LLM

        Args:
            query: Natural language query
            custom_prompt: Optional custom prompt to override template

        Returns:
            ExtractedInfo object with extracted information
        """
        if not query:
            return ExtractedInfo()

        try:
            # Prepare the prompt
            # Use simple string replacement instead of .format() to avoid issues with JSON examples containing {}
            if custom_prompt:
                if "{query}" in custom_prompt:
                    user_prompt = custom_prompt.replace("{query}", query)
                else:
                    user_prompt = custom_prompt
            else:
                user_prompt = self.extraction_prompt_template.replace("{query}", query)

            logger.debug(f"[LLM_EXTRACT] Prepared prompt with query: {query[:50]}...")
            # Call Volcano Engine LLM
            response = self._call_llm(user_prompt)

            # Parse response
            extracted_info = self._parse_llm_response(response, query)

            return extracted_info

        except Exception as e:
            logger.error(f"Error extracting information with LLM: {e}")
            # Return basic extraction on error
            return self._fallback_extraction(query)

    def _call_llm(self, user_prompt: str) -> str:
        """
        Call Volcano Engine LLM API

        Args:
            user_prompt: User prompt for the model

        Returns:
            LLM response text
        """
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Log LLM request details
            logger.debug(f"[LLM] Calling model: {self.model}")
            logger.debug(f"[LLM] System prompt ({len(self.system_prompt)} chars): {self.system_prompt[:200]}...")
            logger.debug(f"[LLM] User prompt ({len(user_prompt)} chars): {user_prompt[:500]}...")
            logger.debug(f"[LLM] Parameters - max_tokens: {self.max_tokens}, temperature: {self.temperature}")

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            if completion and completion.choices:
                response_content = completion.choices[0].message.content
                logger.debug(f"[LLM] Response ({len(response_content)} chars): {response_content[:500]}...")
                return response_content
            else:
                raise ValueError("Empty response from LLM")

        except Exception as e:
            logger.error(f"Error calling Volcano Engine LLM: {e}")
            raise

    def _parse_llm_response(self, response: str, original_query: str) -> ExtractedInfo:
        """
        Parse LLM response to ExtractedInfo

        Args:
            response: LLM response text
            original_query: Original user query

        Returns:
            Parsed ExtractedInfo object
        """
        extracted_info = ExtractedInfo(raw_llm_response=response)

        try:
            # Try to parse JSON response
            # Remove any markdown code blocks if present
            json_str = response.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            if json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]

            parsed = json.loads(json_str.strip())

            logger.debug(f"[LLM] Successfully parsed JSON response")

            # Extract entities
            if "entities" in parsed and isinstance(parsed["entities"], list):
                extracted_info.entities = parsed["entities"]

            # Extract intent
            if "intent_type" in parsed:
                extracted_info.intent_type = parsed["intent_type"]
            if "intent_confidence" in parsed:
                extracted_info.intent_confidence = float(parsed["intent_confidence"])

            # Extract key terms
            if "key_terms" in parsed and isinstance(parsed["key_terms"], list):
                extracted_info.key_terms = parsed["key_terms"]

            # Extract filters
            if "filters" in parsed and isinstance(parsed["filters"], list):
                extracted_info.filters = parsed["filters"]

            # Extract temporal info
            if "temporal_info" in parsed and isinstance(parsed["temporal_info"], dict):
                extracted_info.temporal_info = parsed["temporal_info"]

            # Extract numerical info
            if "numerical_info" in parsed and isinstance(parsed["numerical_info"], list):
                extracted_info.numerical_info = parsed["numerical_info"]

            # Extract structured query
            # Support both formats:
            # 1. Nested format: {"structured_query": {"person_attributes": ..., "events": ...}}
            # 2. Top-level format: {"person_attributes": ..., "behavioral_events": ...}
            if "structured_query" in parsed and isinstance(parsed["structured_query"], dict):
                extracted_info.structured_query = parsed["structured_query"]
                logger.debug(f"[LLM] Parsed structured_query (nested format): {json.dumps(extracted_info.structured_query, ensure_ascii=False)[:300]}...")
            elif "person_attributes" in parsed or "behavioral_events" in parsed:
                # Convert top-level format to structured_query format
                extracted_info.structured_query = {}
                if "person_attributes" in parsed:
                    extracted_info.structured_query["person_attributes"] = parsed["person_attributes"]
                if "behavioral_events" in parsed:
                    extracted_info.structured_query["behavioral_events"] = parsed["behavioral_events"]
                logger.debug(f"[LLM] Parsed structured_query (top-level format): {json.dumps(extracted_info.structured_query, ensure_ascii=False)[:300]}...")

        except json.JSONDecodeError as e:
            logger.warning(f"[LLM] Failed to parse LLM response as JSON: {e}")
            logger.debug(f"[LLM] Raw response that failed to parse: {response[:500]}...")
            # Try to extract basic information from text
            extracted_info = self._extract_from_text(response, original_query)
        except Exception as e:
            logger.error(f"[LLM] Error parsing LLM response: {e}")

        # Log final extracted info summary
        logger.debug(
            f"[LLM] Extraction complete - "
            f"intent: {extracted_info.intent_type}, "
            f"confidence: {extracted_info.intent_confidence:.2f}, "
            f"has_structured_query: {extracted_info.structured_query is not None}"
        )

        return extracted_info

    def _extract_from_text(self, text: str, query: str) -> ExtractedInfo:
        """
        Extract basic information from text when JSON parsing fails

        Args:
            text: LLM response text
            query: Original query

        Returns:
            Basic ExtractedInfo
        """
        import re

        extracted_info = ExtractedInfo(raw_llm_response=text)

        # Extract keywords (simple word extraction)
        words = re.findall(r'\b[\w\u4e00-\u9fff]+\b', query)
        extracted_info.key_terms = [w for w in words if len(w) > 1]

        # Basic intent detection
        if any(word in query.lower() for word in ['search', '搜索', '查找', '查询']):
            extracted_info.intent_type = 'search'
            extracted_info.intent_confidence = 0.6
        elif any(word in query.lower() for word in ['filter', '筛选', '过滤']):
            extracted_info.intent_type = 'filter'
            extracted_info.intent_confidence = 0.6

        return extracted_info

    def _fallback_extraction(self, query: str) -> ExtractedInfo:
        """
        Fallback extraction when LLM is not available

        Args:
            query: User query

        Returns:
            Basic extracted info
        """
        import re

        extracted_info = ExtractedInfo()

        # Extract basic keywords
        words = re.findall(r'\b[\w\u4e00-\u9fff]+\b', query)
        extracted_info.key_terms = [w for w in words if len(w) > 1][:10]

        # Basic intent detection
        query_lower = query.lower()
        if any(word in query_lower for word in ['search', '搜索', '查找', '查询', 'find']):
            extracted_info.intent_type = 'search'
        elif any(word in query_lower for word in ['filter', '筛选', '过滤', 'where']):
            extracted_info.intent_type = 'filter'
        elif any(word in query_lower for word in ['sum', '统计', '总计', 'count', '计算']):
            extracted_info.intent_type = 'aggregate'
        elif any(word in query_lower for word in ['compare', '比较', '对比', 'vs']):
            extracted_info.intent_type = 'compare'
        else:
            extracted_info.intent_type = 'search'

        extracted_info.intent_confidence = 0.3

        # Extract numbers
        numbers = re.findall(r'\d+(?:\.\d+)?', query)
        for num in numbers:
            extracted_info.numerical_info.append({
                "value": float(num) if '.' in num else int(num),
                "unit": "unknown",
                "context": "number"
            })

        # Extract dates (simple pattern)
        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            r'\d{4}年\d{1,2}月\d{1,2}日',
        ]
        for pattern in date_patterns:
            dates = re.findall(pattern, query)
            for date in dates:
                extracted_info.entities.append({
                    "type": "time",
                    "value": date
                })

        return extracted_info

    def set_system_prompt(self, prompt: str):
        """Update system prompt"""
        self.system_prompt = prompt

    def set_extraction_template(self, template: str):
        """Update extraction prompt template"""
        self.extraction_prompt_template = template