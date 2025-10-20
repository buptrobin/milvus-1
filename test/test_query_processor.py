"""
Unit tests for query processor structured extraction
"""
import pytest
from src.llm_extractor import ExtractedInfo
from src.query_processor import parse_structured_extraction, ProfileAttribute, EventInfo


def test_parse_structured_extraction_with_profile_attributes():
    """测试解析人属性"""
    llm_response = ExtractedInfo(
        structured_query={
            "person_attributes": ["年龄", "性别", "城市"],
            "events": []
        }
    )

    profiles, events = parse_structured_extraction(llm_response)

    assert len(profiles) == 3
    assert profiles[0].attribute_name == "年龄"
    assert profiles[0].query_text == "年龄"
    assert profiles[1].attribute_name == "性别"
    assert profiles[2].attribute_name == "城市"
    assert len(events) == 0


def test_parse_structured_extraction_with_events():
    """测试解析事件"""
    llm_response = ExtractedInfo(
        structured_query={
            "person_attributes": [],
            "events": [
                {
                    "event_description": "购买事件",
                    "attributes": ["购买金额", "购买时间"]
                },
                {
                    "event_description": "登录事件",
                    "attributes": ["登录IP", "登录时间", "设备类型"]
                }
            ]
        }
    )

    profiles, events = parse_structured_extraction(llm_response)

    assert len(profiles) == 0
    assert len(events) == 2

    assert events[0].event_description == "购买事件"
    assert len(events[0].event_attributes) == 2
    assert "购买金额" in events[0].event_attributes
    assert "购买时间" in events[0].event_attributes

    assert events[1].event_description == "登录事件"
    assert len(events[1].event_attributes) == 3


def test_parse_structured_extraction_mixed():
    """测试同时解析人属性和事件"""
    llm_response = ExtractedInfo(
        structured_query={
            "person_attributes": ["年龄", "性别"],
            "events": [
                {
                    "event_description": "注册事件",
                    "attributes": ["注册渠道", "注册时间"]
                }
            ]
        }
    )

    profiles, events = parse_structured_extraction(llm_response)

    assert len(profiles) == 2
    assert len(events) == 1
    assert profiles[0].attribute_name == "年龄"
    assert events[0].event_description == "注册事件"


def test_parse_structured_extraction_empty():
    """测试空结果"""
    llm_response = ExtractedInfo(structured_query={})
    profiles, events = parse_structured_extraction(llm_response)
    assert len(profiles) == 0
    assert len(events) == 0


def test_parse_structured_extraction_none():
    """测试None输入"""
    profiles, events = parse_structured_extraction(None)
    assert len(profiles) == 0
    assert len(events) == 0


def test_parse_structured_extraction_no_structured_query():
    """测试没有structured_query字段"""
    llm_response = ExtractedInfo()
    profiles, events = parse_structured_extraction(llm_response)
    assert len(profiles) == 0
    assert len(events) == 0


def test_parse_structured_extraction_filters_empty_strings():
    """测试过滤空字符串"""
    llm_response = ExtractedInfo(
        structured_query={
            "person_attributes": ["年龄", "", "  ", "性别"],
            "events": [
                {
                    "event_description": "购买事件",
                    "attributes": ["金额", "", "  ", "时间"]
                }
            ]
        }
    )

    profiles, events = parse_structured_extraction(llm_response)

    assert len(profiles) == 2  # 只保留非空值
    assert "年龄" in [p.attribute_name for p in profiles]
    assert "性别" in [p.attribute_name for p in profiles]

    assert len(events) == 1
    assert len(events[0].event_attributes) == 2  # 过滤掉空字符串
    assert "金额" in events[0].event_attributes
    assert "时间" in events[0].event_attributes


def test_parse_structured_extraction_invalid_event_format():
    """测试无效的事件格式（缺少event_description）"""
    llm_response = ExtractedInfo(
        structured_query={
            "person_attributes": [],
            "events": [
                {
                    # 缺少 event_description
                    "attributes": ["属性1", "属性2"]
                },
                {
                    "event_description": "",  # 空描述
                    "attributes": ["属性3"]
                },
                {
                    "event_description": "有效事件",
                    "attributes": ["属性4"]
                }
            ]
        }
    )

    profiles, events = parse_structured_extraction(llm_response)

    # 只保留有效的事件（event_description非空）
    assert len(events) == 1
    assert events[0].event_description == "有效事件"


def test_parse_structured_extraction_wrong_types():
    """测试错误的数据类型"""
    llm_response = ExtractedInfo(
        structured_query={
            "person_attributes": "not a list",  # 应该是列表
            "events": "also not a list"
        }
    )

    profiles, events = parse_structured_extraction(llm_response)

    # 应该优雅处理错误类型
    assert len(profiles) == 0
    assert len(events) == 0


def test_parse_structured_extraction_event_without_attributes():
    """测试没有attributes字段的事件"""
    llm_response = ExtractedInfo(
        structured_query={
            "person_attributes": [],
            "events": [
                {
                    "event_description": "简单事件"
                    # 缺少 attributes 字段
                }
            ]
        }
    )

    profiles, events = parse_structured_extraction(llm_response)

    assert len(events) == 1
    assert events[0].event_description == "简单事件"
    assert len(events[0].event_attributes) == 0  # 默认空列表


# ===== 新格式测试 (prompt.txt格式) =====

def test_parse_new_format_person_attributes_dict():
    """测试新格式：person_attributes为字典"""
    llm_response = ExtractedInfo(
        structured_query={
            "person_attributes": {
                "年龄": "25到35岁之间",
                "性别": "男性",
                "地理位置": "北京"
            },
            "behavioral_events": []
        }
    )

    profiles, events = parse_structured_extraction(llm_response)

    assert len(profiles) == 3
    # 验证query_text包含键值对
    query_texts = [p.query_text for p in profiles]
    assert "年龄: 25到35岁之间" in query_texts
    assert "性别: 男性" in query_texts
    assert "地理位置: 北京" in query_texts
    assert len(events) == 0


def test_parse_new_format_behavioral_events():
    """测试新格式：behavioral_events"""
    llm_response = ExtractedInfo(
        structured_query={
            "person_attributes": {},
            "behavioral_events": [
                {
                    "event_type": "下单",
                    "attributes": {
                        "时间范围": "过去90天内",
                        "渠道": "App端",
                        "频率": "至少下过3次单"
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
    )

    profiles, events = parse_structured_extraction(llm_response)

    assert len(profiles) == 0
    assert len(events) == 2

    # 验证第一个事件
    assert events[0].event_description == "下单"
    assert len(events[0].event_attributes) == 3
    assert "时间范围: 过去90天内" in events[0].event_attributes
    assert "渠道: App端" in events[0].event_attributes

    # 验证第二个事件
    assert events[1].event_description == "登录"
    assert len(events[1].event_attributes) == 1
    assert "最近一次登录时间: 昨天" in events[1].event_attributes


def test_parse_new_format_mixed():
    """测试新格式：同时有人属性和事件"""
    llm_response = ExtractedInfo(
        structured_query={
            "person_attributes": {
                "职业": "软件工程师",
                "年龄": "25到35岁"
            },
            "behavioral_events": [
                {
                    "event_type": "购买",
                    "attributes": {
                        "购买品类": "数码产品",
                        "购买频率": "每月至少一次"
                    }
                }
            ]
        }
    )

    profiles, events = parse_structured_extraction(llm_response)

    assert len(profiles) == 2
    assert len(events) == 1

    # 验证人属性
    profile_query_texts = [p.query_text for p in profiles]
    assert "职业: 软件工程师" in profile_query_texts
    assert "年龄: 25到35岁" in profile_query_texts

    # 验证事件
    assert events[0].event_description == "购买"
    assert "购买品类: 数码产品" in events[0].event_attributes


def test_parse_mixed_old_and_new_formats():
    """测试混合格式兼容性：behavioral_events + 列表格式属性"""
    llm_response = ExtractedInfo(
        structured_query={
            "person_attributes": ["年龄", "性别"],  # 旧格式列表
            "behavioral_events": [  # 新格式字段名
                {
                    "event_type": "注册",
                    "attributes": ["注册渠道", "注册时间"]  # 旧格式列表
                }
            ]
        }
    )

    profiles, events = parse_structured_extraction(llm_response)

    assert len(profiles) == 2
    assert len(events) == 1
    assert events[0].event_description == "注册"
    assert "注册渠道" in events[0].event_attributes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])