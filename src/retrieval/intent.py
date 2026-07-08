from __future__ import annotations

import json
from pathlib import Path
from typing import Any

INTENT_KEYWORDS = {
    "expense": [
        "报销", "差旅", "出差", "住宿", "酒店", "交通", "打车", "发票", "费用", "招待",
        "接待", "餐费", "超标", "付款", "预付款", "预算", "补贴", "标准", "打款",
    ],
    "leave": [
        "请假", "病假", "年假", "事假", "婚假", "产假", "调休", "考勤", "旷工",
        "迟到", "早退", "病假工资", "长期病假", "超过30天", "30天",
    ],
    "it": [
        "it", "IT", "VPN", "vpn", "WiFi", "wifi", "工单", "故障", "账号", "账户",
        "权限", "设备", "电脑", "软件", "系统", "数据库", "生产环境", "响应", "解决",
    ],
    "office": [
        "办公用品", "领用", "印章", "用印", "盖章", "公章", "办公室", "宠物", "空白合同",
    ],
    "performance": [
        "绩效", "考核", "申诉", "评定", "评分", "评级", "目标", "提交材料",
    ],
    "security": [
        "信息安全", "数据", "客户资料", "商业秘密", "秘密", "网盘", "密码", "分级",
        "泄露", "下载", "外发", "保密", "风险", "个人网盘",
    ],
    "hr": [
        "入职", "转正", "离职", "交接", "外包", "供应商", "驻场", "员工", "人事",
        "账号关闭", "归还", "驻场使用", "外包人员",
    ],
    "contract": [
        "合同", "采购", "招标", "供应商", "20万", "5万", "审核", "紧急采购", "订阅合同",
    ],
}

# Strong phrase rules are applied before the generic keyword score.
STRONG_QUERY_RULES = [
    ("离职", "hr", 4.0),
    ("入职", "hr", 4.0),
    ("转正", "hr", 4.0),
    ("外包人员", "hr", 3.0),
    ("供应商驻场", "hr", 3.0),
    ("VPN", "it", 3.0),
    ("生产数据库", "it", 3.0),
    ("生产环境权限", "it", 2.0),
    ("病假", "leave", 3.0),
    ("绩效", "performance", 3.0),
    ("客户资料", "security", 5.0),
    ("下载客户资料", "security", 5.0),
    ("批量下载", "security", 4.0),
    ("商业秘密", "security", 3.0),
    ("个人网盘", "security", 3.0),
    ("密码", "security", 2.5),
    ("办公用品", "office", 3.0),
    ("印章", "office", 3.0),
    ("盖章", "office", 3.0),
    ("预付款", "expense", 4.0),
    ("合同", "contract", 3.0),
    ("报销", "expense", 3.0),
    ("差旅", "expense", 3.0),
    ("出差", "expense", 3.0),
    ("住宿", "expense", 2.5),
    ("招待", "expense", 2.5),
]

# 跨文档多标签映射：文档包含以下关键词时，除主标签外额外附加的政策类型
# 用于解决 cross_doc 类问题（如 IT 设备文档也涉及 hr 离职流程）
CROSS_DOC_MULTI_TAGS = {
    "离职": {"it", "hr"},
    "设备": {"it", "hr"},
    "权限": {"it", "hr"},
    "账号": {"it", "hr"},
    "归还": {"it", "hr"},
    "服务": {"it", "hr"},
    "数据分级": {"security", "it"},
    "客户资料": {"security", "hr"},
    "商业秘密": {"security", "hr"},
    "合同审批": {"contract", "expense"},
    "预算": {"expense", "contract"},
    "付款审批": {"expense", "contract"},
    "外包": {"hr", "contract"},
    "供应商": {"hr", "contract"},
}

FILE_INTENT_RULES = [
    ("请假", "leave"),
    ("考勤", "leave"),
    ("报销", "expense"),
    ("差旅", "expense"),
    ("接待", "expense"),
    ("会议", "expense"),
    ("IT", "it"),
    ("服务", "it"),
    ("设备", "it"),
    ("权限", "it"),
    ("信息安全", "security"),
    ("数据分级", "security"),
    ("客户资料", "security"),
    ("商业秘密", "security"),
    ("办公用品", "office"),
    ("印章", "office"),
    ("用印", "office"),
    ("绩效", "performance"),
    ("考核", "performance"),
    ("外包", "hr"),
    ("供应商驻场", "hr"),
    ("入职", "hr"),
    ("转正", "hr"),
    ("离职", "hr"),
    ("采购与合同", "contract"),
    ("合同审批", "contract"),
    ("预算", "expense"),
    ("付款审批", "expense"),
]

AMBIGUOUS_KEYWORDS = {
    "采购": {"office": ["办公用品", "领用"], "contract": ["合同", "20万", "招标", "供应商", "紧急采购"]},
    "审批": {"leave": ["请假", "病假"], "expense": ["报销", "费用", "招待", "差旅", "打款"], "contract": ["合同", "采购"]},
    "付款": {"contract": ["合同"], "expense": ["报销", "预算", "预付", "打款"]},
}

FOLLOW_UP_MARKERS = ("那", "这个", "如果", "超过", "多久", "也可以", "怎么办", "以后")


def _normalize(text: str) -> str:
    return (text or "").lower()


def _history_text(chat_history: Any) -> str:
    if not chat_history:
        return ""
    if isinstance(chat_history, str):
        try:
            chat_history = json.loads(chat_history)
        except Exception:
            return chat_history
    if not isinstance(chat_history, list):
        return ""
    parts = []
    for item in chat_history:
        if isinstance(item, dict):
            parts.append(str(item.get("content", "")))
        else:
            parts.append(str(item))
    return " ".join(parts)


def _looks_like_follow_up(query: str) -> bool:
    stripped = (query or "").strip()
    return len(stripped) <= 18 or any(marker in stripped for marker in FOLLOW_UP_MARKERS)


def classify_query_intent(
    query: str,
    chat_history: Any = None,
    prev_intent: str | None = None,
) -> str | None:
    """Classify a user query into a policy type using deterministic rules.

    Args:
        query: The user query to classify.
        chat_history: Previous conversation turns for context.
        prev_intent: Previous round's detected intent (used for follow-up inheritance).
    """
    text = query or ""

    # --- 追问继承上一轮意图 ---
    # 当检测到追问且有 prev_intent 时，默认继承上一轮意图
    # 除非追问中包含强信号表明话题切换（STRONG_QUERY_RULES 权重 >= 3）
    if prev_intent and _looks_like_follow_up(text):
        override_scores: dict[str, float] = {}
        for phrase, intent, weight in STRONG_QUERY_RULES:
            if phrase in text:
                override_scores[intent] = override_scores.get(intent, 0.0) + weight
        if override_scores:
            best_override = max(override_scores.items(), key=lambda x: x[1])
            if best_override[1] >= 3.0:
                # 用户明确提到了新主题，允许覆盖
                return best_override[0]
        # 默认继承上一轮意图
        return prev_intent

    # --- 常规意图分类（原逻辑） ---
    if chat_history and _looks_like_follow_up(text):
        text = f"{_history_text(chat_history)} {text}"
    if not text.strip():
        return None

    normalized = _normalize(text)
    scores: dict[str, float] = {}

    for phrase, intent, weight in STRONG_QUERY_RULES:
        if phrase in text:
            scores[intent] = scores.get(intent, 0.0) + weight

    for intent, keywords in INTENT_KEYWORDS.items():
        for keyword in keywords:
            if _normalize(keyword) in normalized:
                scores[intent] = scores.get(intent, 0.0) + 1.0

    for keyword, intent_map in AMBIGUOUS_KEYWORDS.items():
        if keyword not in text:
            continue
        for intent, context_words in intent_map.items():
            if any(word in text for word in context_words):
                scores[intent] = scores.get(intent, 0.0) + 1.5

    if not scores:
        return None
    return max(scores.items(), key=lambda item: item[1])[0]


def classify_query_intents(
    query: str,
    chat_history: Any = None,
    prev_intent: str | None = None,
) -> list[str]:
    """Return ordered policy intent candidates for cross-document retrieval."""
    primary = classify_query_intent(query, chat_history=chat_history, prev_intent=prev_intent)
    text = query or ""
    intents: list[str] = []

    def add(intent: str | None):
        if intent and intent not in intents:
            intents.append(intent)

    add(primary)

    if "离职" in text and any(word in text for word in ("账号", "账户", "权限", "设备", "归还", "交接")):
        add("hr")
        add("it")
    if "合同" in text and any(word in text for word in ("预付款", "付款", "预算", "采购", "法务", "审核")):
        add("contract")
        add("expense")
    if "客户资料" in text and any(word in text for word in ("离职", "下载", "外发", "泄露", "风险", "网盘")):
        add("security")
        add("hr")
    if any(word in text for word in ("密码", "数据级别", "敏感数据", "客户资料", "商业秘密")):
        add("security")

    return intents


def infer_policy_type(source: str, title: str = "") -> str:
    """Infer primary policy type from file path and document title (single label)."""
    text = f"{Path(source).name} {title}"
    for keyword, intent in FILE_INTENT_RULES:
        if keyword in text:
            return intent
    return "general"


def infer_policy_types(source: str, title: str = "") -> list[str]:
    """Infer ALL matching policy types from file path and document title (multi-label).

    返回所有匹配的政策类型集合，用于 cross_doc 场景多标签匹配。
    例如：'IT服务与设备权限管理制度.md' → ['hr', 'it', 'security']
    """
    text = f"{Path(source).name} {title}"
    types: set[str] = set()

    # 1. 基础 FILE_INTENT_RULES 匹配
    for keyword, intent in FILE_INTENT_RULES:
        if keyword in text:
            types.add(intent)

    # 2. 跨文档多标签补充
    for keyword, extra_types in CROSS_DOC_MULTI_TAGS.items():
        if keyword in text:
            types.update(extra_types)

    return sorted(types) if types else ["general"]


