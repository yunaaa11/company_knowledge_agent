from __future__ import annotations

from pathlib import Path
#用于匹配用户口语化的短文本
INTENT_KEYWORDS = {
    "expense": ["??", "??", "??", "??", "??", "??", "??", "??", "??", "????", "??"],
    "leave": ["??", "??", "??", "??", "??", "??", "??", "??", "??"],
    "it": ["IT", "??", "??", "VPN", "??", "WiFi", "??", "??", "??", "??", "??"],
    "office": ["????", "??", "??", "???", "??", "??", "??", "??", "??"],
    "performance": ["??", "??", "??", "??", "???", "??", "??"],
    "security": ["????", "??", "??", "????", "????", "??", "??", "???"],
    "hr": ["??", "??", "??", "????", "??", "?????"],
    "contract": ["??", "??", "???", "??", "??", "??"],
}
#用于匹配正式的文件名（如《差旅费管理办法》映射为 expense
FILE_INTENT_RULES = [
    ("??", "expense"),
    ("??", "expense"),
    ("??", "expense"),
    ("??", "expense"),
    ("??", "leave"),
    ("??", "leave"),
    ("IT", "it"),
    ("??", "it"),
    ("????", "security"),
    ("??", "security"),
    ("????", "security"),
    ("????", "security"),
    ("????", "office"),
    ("??", "office"),
    ("??", "office"),
    ("??", "performance"),
    ("??", "hr"),
    ("??", "hr"),
    ("??", "hr"),
    ("??", "contract"),
    ("??", "contract"),
]


def classify_query_intent(query: str) -> str | None:
    """查询意图分类"""
    #词频累加，命中关键词越多，该意图得分越高，最后取最高分。虽然没有 TF-IDF 那么精确，但胜在零延迟、不依赖外部 API
    #关键词匹配，判断用户问的是“报销（expense）”、“请假（leave）”、“IT 支持”、“绩效（performance）”还是“合同（contract）”等
    query_lower = (query or "").lower()
    scores: dict[str, int] = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword.lower() in query_lower:
                score += 1
        if score:
            scores[intent] = score
    if not scores:
        return None
    return max(scores.items(), key=lambda item: item[1])[0]


def infer_policy_type(source: str, title: str = "") -> str:
    """文档类型推断"""
    #根据文档的文件名或 Markdown 的一级标题，判断这份文档属于哪一类政策（如上）。如果匹配不到，默认返回 "general"
    #将 文件名 + 标题 拼接后再匹配，极大提高了命中率（因为文件名可能叫 001.pdf，但标题是《请假制度》
    text = f"{Path(source).name} {title}"
    for keyword, intent in FILE_INTENT_RULES:
        if keyword in text:
            return intent
    return "general"
