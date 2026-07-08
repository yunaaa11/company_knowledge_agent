import json
import os
import time
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
CHAT_URL = f"{API_BASE_URL}/api/v1/chat"
HEALTH_URL = f"{API_BASE_URL}/health"
REPORT_DIR = "reports/v2_complex_eval"

EXAMPLE_QUESTIONS = [
    "\u5e74\u5047\u5929\u6570\u548c\u5de5\u9f84\u6709\u4ec0\u4e48\u5173\u7cfb\uff1f",
    "\u75c5\u5047\u9700\u8981\u63d0\u4f9b\u4ec0\u4e48\u6750\u6599\uff1f\u75c5\u5047\u5de5\u8d44\u600e\u4e48\u53d1\uff1f",
    "\u5317\u4eac\u51fa\u5dee\u4f4f\u5bbf\u6807\u51c6\u662f\u591a\u5c11\uff1f",
    "\u529e\u516c\u7528\u54c1\u5e94\u8be5\u600e\u4e48\u7533\u8bf7\uff1f",
    "\u90a3\u5ba1\u6279\u5b8c\u4ee5\u540e\u591a\u4e45\u80fd\u6253\u6b3e\uff1f",
]


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_rewrite", None)
    st.session_state.setdefault("last_sources", [])
    st.session_state.setdefault("last_cache_hit", False)
    st.session_state.setdefault("last_retrieval_cache_hit", False)
    st.session_state.setdefault("last_latency", None)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #172033;
            --muted: #697386;
            --line: rgba(23, 32, 51, 0.10);
            --paper: rgba(255, 255, 255, 0.86);
            --accent: #0f766e;
            --accent-2: #d97706;
            --soft: #f4efe6;
        }
        .stApp {
            background:
                radial-gradient(circle at 18% 12%, rgba(15, 118, 110, .16), transparent 30%),
                radial-gradient(circle at 88% 6%, rgba(217, 119, 6, .13), transparent 26%),
                linear-gradient(135deg, #fbf7ef 0%, #eef5f3 48%, #f8fafc 100%);
            color: var(--ink);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255,255,255,.92), rgba(244,239,230,.92));
            border-right: 1px solid var(--line);
        }
        section[data-testid="stSidebar"] h1 {
            font-size: 1.55rem;
            letter-spacing: -0.04em;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 4.5rem;
            padding-bottom: 5rem;
        }
        .hero {
            padding: 2rem 2.2rem;
            border: 1px solid var(--line);
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(255,255,255,.90), rgba(255,252,247,.74));
            box-shadow: 0 24px 70px rgba(23, 32, 51, .10);
            margin-bottom: 1.2rem;
        }
        .eyebrow {
            color: var(--accent);
            font-weight: 800;
            letter-spacing: .14em;
            text-transform: uppercase;
            font-size: .78rem;
            margin-bottom: .55rem;
        }
        .hero-title {
            font-size: clamp(2.4rem, 6vw, 4.7rem);
            line-height: .96;
            letter-spacing: -0.07em;
            color: var(--ink);
            font-weight: 900;
            margin: 0;
        }
        .hero-subtitle {
            max-width: 760px;
            color: var(--muted);
            font-size: 1.04rem;
            line-height: 1.8;
            margin-top: 1.05rem;
        }
        .status-card {
            padding: 1rem 1.15rem;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: rgba(255, 255, 255, .68);
        }
        .status-ok { color: #047857; font-weight: 800; }
        .status-bad { color: #b45309; font-weight: 800; }
        .hint-card {
            border-left: 5px solid var(--accent-2);
            background: rgba(255, 251, 235, .86);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            color: #5f370e;
            margin: .5rem 0 1rem;
        }
        .source-card {
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: .9rem 1rem;
            background: rgba(255,255,255,.72);
            margin-bottom: .65rem;
        }
        .source-title { font-weight: 800; color: var(--ink); }
        .source-snippet { color: var(--muted); font-size: .92rem; line-height: 1.65; }
        div[data-testid="stChatMessage"] {
            border-radius: 22px;
            background: rgba(255,255,255,.70);
            border: 1px solid rgba(23,32,51,.07);
            box-shadow: 0 10px 34px rgba(23,32,51,.06);
        }
        .stButton > button {
            border-radius: 14px;
            border: 1px solid rgba(15, 118, 110, .18);
            background: rgba(255,255,255,.76);
            transition: all .18s ease;
        }
        .stButton > button:hover {
            border-color: rgba(15, 118, 110, .55);
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(15, 118, 110, .12);
        }
        div[data-testid="stChatInput"] {
            border-radius: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_chat_history() -> List[Dict[str, str]]:
    """将用户输入和最近 8 条对话历史打包成 JSON，发送给后端"""
    return [
        {"role": item["role"], "content": item["content"]}
        for item in st.session_state.messages[-8:]
        if item["role"] in {"user", "assistant"}
    ]


def parse_sse_lines(lines: Iterable[str]) -> Iterable[Dict[str, Any]]:
    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            yield {"done": True}
            continue
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            yield {"error": payload}


def stream_chat(query: str) -> Tuple[str, Dict[str, Any]]:
    payload = {"query": query, "chat_history": build_chat_history()}
    answer = ""
    meta: Dict[str, Any] = {"rewrite_query": None, "sources": [], "cache_hit": False, "retrieval_cache_hit": False, "error": None}

    with requests.post(CHAT_URL, json=payload, stream=True, timeout=120) as response:
        response.raise_for_status()
        placeholder = st.empty()
        for event in parse_sse_lines(response.iter_lines(decode_unicode=True)):
            if event.get("done"):
                break
            if "rewrite_query" in event:
                meta["rewrite_query"] = event["rewrite_query"]
            if "sources" in event:
                meta["sources"] = event["sources"]
            if "cache_hit" in event:
                meta["cache_hit"] = bool(event["cache_hit"])
            if "retrieval_cache_hit" in event:
                meta["retrieval_cache_hit"] = bool(event["retrieval_cache_hit"])
            if "error" in event:
                meta["error"] = event["error"]
                answer = event["error"]
                placeholder.warning(answer)
                continue
            chunk = event.get("answer_chunk")
            if chunk:
                answer += chunk
                placeholder.markdown(answer)
        placeholder.markdown(answer or "\u672a\u6536\u5230\u56de\u7b54\u3002")
    return answer, meta


def api_health() -> bool:
    try:
        response = requests.get(HEALTH_URL, timeout=3)
        return response.ok
    except requests.RequestException:
        return False


def render_sidebar(healthy: bool) -> None:
    with st.sidebar:
        st.title("\u4f01\u4e1a\u77e5\u8bc6\u5e93 RAG")
        st.markdown(
            f"""
            <div class="status-card">
              <div>API \u72b6\u6001</div>
              <div class="{'status-ok' if healthy else 'status-bad'}">{'\u6b63\u5e38' if healthy else '\u4e0d\u53ef\u7528'}</div>
              <small>{API_BASE_URL}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        col1.metric("\u8017\u65f6", f"{st.session_state.last_latency:.2f}s" if st.session_state.last_latency is not None else "-")
        col2.metric("\u7f13\u5b58", "\u547d\u4e2d" if (st.session_state.last_cache_hit or st.session_state.last_retrieval_cache_hit) else "\u672a\u547d\u4e2d")

        st.divider()
        st.subheader("\u793a\u4f8b\u95ee\u9898")
        for question in EXAMPLE_QUESTIONS:
            if st.button(question, use_container_width=True):
                st.session_state.pending_question = question
        #会话状态管理：利用 st.session_state 持久化消息历史、缓存标志、延迟时间。
        st.divider()
        if st.button("\u6e05\u7a7a\u5bf9\u8bdd", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_rewrite = None
            st.session_state.last_sources = []
            st.session_state.last_cache_hit = False
            st.session_state.last_retrieval_cache_hit = False
            st.session_state.last_latency = None
            st.rerun()


def render_header(healthy: bool) -> None:
    badge = "\u540e\u7aef\u5df2\u8fde\u63a5" if healthy else "\u7b49\u5f85\u540e\u7aef\u542f\u52a8"
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">Enterprise Agentic RAG</div>
          <h1 class="hero-title">\u884c\u653f\u5236\u5ea6\u95ee\u7b54\uff0c<br/>\u6709\u636e\u53ef\u67e5</h1>
          <div class="hero-subtitle">
            \u9762\u5411\u516c\u53f8\u5236\u5ea6\u7684\u77e5\u8bc6\u5e93 Demo\uff1a\u652f\u6301\u67e5\u8be2\u6539\u5199\u3001\u6df7\u5408\u68c0\u7d22\u3001\u91cd\u6392\u3001\u53cd\u601d\u91cd\u8bd5\u3001Redis \u7f13\u5b58\u548c\u6765\u6e90\u8ffd\u8e2a\u3002
          </div>
          <p><span class="{'status-ok' if healthy else 'status-bad'}">{badge}</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_backend_hint() -> None:
    st.markdown(
        """
        <div class="hint-card">
          <b>\u540e\u7aef\u670d\u52a1\u8fd8\u6ca1\u8fde\u4e0a\u3002</b><br/>
          \u8fd9\u4e2a\u9875\u9762\u53ea\u662f\u5c55\u793a\u5c42\uff0c\u8fd8\u9700\u8981\u53e6\u5916\u542f\u52a8 FastAPI \u95ee\u7b54\u670d\u52a1\u3002\u5982\u679c\u4f60\u662f\u672c\u5730\u8fd0\u884c\uff0c\u8bf7\u518d\u6253\u5f00\u4e00\u4e2a\u7ec8\u7aef\u542f\u52a8\uff1a<br/>
          <code>uvicorn src.api.app:app --host 0.0.0.0 --port 8000</code><br/>
          \u6216\u8005\u76f4\u63a5\u7528 Docker Compose \u540c\u65f6\u542f\u52a8\u524d\u540e\u7aef\u3002
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_debug_panel() -> None:
    """折叠在底部，展示改写后的查询和召回文档详情（包含文件名、相关性分数、片段）"""
    with st.expander("\u68c0\u7d22\u8c03\u8bd5\u4fe1\u606f", expanded=False):
        rewrite = st.session_state.last_rewrite
        if rewrite:
            st.markdown("**\u6539\u5199\u540e\u7684\u67e5\u8be2**")
            if isinstance(rewrite, list):
                for item in rewrite:
                    st.code(item, language="text")
            else:
                st.code(str(rewrite), language="text")
        else:
            st.caption("\u8fd8\u6ca1\u6709\u67e5\u8be2\u6539\u5199\u4fe1\u606f\u3002")

        sources = st.session_state.last_sources or []
        st.markdown("**\u5f15\u7528\u6765\u6e90**")
        if not sources:
            st.caption("\u5f53\u524d\u56de\u7b54\u672a\u8fd4\u56de\u6765\u6e90\u4fe1\u606f\u3002")
            return
        for source in sources:
            score = source.get("relevance_score", 0)
            title = source.get("source") or "\u672a\u77e5\u6765\u6e90"
            cache_label = "\u6765\u81ea\u7f13\u5b58" if source.get("from_cache") else "\u5b9e\u65f6\u68c0\u7d22"
            st.markdown(
                f"""
                <div class="source-card">
                  <div class="source-title">#{source.get('rank', '-')} {title}</div>
                  <div class="source-snippet">score={score:.4f} - {cache_label}</div>
                  <div class="source-snippet">{source.get('snippet', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("\u67e5\u770b\u5b8c\u6574\u7247\u6bb5", expanded=False):
                st.write(source.get("page_content") or source.get("snippet", ""))



def read_report_csv(name: str) -> pd.DataFrame:
    path = os.path.join(REPORT_DIR, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def render_metric_card(label: str, value, help_text: str = "") -> None:
    st.metric(label, value, help=help_text or None)


def render_eval_dashboard() -> None:
    """评估看板"""
    st.markdown("### 复杂 RAG 评估看板")
    st.caption("读取 reports/v2_complex_eval/ 下的评估结果，展示检索召回、意图命中、无答案拒答和 ablation 对比。")

    summary = read_report_csv("eval_summary.csv")
    category = read_report_csv("category_summary.csv")
    ablation = read_report_csv("ablation_summary.csv")
    errors = read_report_csv("error_cases.csv")
    report = read_report_csv("eval_report.csv")

    if summary.empty and category.empty and ablation.empty and report.empty:
        st.info("还没有生成评估报告。可以先运行下面两条命令生成基础诊断结果。")
        st.code("python test/run_eval_complex.py --skip-ragas", language="powershell")
        st.code("python test/run_ablation_complex.py --limit 10", language="powershell")
        return

    if not summary.empty:
        row = summary.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("样本数", int(row.get("sample_count", 0)))
        c2.metric("预期制度命中率", f"{row.get('expected_policy_hit_rate', 0):.2%}")
        c3.metric("Top3 来源命中率", f"{row.get('top3_source_hit_rate', 0):.2%}")
        c4.metric("平均耗时", f"{row.get('avg_latency_seconds', 0):.2f}s")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("意图命中率", f"{row.get('intent_accuracy', 0):.2%}")
        c6.metric("无答案拒答准确率", f"{row.get('no_answer_rejection_accuracy', 0):.2%}")
        c7.metric("平均召回文档数", f"{row.get('avg_retrieval_count', 0):.2f}")
        c8.metric("Strict Score", f"{row.get('avg_strict_score', 0):.3f}")

    if not category.empty:
        st.markdown("#### 分类表现")
        chart_df = category.set_index("category")[[col for col in ["expected_policy_hit_rate", "top3_source_hit_rate", "avg_strict_score"] if col in category.columns]]
        if not chart_df.empty:
            st.bar_chart(chart_df)
        st.dataframe(category, use_container_width=True)

    if not ablation.empty:
        st.markdown("#### Ablation 对比")
        display_cols = [col for col in ["variant", "expected_policy_hit_rate", "top3_source_hit_rate", "intent_accuracy", "avg_latency_seconds", "avg_retrieval_count", "avg_strict_score"] if col in ablation.columns]
        st.dataframe(ablation[display_cols], use_container_width=True)
        if "variant" in ablation.columns:
            chart_cols = [col for col in ["expected_policy_hit_rate", "top3_source_hit_rate", "avg_strict_score"] if col in ablation.columns]
            if chart_cols:
                st.bar_chart(ablation.set_index("variant")[chart_cols])

    if not report.empty:
        st.markdown("#### 检索明细")
        filters = st.multiselect("按问题类型筛选", sorted(report["category"].dropna().unique().tolist()) if "category" in report.columns else [])
        view = report
        if filters:
            view = view[view["category"].isin(filters)]
        show_cols = [
            col for col in [
                "id",
                "category",
                "question",
                "intent",
                "expected_policy_type",
                "hit_expected_policy",
                "top1_source_hit",
                "top3_source_hit",
                "original_query_doc_count",
                "rewrite_query_doc_count",
                "hyde_query_doc_count",
                "retrieval_cache_hit",
                "retrieved_sources",
                "answer",
            ] if col in view.columns
        ]
        st.dataframe(view[show_cols], use_container_width=True, height=360)

    if not errors.empty:
        st.markdown("#### 需要重点排查的样本")
        show_cols = [col for col in ["id", "category", "question", "expected_policy_type", "retrieved_policy_types", "retrieved_sources", "answer"] if col in errors.columns]
        st.dataframe(errors[show_cols], use_container_width=True, height=260)


def main() -> None:
    st.set_page_config(page_title="\u4f01\u4e1a\u884c\u653f\u77e5\u8bc6\u5e93\u95ee\u7b54", page_icon="\U0001F4DA", layout="wide")
    init_state()
    inject_css()
    healthy = api_health()
    render_sidebar(healthy)
    tab_chat, tab_eval = st.tabs(["问答演示", "评估看板"])

    with tab_chat:
        render_header(healthy)

        if not healthy:
            render_backend_hint()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        pending = st.session_state.pop("pending_question", None)
        query = pending or st.chat_input("请输入关于公司制度、报销、请假、IT 支持等问题")

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                start = time.perf_counter()
                try:
                    answer, meta = stream_chat(query)
                except requests.RequestException as exc:
                    answer = f"后端请求失败：{exc}"
                    meta = {"rewrite_query": None, "sources": [], "cache_hit": False, "error": answer}
                    st.warning(answer)
                st.session_state.last_latency = time.perf_counter() - start

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.session_state.last_rewrite = meta.get("rewrite_query")
            st.session_state.last_sources = meta.get("sources", [])
            st.session_state.last_cache_hit = bool(meta.get("cache_hit"))
            st.session_state.last_retrieval_cache_hit = bool(meta.get("retrieval_cache_hit"))
            st.rerun()

        render_debug_panel()

    with tab_eval:
        render_eval_dashboard()


if __name__ == "__main__":
    main()
