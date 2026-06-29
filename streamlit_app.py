import json
import os
import time
from typing import Any, Dict, Iterable, List, Tuple

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
CHAT_URL = f"{API_BASE_URL}/api/v1/chat"
HEALTH_URL = f"{API_BASE_URL}/health"

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
    meta: Dict[str, Any] = {"rewrite_query": None, "sources": [], "cache_hit": False, "error": None}

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
        col2.metric("\u7f13\u5b58", "\u547d\u4e2d" if st.session_state.last_cache_hit else "\u672a\u547d\u4e2d")

        st.divider()
        st.subheader("\u793a\u4f8b\u95ee\u9898")
        for question in EXAMPLE_QUESTIONS:
            if st.button(question, use_container_width=True):
                st.session_state.pending_question = question

        st.divider()
        if st.button("\u6e05\u7a7a\u5bf9\u8bdd", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_rewrite = None
            st.session_state.last_sources = []
            st.session_state.last_cache_hit = False
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
            st.markdown(
                f"""
                <div class="source-card">
                  <div class="source-title">{source.get('rank', '-')}. {title} - score={score:.4f}</div>
                  <div class="source-snippet">{source.get('snippet', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    st.set_page_config(page_title="\u4f01\u4e1a\u884c\u653f\u77e5\u8bc6\u5e93\u95ee\u7b54", page_icon="\U0001F4DA", layout="wide")
    init_state()
    inject_css()
    healthy = api_health()
    render_sidebar(healthy)
    render_header(healthy)

    if not healthy:
        render_backend_hint()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    pending = st.session_state.pop("pending_question", None)
    query = pending or st.chat_input("\u8bf7\u8f93\u5165\u5173\u4e8e\u516c\u53f8\u5236\u5ea6\u3001\u62a5\u9500\u3001\u8bf7\u5047\u3001IT \u652f\u6301\u7b49\u95ee\u9898")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            start = time.perf_counter()
            try:
                answer, meta = stream_chat(query)
            except requests.RequestException as exc:
                answer = f"\u8bf7\u6c42\u540e\u7aef\u5931\u8d25\uff1a{exc}"
                meta = {"rewrite_query": None, "sources": [], "cache_hit": False, "error": answer}
                st.warning(answer)
            st.session_state.last_latency = time.perf_counter() - start

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.last_rewrite = meta.get("rewrite_query")
        st.session_state.last_sources = meta.get("sources", [])
        st.session_state.last_cache_hit = bool(meta.get("cache_hit"))
        st.rerun()

    render_debug_panel()


if __name__ == "__main__":
    main()
