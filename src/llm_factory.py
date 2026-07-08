from config import Config


def _load_chat_openai():
    try:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI
    except ImportError:
        from langchain_community.chat_models import ChatOpenAI

        return ChatOpenAI


def create_chat_openai(temperature=0):
    ChatOpenAI = _load_chat_openai()
    return ChatOpenAI(
        model=Config.LLM_MODEL,
        api_key=Config.OPENAI_API_KEY,
        base_url=Config.OPENAI_BASE_URL,
        temperature=temperature,
    )
