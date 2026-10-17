import json
import logging
import re

from assistant import Assistant

# from langchain_google_genai import ChatGoogleGenerativeAI
from search_document_v2.tavily_search import search

logger = logging.getLogger(__name__)
TOP_K_HISTORY = 10


assistant = Assistant(top_n=16, top_k=5)



def generate_conversation_text(conversations):
    conversation_text = ""
    for conversation in conversations[:TOP_K_HISTORY]:
        logger.info("Generate conversation: {}".format(conversation))
        if len(conversation) == 1:
            conversation_text += f"{content}\n"
            continue
        elif len(conversation) == 2:
            role, content = conversation
            conversation_text += f"{role}: {content}\n"
    return conversation_text


# Original rewrite instruction (graph mode off)
PLAIN_REWRITE_INSTRUCTIONS = "Please return only the rewritten sentence. Please answer in Vietnamese."

# Graph mode: the same call also extracts LightRAG low / high-level keywords
KEYWORD_INSTRUCTIONS = """
    Also extract keywords from the rewritten question, in Vietnamese:
    - low_level: specific legal entities it mentions (e.g. "hội đồng thành viên", "giấy chứng nhận đăng ký doanh nghiệp")
    - high_level: the themes or kinds of relation it asks about (e.g. "bổ nhiệm, miễn nhiệm", "chuyển đổi loại hình doanh nghiệp")

    Return only JSON: {"question": "...", "low_level": ["..."], "high_level": ["..."]}"""


# Dựa vào history và câu hỏi hiện tại => Viết lại câu hỏi.
def detect_user_intent(history_messages, message, extract_keywords=True):
    logger.info(f"History messages: {history_messages}")
    logger.info(f"messages: {message}")
    # Update documents to prompt
    user_prompt = f"""
    Based on the following conversation history and the latest user query, rewrite the latest query as 
    a standalone question in Vietnamese. The user may switch between different legal topics, such as 
    traffic laws, economic regulations, etc., so ensure the intent of the user is accurately
    identified at the current moment to rephrase the query as precisely as possible. 
    The rewritten question should be clear, complete, and understandable without additional context.
    {KEYWORD_INSTRUCTIONS if extract_keywords else PLAIN_REWRITE_INSTRUCTIONS}

    Chat History:
    {history_messages}

    Original Question: {message}
    """
    gemini_messages = [
        ("system", "You are an amazing virtual assistant"),
        ("human", user_prompt),
    ]
    logger.info(f"Rephrase input messages: {gemini_messages}")

    content = assistant.llm.invoke(gemini_messages).content
    if isinstance(content, list):  # some models return content as a list of parts
        content = "".join(part if isinstance(part, str) else part.get("text", "") for part in content)
    if not extract_keywords:
        return content.strip(), None
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())

    # Rewrite + LightRAG keywords in one call; fall back to the plain rewritten text
    # (graph search then uses the whole question) if the model did not return valid JSON
    try:
        data = json.loads(raw)
        question = str(data.get("question") or "").strip() or message
        keywords = {
            "low_level": [str(k).strip() for k in data.get("low_level") or [] if str(k).strip()],
            "high_level": [str(k).strip() for k in data.get("high_level") or [] if str(k).strip()],
        }
        return question, keywords
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Rephrase did not return JSON; continuing without graph keywords")
        return raw or message, None

# define agent for process search internet + gen response
def get_legal_agent_anwer(history_messages, question):
    logger.info(f"Call tavily tool search")

    observation = search(question)
    response = assistant.generation.generate_for_search_internet(question, observation, history_messages)

    return response
