import logging

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


# Dựa vào history và câu hỏi hiện tại => Viết lại câu hỏi.
def detect_user_intent(history_messages, message):
    logger.info(f"History messages: {history_messages}")
    logger.info(f"messages: {message}")
    # Update documents to prompt
    user_prompt = f"""
    Based on the following conversation history and the latest user query, rewrite the latest query as 
    a standalone question in Vietnamese. The user may switch between different legal topics, such as 
    traffic laws, economic regulations, etc., so ensure the intent of the user is accurately
    identified at the current moment to rephrase the query as precisely as possible. 
    The rewritten question should be clear, complete, and understandable without additional context.
    Please return only the rewritten sentence. Please answer in Vietnamese.

    Chat History:
    {history_messages}

    Original Question: {message}
    """
    gemini_messages = [
        ("system", "You are an amazing virtual assistant"),
        ("human", user_prompt),
    ]
    logger.info(f"Rephrase input messages: {gemini_messages}")
    
    new_question = assistant.llm.invoke(gemini_messages).content

    return new_question

# define agent for process search internet + gen response
def get_legal_agent_anwer(history_messages, question):
    logger.info(f"Call tavily tool search")

    observation = search(question)
    response = assistant.generation.generate_for_search_internet(question, observation, history_messages)

    return response
