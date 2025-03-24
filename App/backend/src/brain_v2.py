import json
import logging
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from tavily_search import functions_info, search

logger = logging.getLogger(__name__)

def get_openai_client():
    return ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)



client = get_openai_client()

def openai_chat_complete(messages=()):
    ai_msg = client.invoke(messages)
    return ai_msg.content


def gen_doc_prompt(docs):
    """
    """
    doc_prompt = "Dưới đây là tài liệu về các điều luật liên quan đến câu hỏi của người dùng:"
    for i,doc in enumerate(docs):
        doc_prompt += f"{i}. {doc} \n"
    doc_prompt += "Kết thúc phần các tài liệu liên quan."

    return doc_prompt


def generate_conversation_text(conversations):
    conversation_text = ""
    for conversation in conversations:
        logger.info("Generate conversation: {}".format(conversation))
        role = conversation.get("role", "user")
        content = conversation.get("content", "")
        conversation_text += f"{role}: {content}\n"
    return conversation_text

# Dựa vào history và câu hỏi hiện tại => Viết lại câu hỏi.
def detect_user_intent(history, message):
    # Convert history to list messages
    history_messages = generate_conversation_text(history)
    logger.info(f"History messages: {history_messages}")
    # Update documents to prompt
    user_prompt = f"""
    Based on the following conversation history and the latest user query, rewrite the latest query as 
    a standalone question in Vietnamese. The user may switch between different legal topics, such as 
    traffic laws, economic regulations, etc., so ensure the intent of the user is accurately
    identified at the current moment to rephrase the query as precisely as possible. 
    The rewritten question should be clear, complete, and understandable without additional context.

    Chat History:
    {history_messages}

    Original Question: {message}

    Answer:
    """
    openai_messages = [
        {"role": "system", "content": "You are an amazing virtual assistant"},
        {"role": "user", "content": user_prompt}
    ]
    logger.info(f"Rephrase input messages: {openai_messages}")
    # call openai
    return openai_chat_complete(openai_messages)


# Classify xem câu query thuộc loại nào?
def detect_route(history, message):
    logger.info(f"Detect route on history messages: {history}")
    # Update documents to prompt
    user_prompt = f"""
    Given the following chat history and the user's latest message. Hãy phân loại xu hướng mong muốn trong tin nhắn của user là loại nào trong 2 loại sau. \n
    1. Mong muốn hỏi các thông tin liên quan đến luật pháp tại Việt Nam, các tình huống thực tế gặp phải liên quan đến luật 
    Ví dụ: -  Nếu xe máy không đội mũ bảo hiểm thì bị phạt bao nhiêu tiền?
           -  Nếu ô tô đi ngược chiều thì bị phạt thế nào?
           -  Lập kế hoạch đấu giá quyền khai thác khoáng sản dựa trên các căn cứ nào ?
           -  Mục đích của bảo hiểm tiền gửi là gì ?
    => Loại này có nhãn là : "legal"
    2. Mong muốn chitchat thông thường.
    Ví dụ:  - Hi, xin chào, tôi cần bạn hỗ trợ,....
            - Chủ tịch nước Việt Nam là ai ,....
    => Loại này có nhãn là : "chitchat"
    Provide only the classification label as your response.

    Chat History:
    {history}

    Latest User Message:
    {message}

    Classification (choose either "chitchat" or "legal"):
    """
    openai_messages = [
        ("system", "You are a highly intelligent assistant that helps classify customer queries"),
        ("human", user_prompt)
    ]
    logger.info(f"Route output: {openai_messages}")
    # call openai
    return openai_chat_complete(openai_messages)

# define agent for process search internet + gen response
def get_legal_agent_anwer(messages):
    logger.info(f"Call tavily tool search")
    
    observation = search(messages[-1][1])
    full_message = messages + [(
            "human",
            observation
    )]
    response = openai_chat_complete(full_message)
    return response



if __name__ == "__main__":
    history = [{"role": "system", "content": "You are an amazing virtual assistant"}]
    message = "Hello"
    output_detect = detect_route(history, message)
    print(output_detect)