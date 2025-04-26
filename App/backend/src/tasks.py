from dotenv import load_dotenv

load_dotenv()

import logging


# from agent import react_agent_handle
from brain_v2 import (
    assistant,
    detect_user_intent,
    get_legal_agent_anwer,
    generate_conversation_text
)
from celery import shared_task
from database import get_celery_app
from models import get_conversation_messages, update_chat_conversation
from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
logging.getLogger("celery.redirected").setLevel(logging.ERROR)

celery_app = get_celery_app("celery_app")
celery_app.autodiscover_tasks()


@shared_task()
def bot_answer_message(history, question):
    # Retrieval
    top_docs = assistant.retrieval.search(question)

    if len(top_docs) > 0:
        logger.info(f"Only call RAG")
        assistant_answer = assistant.generation.generate(question, top_docs)
        return assistant_answer

    logger.warning(f"call Tavily search")
    agent_answer = get_legal_agent_anwer(history, question)

    return agent_answer


@shared_task()
def bot_route_answer_message(history, question):
    history_messages = generate_conversation_text(history)

    new_question = detect_user_intent(history_messages, question)
    logger.warning(f"User intent: {new_question}")

    # detect the route
    result, explanation = assistant.router(history_messages, new_question)
    if not result:
        logger.warning(f"Router to chitchat")
        return explanation
    else:
        return bot_answer_message(history_messages, new_question)

@shared_task()
def llm_handle_message(bot_id, user_id, conv_id, question):
    logger.info("Start handle message")
    # Update chat conversation for new user_question
    conversation_id = update_chat_conversation(bot_id, user_id, conv_id, question, True)
    logger.info("Conversation id: %s", conversation_id)

    # Convert history to list messages
    messages = get_conversation_messages(conversation_id)
    logger.info("Conversation messages: %s", messages)
    history = messages[-5:-1]
    # Use bot route to handle message
    response = bot_route_answer_message(history, question)
    logger.info(f"Chatbot response: {response}")
    # Save response to history
    update_chat_conversation(bot_id, user_id, conversation_id, response, False)
    # Return response
    return {"role": "assistant", "content": response, "chat_id": conversation_id}

