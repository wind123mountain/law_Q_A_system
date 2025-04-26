from dotenv import load_dotenv

load_dotenv()

import asyncio
import logging
import time
from typing import Dict, Optional

from auth import router as auth_router
# from assistant import Assistant
from celery.result import AsyncResult
from config import JWT_SECRET
from database import get_celery_app
from fastapi import FastAPI, HTTPException, Request
from models import chat_conversations
from pydantic import BaseModel
# from search_document.combine_search import CombinedSearch
# from search_document.rerank import BGEReranker
# from tasks import async_retrieval, llm_handle_message
from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# init retriever and reranker
# combined_search_instance = CombinedSearch()
# reranker_instance = BGEReranker(model_name="/home/ivirse/ivirse_all_data/namnt/soict/checkpoint/rerank/bge_v2_part2/checkpoint-225000", use_fp16=True)


app = FastAPI()
app.include_router(auth_router)
celery_app = get_celery_app("celery_app")
# assistant = Assistant(top_n=16, top_k=5)


# define class name
class CompleteRequest(BaseModel):
    bot_id: Optional[str] = "bot_Legal_VN"
    user_id: str
    conversation_id: str
    user_message: str
    sync_request: Optional[bool] = False


class RetrievalRequest(BaseModel):
    query: str
    top_k_search: int = 30
    top_k_rerank: int = 5


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/conversations/{id}")
async def conversations(id: str, request: Request):
    user_id = request.headers.get("user_id")
    db_conversation = chat_conversations.find(
        {"conversation_id": id, "user_id": user_id}
    )
    return {
        "result": [
            {
                "role": "assistant" if not conv["is_request"] else "human",
                "content": str(conv["message"]),
            }
            for conv in db_conversation
        ]
    }


@app.get("/conversations")
async def conversations(request: Request):
    user_id = request.headers.get("user_id")
    conversations = chat_conversations.find({"user_id": user_id}).sort("created_at", -1)

    seen = set()
    unique_conversations = []

    for conv in conversations:
        cid = conv["conversation_id"]
        if conv["is_request"] and cid not in seen:
            seen.add(cid)
            unique_conversations.append((cid, conv["message"]))
    return {
        "result": unique_conversations
    }


@app.post("/retrieval")
async def retrieval(request: RetrievalRequest):
    try:
        query = request.query
        task = celery_app.send_task("tasks.async_retrieval", kwargs={"query": query})
        docs = task.get(timeout=20)
        # contexts = [f"- doc 1: + info:{doc.metadata} \n + content:{doc.page_content}" for doc in docs]
        # context = "\n".join(contexts)

        return {"results": docs}  # reranked_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@app.post("/chat/complete")
async def complete(data: CompleteRequest):
    bot_id = data.bot_id
    user_id = data.user_id
    conversation_id = data.conversation_id
    user_message = data.user_message
    logger.info(
        f"Complete chat from user {user_id} to {bot_id} {conversation_id}: {user_message}"
    )

    if not user_message or not user_id:
        raise HTTPException(
            status_code=400, detail="User id and user message are required"
        )

    # if data.sync_request:
    #     response = llm_handle_message(bot_id, user_id, user_message)
    #     return {"response": str(response)}
    # else:
    task = celery_app.send_task(
        "tasks.llm_handle_message",
        kwargs={
            "bot_id": bot_id,
            "user_id": user_id,
            "conv_id": conversation_id,
            "question": user_message,
        },
    )
    return {"task_id": task.id}


@app.get("/chat/complete_v2/{task_id}")
async def get_response(task_id: str):
    start_time = time.time()
    timeout = 60  # Timeout sau 60 giây
    polling_interval = 0.1  # Thời gian chờ giữa mỗi lần kiểm tra (100ms)

    while True:
        # Lấy trạng thái task từ Celery
        task_result = AsyncResult(task_id)
        task_status = task_result.status

        # Ghi log trạng thái task
        # logger.info(f"Task ID: {task_id}, Status: {task_status}")

        # Nếu task đã hoàn tất, trả về kết quả
        if task_status not in ("PENDING", "STARTED"):
            return {
                "task_id": task_id,
                "task_status": task_status,
                "task_result": task_result.result,
            }

        # Kiểm tra timeout
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            logger.warning(f"Task {task_id} timed out after {timeout} seconds.")
            return {
                "task_id": task_id,
                "task_status": task_status,
                "error_message": "Service timeout, please retry.",
            }

        # Chờ trước khi kiểm tra lại
        await asyncio.sleep(polling_interval)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8002, workers=1, log_level="info")
