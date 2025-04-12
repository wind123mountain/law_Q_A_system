import logging
import logging.config
from fastapi import FastAPI
from pydantic import BaseModel
from assistant import Assistant

logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "default",
                "class": "logging.StreamHandler",
            },
        },
        "loggers": {
            "": {
                "handlers": ["default"],
                "level": "INFO",
            },
        },
    })

logger = logging.getLogger(__name__)

app = FastAPI()

assistant = Assistant(top_n=16, top_k=5)

# define class name
class QuestionRequest(BaseModel):
    question: str


@app.get("/")
async def root():
    return {"message": "Welcome to the law chatbot"}


@app.post("/ask")
def ask(data: QuestionRequest):
    anwser = assistant.ask(data.question)
    return {"response": anwser}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=1, log_level="info")

