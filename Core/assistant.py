import os

import numpy as np
import torch
from FlagEmbedding import FlagReranker
from FlagEmbedding.abc.inference import AbsReranker
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client import models
from utils.custom_retriever import init_retriever

os.environ["GOOGLE_API_KEY"] = "AIzaSyDpQ3AgD14XuY6kFvRvUrh7jUsjJEhVpv4"

class Retrieval:
    def __init__(self, top_n=20, top_k=10):
        self.retriver = init_retriever()
        condition = models.FieldCondition(key="metadata.status", 
                                          match=models.MatchValue(value="Hết hiệu lực toàn bộ"))
        self.retriver.search_kwargs['filter'] = models.Filter(must_not=[condition])
        self.retriver.search_kwargs['k'] = top_n

        self.top_k = top_k

        device = "cuda" if torch.cuda.is_available() else "cpu"
        reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, devices=device, batch_size=4)
        reranker.compute_score(["mode to gpu", "cuda"])
        self.reranker = reranker
        
    def search(self, query):
        parent_docs, candidates =  self.retriver.invoke(query)

        if len(candidates) == 0:
            return []
        
        inputs = [[query, candidate[0].page_content] for candidate in candidates]
        scores = self.reranker.compute_score(inputs, batch_size=2)
        sorted_indices = np.argsort(scores)[::-1]

        top_k_docs = [parent_docs[candidates[i][0].metadata['doc_id']] for i in sorted_indices[:self.top_k]]
        
        return top_k_docs
    

class Generation:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, 
                                          max_tokens=None, timeout=None, max_retries=2)
        
    def check_question(self, question):
        messages = [
            ("human", 
             f"Câu hỏi:\"{question}\" có phải là câu hỏi về luật doanh nghiệp Việt Nam không? \
                Chỉ trả lời \"yes\" hoặc \"no\"."
            ),
        ]

        messages_explanation = [
            (
                "system",
                "You are an expert lawyer in Vietnam, tasked with answering only asked \
                    questions from customers about Vietnamese business Law. \
                        You are asked one question unrelated to Vietnamese business Law. \
                            Please answer in Vietnamese so that the customer understands."
            ),
            ("human", f"{question}"),
        ]

        answer = self.llm.invoke(messages).content
        print('check_question: ', answer)
        result = answer.lower() == "yes"

        if not result:
            explanation = self.llm.invoke(messages_explanation).content
        else:
            explanation = None
        
        return result, explanation

    def generate(self, question, docs:Document):

        context = [f"- doc 1: + info:{doc.metadata} \n + content:{doc.page_content}" for doc in docs]
        context = "\n".join(context)

        messages = [
            (
                "system",
                f"You are an expert lawyer in Vietnam, \
                    tasked with answering only asked questions from customers about Vietnamese \
                        business Law based on the given information. Please use, gather, and deduce based on the \
                            knowledge in the following information to answer the user’s question. \
                                Please respond accurately, fully, clearly citing the law. \n Relevant legal information: \n {context}",
            ),
            ("human", f"{question}"),
        ]

        ai_msg = self.llm.invoke(messages)

        return ai_msg.content
    
class Assistant:
    def __init__(self, top_n=20, top_k=10):
        self.retrieval = Retrieval(top_n, top_k)
        self.generation = Generation()
    
    def ask(self, question):
        result, explanation = self.generation.check_question(question)
        if not result:
            return explanation
        
        docs = self.retrieval.search(question)
        answer = self.generation.generate(question, docs)

        return answer
    
def init_assistant():
    assistant = Assistant(top_n=16, top_k=5)
    return assistant