from typing import List
from transformers import AutoTokenizer
import torch

from langchain_core.embeddings import Embeddings
from .BGE_M3.modeling import BGEM3ForInference

from openai import OpenAI


class M3Embeddings(Embeddings):
    def __init__(self, model_name, batch_size=8, query_max_length=128, positive_max_length=256):

        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.positive_max_length = positive_max_length

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3', use_fast=True)

        model_inference = BGEM3ForInference(model_name=model_name, tokenizer=self.tokenizer,
                                            enable_sub_batch=False, unified_finetuning=False)

        self.model = model_inference.to(self.device).half().eval()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [t.replace("\n", " ") for t in texts]

        dense_vecs = self.endcode(texts, self.positive_max_length)
        
        return dense_vecs.to_list()

    def embed_query(self, text: str) -> List[float]:
        text = text.replace("\n", " ")

        dense_vec = self.endcode([text], self.query_max_length)[0]

        return dense_vec.to_list()
    
    def endcode(self, texts, max_length):
        dense_vecs = []

        with torch.no_grad():
            for start_batch in range(0, len(texts), self.batch_size):
                inputs = self.tokenizer.batch_encode_plus(texts[start_batch:start_batch + self.batch_size], 
                                                          padding='max_length', truncation=True, 
                                                          max_length=max_length, return_tensors="pt")
                
                inputs = {key: val.to(self.device) for key, val in inputs.items()}

                query_outputs = self.model(inputs)['dense_vecs']
                
                dense_vecs.append(query_outputs.cpu())

        return torch.cat(dense_vecs, dim=0)
    
class M3EmbeddingsAPI(Embeddings):
    def __init__(self, nvidia_api_key, batch_size=8):
        client = OpenAI(api_key=nvidia_api_key, base_url="https://integrate.api.nvidia.com/v1")
        self.embeddings = client.embeddings

        self.batch_size = batch_size


    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = [t.replace("\n", " ") for t in texts]
        return self.endcode(texts)

    def embed_query(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        return self.endcode([text])[0]
    
    def endcode(self, texts):
        dense_vecs = []

        for start_batch in range(0, len(texts), self.batch_size): 
            response = self.embeddings.create(input=texts[start_batch:start_batch + self.batch_size],
                                              model="baai/bge-m3",
                                              encoding_format="float",
                                              extra_body={"truncate": "NONE"})

            for data in response.data:
                dense_vecs.append(data.embedding)

        return dense_vecs
    
