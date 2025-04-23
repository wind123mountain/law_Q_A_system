from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import json
import torch
import faiss

from .modeling import BGEM3ForInference



class BGEM3Eval:
    def __init__(self, model_name, tokenizer_name, data_path, candidate_pool = None, 
                 batch_size=8, query_max_length=128, positive_max_length=256, max_neg = 1):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.data_path = data_path
        self.candidate_pool = candidate_pool
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.positive_max_length = positive_max_length
        self.max_neg = max_neg

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)


        model_inference = BGEM3ForInference(model_name=model_name, tokenizer=self.tokenizer,
                                            enable_sub_batch=False, unified_finetuning=False)

        self.model = model_inference.to(self.device).half().eval()

        self.q_dataloader, self.p_dataloader, self.labels = self.load_data()
        self.passage_index = None

    def load_data(self):
        corpus = []
        queries = []
        positives = []
        labels = []

        with open(self.data_path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                line = json.loads(line.strip())
                corpus.extend(line['pos'])
                positives.append(line['pos'])
                if self.candidate_pool is None and 'neg' in line:
                    corpus.extend(line['neg'][:self.max_neg])
                queries.append(line['query'])

        if self.candidate_pool is not None:
            candidate_pool = self.get_corpus()
            corpus.extend(candidate_pool)
        
        corpus = list(set(corpus))

        for q_positives in positives:
            q_labels = [corpus.index(pos) for pos in q_positives]
            labels.append(q_labels)

        # encode queries and positives
        queries_tk = self.tokenizer.batch_encode_plus(queries, padding='max_length', truncation=True, 
                                                   max_length=self.query_max_length, return_tensors="pt")
        passage_tk = self.tokenizer.batch_encode_plus(corpus, padding='max_length', truncation=True,
                                                     max_length=self.positive_max_length, return_tensors="pt")

        q_dataset = TensorDataset(queries_tk['input_ids'], queries_tk['attention_mask'])
        p_dataset = TensorDataset(passage_tk['input_ids'], passage_tk['attention_mask'])

        q_dataloader = DataLoader(q_dataset, batch_size=self.batch_size)
        p_dataloader = DataLoader(p_dataset, batch_size=self.batch_size)

        print("Queries shape:", queries_tk['input_ids'].shape)
        print("Passage shape:", passage_tk['input_ids'].shape)

        self.queries = queries
        self.corpus = corpus

        return q_dataloader, p_dataloader, labels
    
    def get_corpus(self):
        corpus = []
        with open(self.candidate_pool, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = json.loads(line.strip())
                corpus.append(line['text'])
        return corpus
    
    @staticmethod
    def batch_search(index: faiss.Index, query: np.ndarray,
                     topk: int = 200, batch_size: int = 512):
        all_scores, all_inxs = [], []
        for start_index in tqdm(range(0, len(query), batch_size), desc="Batches", disable=len(query) < 256):
            batch_query = query[start_index:start_index + batch_size]
            batch_scores, batch_inxs = index.search(np.asarray(batch_query, dtype=np.float32), k=topk)
            all_scores.extend(batch_scores.tolist())
            all_inxs.extend(batch_inxs.tolist())
        return all_scores, all_inxs

    @staticmethod
    def create_index(embeddings: np.ndarray, use_gpu: bool = False):
        index = faiss.IndexFlatIP(len(embeddings[0]))
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if use_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            index = faiss.index_cpu_to_all_gpus(index, co=co)
        index.add(embeddings)
        return index

    @staticmethod
    def top_k_accuracy_recall(queries_vecs, passages_vecs, lables, k, use_gpu: bool = False):
        index = BGEM3Eval.create_index(passages_vecs, use_gpu=use_gpu)
        _, all_inxs = BGEM3Eval.batch_search(index, queries_vecs, topk=k)

        assert len(all_inxs) == len(queries_vecs)

        sum_recall = 0
        sum_acc = 0

        for i, rel_indexs in enumerate(lables):
            topk_candidates = all_inxs[i][:k]
            n_recall = 0
            for rel_indx in rel_indexs:
                if rel_indx in topk_candidates:
                    n_recall += 1
            if n_recall > 0:
                sum_acc += 1
            sum_recall += n_recall / len(rel_indexs)

        return sum_acc / len(queries_vecs), sum_recall / len(queries_vecs), all_inxs
            
    def inference(self):
        queries_dense_vecs = []
        passages_dense_vecs = []

        with torch.no_grad():
            for batch in tqdm(self.q_dataloader, desc="Queries inferencing"):
                query_input, query_mask = batch
                queries = {'input_ids': query_input.to(self.device),
                           'attention_mask': query_mask.to(self.device)}

                query_outputs = self.model(queries)['dense_vecs']
                
                queries_dense_vecs.append(query_outputs.cpu())

            for batch in tqdm(self.p_dataloader, desc="Passage inferencing"):
                p_input, p_mask = batch
                passages = {'input_ids': p_input.to(self.device), 
                            'attention_mask': p_mask.to(self.device)}

                passages_outputs = self.model(passages)['dense_vecs']
                
                passages_dense_vecs.append(passages_outputs.cpu())

        queries_dense_vecs = torch.cat(queries_dense_vecs, dim=0)
        passages_dense_vecs = torch.cat(passages_dense_vecs, dim=0)

        return queries_dense_vecs, passages_dense_vecs
        


