import uuid
from typing import List, Optional, Tuple

from langchain.retrievers import ParentDocumentRetriever
from langchain.schema import Document
from langchain.retrievers.multi_vector import SearchType
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from qdrant_client import models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_vector import SearchType
from langchain.schema import Document
from utils.mongodb_store import MongoDBDocstore
from utils.vector_store import init_vector_store

class CustomParentDocumentRetriever(ParentDocumentRetriever):
    batch_size: int = 32

    def _split_docs_for_adding(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        add_to_docstore: bool = True,
    ) -> Tuple[List[Document], List[Tuple[str, Document]]]:
        if self.parent_splitter is not None:
            documents = self.parent_splitter.split_documents(documents)
        if ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in documents]
            if not add_to_docstore:
                raise ValueError(
                    "If ids are not passed in, `add_to_docstore` MUST be True"
                )
        else:
            if len(documents) != len(ids):
                raise ValueError(
                    "Got uneven list of documents and ids. "
                    "If `ids` is provided, should be same length as `documents`."
                )
            doc_ids = ids

        docs = []
        full_docs = []
        for i, doc in enumerate(documents):
            _id = doc_ids[i]
            sub_docs = self.child_splitter.split_documents([doc])
            if self.child_metadata_fields is not None:
                for _doc in sub_docs:
                    _doc.metadata = {
                        k: _doc.metadata[k] for k in self.child_metadata_fields
                    }
            for _doc in sub_docs:
                _doc.metadata[self.id_key] = _id
            docs.extend(sub_docs)
            full_docs.append((_id, doc))

        return docs, full_docs

    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        if self.search_type == SearchType.mmr:
            chunks = sub_docs = self.vectorstore.max_marginal_relevance_search(
                query, **self.search_kwargs
            )
        elif self.search_type == SearchType.similarity_score_threshold:
            chunks = sub_docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            sub_docs = [sub_doc for sub_doc, _ in sub_docs_and_similarities]
        else:
            chunks = sub_docs = self.vectorstore.similarity_search(query, **self.search_kwargs)

        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = self.docstore.mget(ids)
        return docs, chunks

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:

        if self.search_type == SearchType.mmr:
            chunks = sub_docs = await self.vectorstore.amax_marginal_relevance_search(
                query, **self.search_kwargs
            )
        elif self.search_type == SearchType.similarity_score_threshold:
            chunks = sub_docs_and_similarities = (
                await self.vectorstore.asimilarity_search_with_relevance_scores(
                    query, **self.search_kwargs
                )
            )
            sub_docs = [sub_doc for sub_doc, _ in sub_docs_and_similarities]
        else:
            chunks = sub_docs = await self.vectorstore.asimilarity_search(
                query, **self.search_kwargs
            )

        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = await self.docstore.amget(ids)
        return docs, chunks

def init_retriever():
    with open("mongo_read_pass.txt", "r", encoding="utf-8") as file:
        mongodb_pass = file.read()

    mongodb_url = f"mongodb+srv://reader:{mongodb_pass}@cluster.ovvrd.mongodb.net/?retryWrites=true&w=majority&appName=cluster"
    mongodb_doc_store = MongoDBDocstore(mongodb_url)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0, 
                                                    separators=[r'\n (?<!“)Điều \d+\. '], 
                                                    is_separator_regex=True)

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0, 
                                                    separators=[r'\n (?<!“)\d+\. ', r'\n (?<!“)[a-z]+\) '], 
                                                    is_separator_regex=True)
    
    vector_store = init_vector_store()

    retriver = CustomParentDocumentRetriever(vectorstore=vector_store, 
                                            docstore=mongodb_doc_store,
                                            child_splitter=child_splitter,
                                            parent_splitter=parent_splitter,
                                            search_kwargs={'k': 24, 
                                                            'hybrid_fusion': models.FusionQuery(fusion=models.Fusion.RRF)},
                                            search_type=SearchType.similarity_score_threshold,
                                            batch_size=32)
    
    return retriver