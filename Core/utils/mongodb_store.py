import pymongo
from langchain_core.stores import BaseStore
from langchain.schema import Document


class MongoDBDocstore(BaseStore):
    """Docstore sử dụng MongoDB để lưu Parent Documents."""
    def __init__(self, mongodb_url, db="law"):
        self.client = pymongo.MongoClient(mongodb_url, server_api=pymongo.server_api.ServerApi('1'))
        try:
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")

            self.law_db = self.client[db]
            self.collection = self.law_db["documents"]
        except Exception as e:
            print("Exception:", e)
        
    def mget(self, ids):
        """Lấy nhiều Parent Documents dựa trên danh sách ID."""
        results = list(self.collection.find({"id": {"$in": ids}}))
        return {res["id"]: Document(page_content=res["content"], metadata=res['metadata']) for res in results}

    def mset(self, key_value_pairs):
        """Lưu nhiều Parent Documents vào MongoDB."""
        docs_to_insert = [{"id": key, 
                           "metadata": value.metadata, 
                           "content": value.page_content} 
                           for key, value in key_value_pairs]
        
        self.collection.insert_many(docs_to_insert)

    def mdelete(self, keys):
        """Xóa Parent Document khỏi MongoDB."""
        self.collection.deleteMany({"id": keys})

    def yield_keys(self, prefix=None):
        if prefix is None:
            for doc in self.collection.find({}, {"id": 1}):
                yield doc["id"]
        else:
            for doc in self.collection.find({}, {"id": 1}):
                if doc["id"].startswith(prefix):
                    yield doc["id"]

    def close(self):
        self.client.close()