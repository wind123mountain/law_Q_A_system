# 🚀 Vietnamese Businiess Q&A System  
## **Project structure**
```
├── App
│   ├── backend
│   │   ├── Dockerfile                              # backend dockerfile
│   │   ├── entrypoint.sh                           # run backend sh file
│   │   ├── requirements.txt                        # backend dependencies
│   │   └── src
│   │       ├── .env                                # .env file
│   │       ├── app.py                              # entry point for the Fast API backend application
│   │       ├── assistant.py                        # assistant file: search, llms
│   │       ├── auth.py                             # authentication file
│   │       ├── brain_v2.py                         # logic for RAG
│   │       ├── cache.py                            # Cache implementation for the application
│   │       ├── config.py                           # configuration file
│   │       ├── database.py                         # database file
│   │       ├── models.py                           # models file: load, updadate history
│   │       ├── schemas.py                          # Data schemas for API endpoints
│   │       ├── search_document_v2                  # search module
│   │       │   ├── BGE_M3                          # embedding module
│   │       │   │   └── modeling.py                 # embedding model
│   │       │   ├── custom_embedding.py             # custom embedding implementtation for LangChain
│   │       │   ├── custom_retriever.py             # custom retriever implementation for LangChain
│   │       │   ├── mongodb_store.py                # mongodb store parent docs implementation for LangChain
│   │       │   ├── tavily_search.py                # tavily search: internet search
│   │       │   └── vector_store.py                 # vector store implementation for LangChain
│   │       ├── tasks.py                            # define task for celery
│   │       └── utils.py                            
│   └── chatbot-ui
│       ├── .env                                    # .env file
│       ├── .streamlit                              # streamlit config
│       │   ├── pages_sections.toml
│       │   └── secrets.toml
│       ├── Dockerfile                              # ui dockerfile
│       ├── config.toml                             # ui config file
│       ├── entrypoint.sh                           # run ui sh file
│       ├── main.py                                 # main file
│       ├── menu.py                                 # menu file
│       ├── pages                                   # pages 
│       │   ├── admin.py                            # admin page
│       │   ├── chat.py                             # chat page
│       │   ├── home.py                             # home page
│       │   └── new_chat.py                         # new chat page
│       ├── requirements.txt                        # ui dependencies
│       └── service.py                              # service file: authentication, call backend
├── BGE_M3                                          # BGE_M3 module
├── Core                                            # Core: build, deploy, experiment RAG
│   ├── README.md                                   # readme file
│   ├── assistant.py                                # assistant file: search, llms
│   ├── chatbot_ui.py                               # chatbot ui file
│   ├── crawl_data.ipynb                            # crawl data notebook
│   ├── deploy.ipynb                                # deploy notebook
│   ├── mongo_read_pass.txt                         # mongo read password
│   ├── qdrantTestApiKey.txt                        # qdrant api key
│   ├── qdrant_read_key.txt                         # qdrant read key
│   ├── requirements.txt                            # core dependencies
│   ├── server.py                                   # simple server file
│   └── utils                                       # utils file
│       ├── __init__.py                             # init file
│       ├── custom_embedding.py                     # custom embedding implementtation for LangChain
│       ├── custom_retriever.py                     # custom retriever implementation for LangChain
│       ├── mongodb_store.py                        # mongodb store parent docs implementation for LangChain
│       └── vector_store.py                         # vector store implementation for LangChain
├── Data                                            # Data folder: corpus, train data, test data
├── bge-m3-output                                   # BGE M3 model checkpoint
├── bge-m3-reranker-output                          # BGE M3 reranker model checkpoint
├── docker-compose.yml                              # docker compose file
├── README.md                                       # readme file
├── finetune_eval.ipynb                             # finetune evaluation BGE M3 model, BGE M3 reranker model notebook
├── preprocessing.ipynb                             # preprocessing data notebook
└── test.ipynb                                      # test notebook
```
## **Docker**
### **Setup config, environment variables**
#### Setup backend:
- **App/backend/src/.env**
```
MONGODB_URL=mongodb://user:password@host:port/dbname
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=your_qdrant_api_key
COLLECTION_NAME=my_vector_collection
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
MONGODB_ADMIN_URL=mongodb://user:password@host:port/dbname
CELERY_BROKER_URL = redis://localhost:6379
CELERY_RESULT_BACKEND = redis://localhost:6379
NVIDIA_API_KEY=your_nvidia_api_key
```

#### Add streamlit env:
- **App/chatbot-ui/.streamlit/pages_sections.toml**
```
[[pages]]
path = "pages/home.py"
name = "Home"
icon = "🏠"

[[pages]]
path = "pages/chat.py"
name = "New Chat"
icon = "💬"
url_path = "new_chat"

[[pages]]
name = "Chat history"
icon = ""
is_section = true
```
- **App/chatbot-ui/.streamlit/secrets.toml**
```
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "cc_secret"
client_id = "<your_key>"
client_secret = "<your_key>"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

```
- **App/chatbot-ui/.env**
```
BACKEND_URL=http://fastapi_app:8002
```

### **Build images**
```bash
docker build --pull --rm -f 'App\backend\Dockerfile' -t 'be_qa:3003' 'App\backend'
```

```bash
docker build --pull --rm -f 'App\chatbot-ui\Dockerfile' -t 'ui:3003' 'App\chatbot-ui'
```

### **Start server & ui**
```bash
docker-compose up -d
```

## **Result**
### Evaluate
The evaluation metrics currently in use are:

  - **Recall@k**: Number of positive found / Total positive
  - **Acc@k**: The returned result has at least 1 positive.
  
  Test on 2000 questions
  
  |Model                    | recall@3  | recall@5  | recall@10 | acc@3     | acc@5     | acc@10    |
  |-------------------------|-----------|-----------|-----------|-----------|-----------|-----------|
  |BGE-m3                   | 0.6264    | 0.7116    | 0.8007    | 0.7856	  | 0.8589	  | 0.917     |
  |BGE-M3 + reank           | 0.6409    | 0.7259    | 0.8077    | 0.8047  	| 0.875	    | 0.9199    |
  |BGE-M3 + BM25 + reank    | 0.6506    | 0.7299    | 0.814     | 0.8101	  | 0.8735	  | 0.9209    |


### **Demo**
![image](https://github.com/user-attachments/assets/8a3ee0a7-93e9-4439-b7c8-831e83864a1b)


