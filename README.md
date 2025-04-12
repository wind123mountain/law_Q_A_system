# 🚀 Vietnamese Businiess Q&A System  

---

## **Requirements**  

```bash
pip install -r requirements.txt
```

## **Run server**
```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

## **Run client**
```bash
python -m streamlit run chatbot_ui.py
```


## Docker:
Build images
```bash
docker build --pull --rm -f 'App\backend\Dockerfile' -t 'be_qa:3003' 'App\backend'
```

```bash
docker build --pull --rm -f 'App\chatbot-ui\Dockerfile' -t 'ui:3003' 'App\chatbot-ui'
```

Setup .env file:
```
MONGODB_URL=mongodb://user:password@host:port/dbname
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=your_qdrant_api_key
COLLECTION_NAME=my_vector_collection
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Start server & ui
```bash
docker compose up -d
```
