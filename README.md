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

Add streamlit env:
- **App\chatbot-ui\.streamlit\pages_sections.toml**
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
- **App\chatbot-ui\.streamlit\secrets.toml**
```
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "cc_secret"
client_id = "<your_key>"
client_secret = "<your_key>"
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

```

Start server & ui
```bash
docker compose up -d
```
