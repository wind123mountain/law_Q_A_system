import json
import logging
import time
import os

import requests
import streamlit as st
from menu import menu_with_redirect
from service import fetch_conversation_details
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

# Redirect to app.py if not logged in, otherwise show the navigation menu
menu_with_redirect()

ctx = get_script_run_ctx()
all_app_pages = ctx.pages_manager.get_pages()
try:
    current_page = all_app_pages[ctx.page_script_hash]
except KeyError:
    current_page = [
        p
        for p in all_app_pages.values()
        if p["relative_page_hash"] == ctx.page_script_hash
    ][0]

header = st.container()
header.title(f"🤖 AI Legal assistant")
header.write("""<div class='fixed-header'/>""", unsafe_allow_html=True)

### Custom CSS for the sticky header
st.markdown("""
    <style>
    .dark-container {
        padding: 20px;
        border-radius: 10px;
        color: white;
    }

    @media (prefers-color-scheme: dark) {
        .dark-container {
            background-color: #1e1e1e;
        }
    }

    @media (prefers-color-scheme: light) {
        .dark-container {
            background-color: #f0f0f0;
            color: black;
        }
    }
    </style>
""", unsafe_allow_html=True)


# Input bot
bot_id = "botLegal"
user_id = st.experimental_user.email
BACKEND_URL = os.getenv("BACKEND_URL", "http://fastapi_app:8002")
converation_id = st.query_params.get("id", current_page["url_pathname"])


def send_user_request(text, chat_id):
    url = f"{BACKEND_URL}/chat/complete"

    payload = json.dumps(
        {
            "user_message": text,
            "user_id": str(user_id),
            "bot_id": bot_id,
            "conversation_id": st.query_params.get("id", chat_id),
        }
    )
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload, timeout=10)
    if response.status_code != 200:
        raise TimeoutError(f"Request to bot fail: {response.text}")
    return json.loads(response.text)


def get_bot_response(request_id):
    url = f"{BACKEND_URL}/chat/complete_v2/{request_id}"

    response = requests.request("GET", url, headers={}, data="", timeout=30)
    if response.status_code != 200:
        raise TimeoutError(f"Get bot response fail: {response.text}")
    return response.status_code, json.loads(response.text)


def get_chat_complete(text, chat_id):
    user_request = send_user_request(text, chat_id)
    request_id = user_request["task_id"]
    status_code, chat_response = get_bot_response(request_id)
    if status_code == 200:
        return (
            chat_response["task_result"]["content"],
            chat_response["task_result"]["chat_id"],
        )
    else:
        raise TimeoutError("Request fail, try again please")


# Streamed response
def response_generator(user_message, chat_id):
    def generate_stream(res_txt):
        for line in res_txt.split("\n\n"):
            logging.info(f"Line: {line}")
            for sen in line.split("\n"):
                yield sen + "\n\n"
                time.sleep(0.05)
            yield "\n"

    res, chat_id = get_chat_complete(user_message, chat_id)
    return generate_stream(res), chat_id


if (
    converation_id
    and converation_id != "new_chat"
    and st.experimental_user.is_logged_in
):
    # Initialize chat history
    history = fetch_conversation_details(converation_id, st.experimental_user.email)
    st.session_state.messages = history["result"]

if "messages" not in st.session_state or converation_id == "new_chat":
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Bạn muốn hỏi gì về luật liên quan đến doanh nghiệp?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        searching = st.markdown(" Searching...")
        resp_message, chat_id = response_generator(prompt, converation_id)
        searching.empty()
        response = st.write_stream(resp_message)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    if converation_id == "new_chat":
        st.query_params["id"] = chat_id
    # st.rerun()
