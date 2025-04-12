import json
import logging
import time

import requests
import streamlit as st

st.title("🤖 Chatbot Interface")

def get_bot_response(user_message):
    url = f"http://127.0.0.1:8000/ask"
    json={"question": user_message}

    response = requests.post(url, json=json, timeout=60)
    if response.status_code != 200:
        raise TimeoutError(f"Get bot response fail: {response.text}")
    return response.json()['response']


# Streamed response
def response_generator(user_message):
    res = get_bot_response(user_message)
    for line in res.split("\n\n"):
        logging.info(f"Line: {line}")
        for sen in line.split("\n"):
            yield sen + '\n\n'
            time.sleep(0.05)
        yield '\n'
    return res


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
