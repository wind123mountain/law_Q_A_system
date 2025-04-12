import os
import time

import requests
import streamlit as st

BACKEND_URL = "http://fastapi_app:8002"
LIST_CONVERSATIONS_ENDPOINT = "/conversations"
GET_CONVERSATION_DETAILS_ENDPOINT = "/conversations/{}"


def fetch_conversations(user_id):
    response = requests.get(
        f"{BACKEND_URL}{LIST_CONVERSATIONS_ENDPOINT}", headers={"user_id": user_id}
    )
    if response.status_code == 200:
        return response.json()["result"][
            -10:
        ]  # Giả sử API trả về danh sách các cuộc trò chuyện dưới dạng JSON
    else:
        # st.error("Không thể tải danh sách cuộc trò chuyện.")
        return []


# Hàm lấy chi tiết một cuộc trò chuyện từ API
def fetch_conversation_details(conversation_id, user_id):
    response = requests.get(
        f"{BACKEND_URL}{GET_CONVERSATION_DETAILS_ENDPOINT.format(conversation_id)}",
        headers={"user_id": user_id},
    )
    if response.status_code == 200:
        return (
            response.json()
        )  # Giả sử API trả về chi tiết cuộc trò chuyện dưới dạng JSON
    else:
        st.error("Không thể tải chi tiết cuộc trò chuyện.")
        return None


# conversations = fetch_conversations()
