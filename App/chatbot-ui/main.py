from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from service import fetch_conversations
from st_pages import get_nav_from_toml, hide_pages

st.set_page_config(layout="wide")

# sections = st.sidebar.toggle("Sections", value=True, key="use_sections")

nav = get_nav_from_toml(
    ".streamlit/pages_sections.toml"  # if sections else ".streamlit/pages.toml"
)

if st.experimental_user.is_logged_in:
    nav["New Chat"] = [
            st.Page(
                "pages/new_chat.py",
                title="New chat",
                icon="💬",
                url_path="new",
            )
        ]

    convs = fetch_conversations(st.experimental_user.email)
    st.sidebar.button("Log out", on_click=st.logout, use_container_width=True)
    nav["Chat history"] = []
    for conv, mess in convs:
        if conv == st.query_params.get("id"):
            continue
        nav["Chat history"] += [
            st.Page(
                "pages/chat.py",
                title=mess,
                icon="💬",
                url_path=conv,
            )
        ]
else:
    hide_pages("New Chat")
# st.logo("logo.png")

pg = st.navigation(nav)

# add_page_title(pg)

pg.run()
