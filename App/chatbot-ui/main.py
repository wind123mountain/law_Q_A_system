# import streamlit as st

# pg = st.navigation([
#     st.Page("chat_login.py", title="Home", icon="🏠"),
#     st.Page("pages/chat.py", title="Chat", icon="💬"),
# ])

# pg.run()

import streamlit as st
from service import fetch_conversations
from st_pages import get_nav_from_toml, hide_pages

st.set_page_config(layout="wide")

# sections = st.sidebar.toggle("Sections", value=True, key="use_sections")

nav = get_nav_from_toml(
    ".streamlit/pages_sections.toml"  # if sections else ".streamlit/pages.toml"
)

if st.experimental_user.is_logged_in:
    convs = fetch_conversations(st.experimental_user.email)
    st.sidebar.button("Log out", on_click=st.logout, use_container_width=True)
    nav["Chat history"] = []
    for conv in convs:
        if conv == st.query_params.get("id"):
            continue
        nav["Chat history"] += [
            st.Page(
                "pages/chat.py",
                title=conv,
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
