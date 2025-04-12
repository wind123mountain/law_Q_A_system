import streamlit as st


def authenticated_menu():
    # Show a navigation menu for authenticated users
    st.sidebar.page_link(
        st.Page(
            "pages/chat.py",
            title="Chat 1",
            icon="💬",
            url_path="chat_1",
        ),
        label="Chat history",
    )
    st.sidebar.page_link(
        st.Page(
            "pages/chat.py",
            title="Chat 2",
            icon="💬",
            url_path="chat_2",
        ),
        label="Chat history",
    )


def menu():
    # Determine if a user is logged in or not, then show the correct
    # navigation menu
    if st.experimental_user.is_logged_in:
        # authenticated_menu()
        pass


def menu_with_redirect():
    # Redirect users to the main page if not logged in, otherwise continue to
    # render the navigation menu
    if not st.experimental_user.is_logged_in:
        st.switch_page("pages/home.py")

    menu()
