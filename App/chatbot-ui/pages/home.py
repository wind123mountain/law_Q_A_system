from datetime import datetime

import requests
import streamlit as st
from menu import menu


def login_screen():
    st.header("This app is private.")
    st.subheader("Please log in.")
    st.button("Log in with Google", on_click=st.login)


if not st.experimental_user.is_logged_in:
    login_screen()
else:
    st.header(f"Welcome, {st.experimental_user.name}!")
    st.success(f"Xin chào, {st.experimental_user['email']}!")
    # Giả lập dữ liệu lấy từ id_token của Google
    user_info = st.experimental_user.to_dict()

    # Cấu hình giao diện
    # st.set_page_config(page_title="Profile", page_icon="👤", layout="centered")

    # Header
    # st.markdown("<h1 style='text-align: center;'>👋 Welcome, {}</h1>".format(user_info['name']), unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    # Ảnh đại diện
    with col1:
        st.image(
            user_info["picture"],
            use_container_width=True,
            caption="Your profile picture",
            output_format="JPEG",
        )
    with col2:
        # Thông tin người dùng
        st.markdown("### 📌 Thông tin cá nhân")
        st.write(f"**Họ và tên:** {user_info['name']}")
        st.write(f"**Email:** {user_info['email']}")
        st.write(f"**Tên riêng:** {user_info['given_name']}")
        st.write(f"**Họ:** {user_info['family_name']}")

    with col3:
        # Thông tin token (hiển thị thời gian đăng nhập hết hạn)
        iat = datetime.fromtimestamp(user_info["iat"])
        exp = datetime.fromtimestamp(user_info["exp"])
        st.markdown("### ⏱ Phiên đăng nhập")
        st.write(f"**Đăng nhập lúc:** {iat.strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Hết hạn lúc:** {exp.strftime('%Y-%m-%d %H:%M:%S')}")

    # Footer hoặc thêm tính năng
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>Made with ❤️ by Huy</div>",
        unsafe_allow_html=True,
    )

menu()  # Render the dynamic menu!
