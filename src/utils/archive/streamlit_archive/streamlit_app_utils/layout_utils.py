# layout_utils.py
import streamlit as st
import base64
from pathlib import Path

def get_base64_asset(relative_path: str = "App_Logo.png") -> str:
    abs_path = Path(__file__).parent.parent.parent / "assets" / "app_images" / relative_path
    with open(abs_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def render_logo(title="Welcome to Blood Cell AI 🧪"):
    logo = get_base64_asset()
    st.markdown(
        f"""
        <style>
            .custom-header {{
                display: flex;
                justify-content: left;
                align-items: center;
                padding: 5px 0;
                border-bottom: 1px solid #ddd;
                margin-bottom: 15px;
                background-color: transparent;  /* 🔲 black background */
            }}
            .logo-wrapper {{
                height: 100px;
                margin-left: 20px;
                margin-right: 30px;  /* ✅ force margin here */
                display: inline-block;
                align-items: center;
            }}
            .custom-header img {{
                height: 100px;
                margin-right: 30px;
                display: block;
                object-fit: contain; 
            }}
            .custom-header-text {{
                font-size: 3.3em;
                font-weight: 600;
                line-height: 1.2;  /* tighter line height for bottom alignment */
                padding-bottom: 0px;  /* tweak to align perfectly */
                /*color: black; */ /* ⚪ white text */
            }}


        </style>

        <div class="custom-header">
            <div class="logo-wrapper">
                <img src="data:image/png;base64,{logo}" alt="NTC-TAP Logo">
            </div>
            <div class="custom-header-text">{title}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
