import streamlit as st

def apply_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
        .stApp { background: linear-gradient(135deg, #0f111a 0%, #1a1d2e 100%); color: #ffffff; }
        header {visibility: hidden;}
        .glass-panel { background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 20px; padding: 24px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2); margin-bottom: 20px; }
        .word-buffer { font-size: 3.5rem; font-weight: 800; color: #00E676; letter-spacing: 4px; min-height: 90px; text-align: center; border-bottom: 2px solid rgba(0, 230, 118, 0.3); padding: 20px; margin-bottom: 10px; text-shadow: 0 0 20px rgba(0, 230, 118, 0.2); }
        .hud-letter { font-size: 10rem; font-weight: 800; text-align: center; color: #ffffff; text-shadow: 0 0 30px rgba(255,255,255,0.4); margin: 0; line-height: 1.2; }
        .stButton > button { background: rgba(0, 230, 118, 0.1); border: 1px solid rgba(0, 230, 118, 0.5); color: #00E676; border-radius: 12px; font-weight: 600; padding: 12px 24px; transition: all 0.3s ease; height: auto; }
        .stButton > button:hover { background: #00E676; color: #0f111a; box-shadow: 0 0 20px rgba(0, 230, 118, 0.4); border-color: #00E676; }
        .stProgress > div > div > div { background-color: #00E676; }
        hr { border-color: rgba(255, 255, 255, 0.1); }
    </style>
    """, unsafe_allow_html=True)
