import streamlit as st
import time
import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from frontend.styles import apply_custom_css
from backend.processor import SignProcessor, get_supported_classes
from utils.speech import speak_text

st.set_page_config(page_title="SignBridge Web-Pro", page_icon="🤟", layout="wide")

# Apply UI styles
apply_custom_css()

# Initialize state
if "word_buffer" not in st.session_state: st.session_state.word_buffer = ""
if "history" not in st.session_state: st.session_state.history = []

# ============================================================
# UI LAYOUT
# ============================================================
st.markdown("<h1 style='text-align: center; color: #ffffff; margin-bottom: 0;'>SignBridge <span style='color: #00E676;'>Web-Pro</span> 🤟</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8892b0; margin-bottom: 30px; font-weight: 600;'>Advanced AI ASL Recognition Engine</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Settings")
    st.markdown("**Supported Gestures:**")
    classes = get_supported_classes()
    if classes:
        st.caption(" ".join(classes))
    else:
        st.caption("No models found. Run data_collector.py")
    st.markdown("---")
    st.markdown("### 📖 Session History")
    
    if len(st.session_state.history) == 0:
        st.caption("No words spoken yet.")
    else:
        for w in reversed(st.session_state.history):
            st.markdown(f"• **{w}**")
    st.markdown("</div>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    webrtc_ctx = webrtc_streamer(
        key="signbridge",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_processor_factory=SignProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass-panel' style='min-height: 480px; display: flex; flex-direction: column; justify-content: center;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #8892b0; margin-top: 0;'>CURRENT GESTURE</h4>", unsafe_allow_html=True)
    
    letter_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    letter_placeholder.markdown("<div class='hud-letter' style='color: rgba(255,255,255,0.1);'>-</div>", unsafe_allow_html=True)
    progress_placeholder.progress(0.0)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
st.markdown("<p style='color: #8892b0; margin-bottom: 0; font-weight: 600;'>CONSTRUCTED WORD</p>", unsafe_allow_html=True)

word_placeholder = st.empty()
display_word = st.session_state.word_buffer if st.session_state.word_buffer else "_"
word_placeholder.markdown(f"<div class='word-buffer'>{display_word}</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
if c1.button("␣ Add Space", use_container_width=True): st.session_state.word_buffer += " "
if c2.button("⌫ Backspace", use_container_width=True): st.session_state.word_buffer = st.session_state.word_buffer[:-1]
if c3.button("🗑️ Clear", use_container_width=True): st.session_state.word_buffer = ""
if c4.button("🔊 Speak Word", use_container_width=True):
    if st.session_state.word_buffer.strip():
        speak_text(st.session_state.word_buffer)
        st.session_state.history.append(st.session_state.word_buffer)
        st.session_state.word_buffer = ""
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# Event Loop Processing
if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
    processor = webrtc_ctx.video_processor
    state_changed = False
    try:
        while True:
            new_letter = processor.result_queue.get_nowait()
            st.session_state.word_buffer += new_letter
            speak_text(f"{new_letter}")
            state_changed = True
    except queue.Empty:
        pass
    
    if state_changed:
        display_word = st.session_state.word_buffer if st.session_state.word_buffer else "_"
        word_placeholder.markdown(f"<div class='word-buffer'>{display_word}</div>", unsafe_allow_html=True)
    
    dl = processor.detected_letter
    prog = processor.progress
    
    if dl: letter_placeholder.markdown(f"<div class='hud-letter'>{dl}</div>", unsafe_allow_html=True)
    else: letter_placeholder.markdown("<div class='hud-letter' style='color: rgba(255,255,255,0.1);'>-</div>", unsafe_allow_html=True)
        
    progress_placeholder.progress(prog)
    time.sleep(0.5)
    st.rerun()
