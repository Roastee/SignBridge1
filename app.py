import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import cv2
import mediapipe as mp
import time
import queue
from collections import deque, Counter
import json
import numpy as np

from utils.speech import speak_text

# Try to load custom AI model globally
try:
    import tensorflow as tf
    ai_model = tf.keras.models.load_model("sign_model.keras")
    with open("classes.json", "r") as f:
        ai_classes = json.load(f)
    print("✅ Custom AI Model loaded into Streamlit successfully!")
except Exception as e:
    ai_model = None
    ai_classes = []
    print(f"⚠️ Neural Network not found or failed to load. Please train it first. Error: {e}")

st.set_page_config(page_title="SignBridge Web-Pro", page_icon="🤟", layout="wide")

# ============================================================
# PREMIUM GLASSMORPHIC DESIGN SYSTEM
# ============================================================
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

# ============================================================
# STATE MANAGEMENT
# ============================================================
if "word_buffer" not in st.session_state: st.session_state.word_buffer = ""
if "history" not in st.session_state: st.session_state.history = []

# ============================================================
# VIDEO PROCESSING ENGINE (MediaPipe Tasks API)
# ============================================================
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize HandLandmarker globally (Modern API immune to Protobuf errors)
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
detector_global = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8), # Index
    (5, 9), (9, 10), (10, 11), (11, 12), # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = detector_global
        self.history = deque(maxlen=12)
        
        self.current_letter = None
        self.letter_start_time = None
        self.already_spoken = False
        self.hold_duration = 2.0
        
        self.result_queue = queue.Queue()
        self.detected_letter = None
        self.progress = 0.0

    def recv(self, frame):
        import av
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process using Tasks API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)
        
        raw_letter = None
        if results.hand_landmarks:
            for hand_lm in results.hand_landmarks:
                h, w, _ = img.shape
                # Custom Drawing
                for connection in HAND_CONNECTIONS:
                    start_lm = hand_lm[connection[0]]
                    end_lm = hand_lm[connection[1]]
                    cv2.line(img, (int(start_lm.x * w), int(start_lm.y * h)), 
                                  (int(end_lm.x * w), int(end_lm.y * h)), (0, 230, 230), 2)
                for lm in hand_lm:
                    cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 3, (50, 50, 50), -1)
                
                # --- AI NEURAL NETWORK INTEGRATION ---
                if 'ai_model' in globals() and ai_model is not None:
                    landmarks_list = []
                    base_x = hand_lm[0].x
                    base_y = hand_lm[0].y
                    base_z = hand_lm[0].z
                    
                    for lm in hand_lm:
                        landmarks_list.extend([
                            lm.x - base_x, 
                            lm.y - base_y, 
                            lm.z - base_z
                        ])
                    
                    input_data = np.array([landmarks_list])
                    prediction = ai_model(input_data, training=False)
                    class_id = np.argmax(prediction[0])
                    confidence = prediction[0][class_id]
                    
                    if confidence > 0.6:
                        raw_letter = ai_classes[class_id]
                
        # Smoothing filter
        self.history.append(raw_letter)
        counter = Counter(self.history)
        most_common = counter.most_common(1)
        
        if most_common:
            smoothed, count = most_common[0]
            self.detected_letter = smoothed if (count >= 7 and smoothed is not None) else None
        else:
            self.detected_letter = None
            
        # Time-based hold logic
        now = time.time()
        self.progress = 0.0
        
        if self.detected_letter:
            if self.detected_letter == self.current_letter:
                if not self.already_spoken and self.letter_start_time is not None:
                    elapsed = now - self.letter_start_time
                    self.progress = min(1.0, elapsed / self.hold_duration)
                    if elapsed >= self.hold_duration:
                        self.result_queue.put(self.detected_letter)
                        self.already_spoken = True
            else:
                self.current_letter = self.detected_letter
                self.letter_start_time = now
                self.already_spoken = False
        else:
            self.current_letter = None
            self.letter_start_time = None
            self.already_spoken = False
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ============================================================
# APP LAYOUT & UI
# ============================================================
st.markdown("<h1 style='text-align: center; color: #ffffff; margin-bottom: 0;'>SignBridge <span style='color: #00E676;'>Web-Pro</span> 🤟</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8892b0; margin-bottom: 30px; font-weight: 600;'>Advanced ASL Recognition Engine</p>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Settings")
    st.markdown("**Supported Gestures:**")
    if 'ai_classes' in globals() and ai_classes:
        st.caption(" ".join(ai_classes))
    else:
        st.caption("A B C D E F I K L O U V W Y")
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
        rtc_configuration=RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }),
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
if c1.button("␣ Add Space", use_container_width=True):
    st.session_state.word_buffer += " "
if c2.button("⌫ Backspace", use_container_width=True):
    st.session_state.word_buffer = st.session_state.word_buffer[:-1]
if c3.button("🗑️ Clear", use_container_width=True):
    st.session_state.word_buffer = ""
if c4.button("🔊 Speak Word", use_container_width=True):
    if st.session_state.word_buffer.strip():
        speak_text(st.session_state.word_buffer)
        st.session_state.history.append(st.session_state.word_buffer)
        st.session_state.word_buffer = ""
        st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

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
    
    if dl:
        letter_placeholder.markdown(f"<div class='hud-letter'>{dl}</div>", unsafe_allow_html=True)
    else:
        letter_placeholder.markdown("<div class='hud-letter' style='color: rgba(255,255,255,0.1);'>-</div>", unsafe_allow_html=True)
        
    progress_placeholder.progress(prog)
    
    time.sleep(0.5)
    st.rerun()
