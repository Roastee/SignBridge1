import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import cv2
import mediapipe as mp
import time
import queue
from collections import deque, Counter

from utils.recognizer import classify_gesture
from utils.speech import speak_text

st.set_page_config(page_title="SignBridge Web-Pro", page_icon="🤟", layout="wide")

# ============================================================
# PREMIUM GLASSMORPHIC DESIGN SYSTEM
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f111a 0%, #1a1d2e 100%);
        color: #ffffff;
    }
    
    /* Hide top bar */
    header {visibility: hidden;}
    
    .glass-panel {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    
    .word-buffer {
        font-size: 3.5rem;
        font-weight: 800;
        color: #00E676;
        letter-spacing: 4px;
        min-height: 90px;
        text-align: center;
        border-bottom: 2px solid rgba(0, 230, 118, 0.3);
        padding: 20px;
        margin-bottom: 10px;
        text-shadow: 0 0 20px rgba(0, 230, 118, 0.2);
    }
    
    .hud-letter {
        font-size: 10rem;
        font-weight: 800;
        text-align: center;
        color: #ffffff;
        text-shadow: 0 0 30px rgba(255,255,255,0.4);
        margin: 0;
        line-height: 1.2;
    }
    
    .stButton > button {
        background: rgba(0, 230, 118, 0.1);
        border: 1px solid rgba(0, 230, 118, 0.5);
        color: #00E676;
        border-radius: 12px;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
        height: auto;
    }
    
    .stButton > button:hover {
        background: #00E676;
        color: #0f111a;
        box-shadow: 0 0 20px rgba(0, 230, 118, 0.4);
        border-color: #00E676;
    }
    
    .stProgress > div > div > div {
        background-color: #00E676;
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# STATE MANAGEMENT
# ============================================================
if "word_buffer" not in st.session_state:
    st.session_state.word_buffer = ""
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# VIDEO PROCESSING ENGINE (MediaPipe)
# ============================================================

# Initialize MediaPipe globally to prevent threading/hot-reloading issues
mp_hands_global = mp.solutions.hands
mp_drawing_global = mp.solutions.drawing_utils

class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp_hands_global
        self.hands = self.mp_hands.Hands(
            model_complexity=0, 
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7, 
            max_num_hands=1
        )
        self.mp_drawing = mp_drawing_global
        self.history = deque(maxlen=12)
        
        self.current_letter = None
        self.letter_start_time = None
        self.already_spoken = False
        self.hold_duration = 2.0
        
        # Thread-safe queue to pass letters back to the Streamlit UI thread
        self.result_queue = queue.Queue()
        
        self.detected_letter = None
        self.progress = 0.0

    def recv(self, frame):
        import av
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror image for selfie view
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process hand landmarks
        rgb.flags.writeable = False
        results = self.hands.process(rgb)
        rgb.flags.writeable = True
        
        raw_letter = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                self.mp_drawing.draw_landmarks(
                    img, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 230, 230), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(50, 50, 50), thickness=2)
                )
                label = hand_info.classification[0].label
                raw_letter = classify_gesture(hand_lm, label)
                
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
                    
                    # Target reached: Add to queue
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

# SIDEBAR
with st.sidebar:
    st.markdown("<div class='glass-panel'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Settings")
    st.markdown("**Supported Gestures:**")
    st.caption("A B C D E F I K L O U V W Y")
    st.markdown("---")
    st.markdown("### 📖 Session History")
    
    if len(st.session_state.history) == 0:
        st.caption("No words spoken yet.")
    else:
        for w in reversed(st.session_state.history):
            st.markdown(f"• **{w}**")
            
    st.markdown("</div>", unsafe_allow_html=True)

# MAIN CONTENT
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
    
    # Default state
    letter_placeholder.markdown("<div class='hud-letter' style='color: rgba(255,255,255,0.1);'>-</div>", unsafe_allow_html=True)
    progress_placeholder.progress(0.0)
    
    st.markdown("</div>", unsafe_allow_html=True)

# WORD BUFFER & CONTROLS
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

# ============================================================
# EVENT LOOP / RERUN TRIGGER
# ============================================================
if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
    processor = webrtc_ctx.video_processor
    
    # Poll the queue for new letters
    state_changed = False
    try:
        while True:
            new_letter = processor.result_queue.get_nowait()
            st.session_state.word_buffer += new_letter
            speak_text(f"{new_letter}")
            state_changed = True
    except queue.Empty:
        pass
    
    # Update word buffer UI if new letter came in
    if state_changed:
        display_word = st.session_state.word_buffer if st.session_state.word_buffer else "_"
        word_placeholder.markdown(f"<div class='word-buffer'>{display_word}</div>", unsafe_allow_html=True)
    
    # Update HUD with current real-time stats
    dl = processor.detected_letter
    prog = processor.progress
    
    if dl:
        letter_placeholder.markdown(f"<div class='hud-letter'>{dl}</div>", unsafe_allow_html=True)
    else:
        letter_placeholder.markdown("<div class='hud-letter' style='color: rgba(255,255,255,0.1);'>-</div>", unsafe_allow_html=True)
        
    progress_placeholder.progress(prog)
    
    # Auto-rerun to keep the UI in sync with the WebRTC thread
    time.sleep(0.5)
    st.rerun()
