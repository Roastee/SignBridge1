import cv2
import mediapipe as mp
import time
import queue
from collections import deque, Counter
import json
import numpy as np
from streamlit_webrtc import VideoProcessorBase
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load Custom AI Model globally
try:
    import tensorflow as tf
    ai_model = tf.keras.models.load_model("models/sign_model.keras")
    with open("models/classes.json", "r") as f:
        ai_classes = json.load(f)
except Exception:
    ai_model = None
    ai_classes = []

# Initialize HandLandmarker globally
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

def get_supported_classes():
    return ai_classes

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
                
                # AI Integration
                if ai_model is not None:
                    landmarks_list = []
                    base_x = hand_lm[0].x
                    base_y = hand_lm[0].y
                    base_z = hand_lm[0].z
                    
                    for lm in hand_lm:
                        landmarks_list.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                    
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
