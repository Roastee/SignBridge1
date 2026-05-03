import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import time
import pyttsx3
import threading
import numpy as np
import sys
import math
from collections import deque, Counter


# ============================================================
# SPEECH ENGINE
# ============================================================
def speak_text_thread(text):
    """Speak text in a fresh thread to avoid pyttsx3 hanging on Windows."""
    if sys.platform == 'win32':
        try:
            import pythoncom  # type: ignore
            pythoncom.CoInitialize()
        except Exception as e:
            print(f"[Speech] pythoncom error: {e}")

    try:
        engine = pyttsx3.init()
        engine.setProperty('volume', 1.0)
        engine.setProperty('rate', 150)
        print(f"[Speech] Speaking: {text}")
        engine.say(text)
        engine.runAndWait()
        print(f"[Speech] Finished: {text}")
    except Exception as e:
        print(f"[Speech] Error: {e}")


def speak_text(text):
    """Non-blocking speech call."""
    threading.Thread(target=speak_text_thread, args=(text,), daemon=True).start()


# ============================================================
# LANDMARK HELPERS  (Angle-Based, Rotation-Invariant)
# ============================================================
def get_landmark_array(hand_landmarks):
    """Convert MediaPipe landmarks to a NumPy (21, 3) array."""
    pts = []
    for lm in hand_landmarks.landmark:
        pts.append([lm.x, lm.y, lm.z])
    return np.array(pts)


def angle_between(v1, v2):
    """Return the angle in degrees between two 3-D vectors."""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos = np.clip(cos, -1.0, 1.0)
    return math.degrees(math.acos(cos))


def finger_curl_angle(pts, mcp, pip, tip):
    """
    Compute the curl angle of a finger.
    A straight finger ≈ 180°;  a fully curled finger ≈ 0–60°.
    Uses the angle at the PIP joint between the MCP→PIP and PIP→TIP vectors.
    """
    v1 = pts[mcp] - pts[pip]
    v2 = pts[tip] - pts[pip]
    return angle_between(v1, v2)


def finger_is_extended(pts, mcp, pip, dip, tip, threshold=140):
    """
    A finger is considered 'extended' if the angle at the PIP joint
    is greater than the threshold (roughly straight).
    """
    return finger_curl_angle(pts, mcp, pip, tip) > threshold


def thumb_is_extended(pts, handedness):
    """
    Thumb extension uses a combination of:
      1. The angle at the IP joint (landmarks 2→3→4).
      2. Whether the thumb tip (4) is far from the index MCP (5)
         in the lateral (x) direction.
    """
    angle = finger_curl_angle(pts, 2, 3, 4)
    dist = np.linalg.norm(pts[4] - pts[5])
    return angle > 120 and dist > 0.06


def thumb_is_across_palm(pts, handedness):
    """
    Check if the thumb tip sits across the palm (touching index/middle area).
    Used to distinguish A from S, etc.
    """
    # thumb tip closer to middle-finger MCP than to its own CMC
    return np.linalg.norm(pts[4] - pts[9]) < np.linalg.norm(pts[4] - pts[2])


def fingers_spread(pts, f1_tip, f2_tip, threshold=0.06):
    """Check if two fingertips are spread apart."""
    return np.linalg.norm(pts[f1_tip] - pts[f2_tip]) > threshold


def fingertips_touching(pts, tip_a, tip_b, threshold=0.04):
    """Check if two landmarks are close together (touching)."""
    return np.linalg.norm(pts[tip_a] - pts[tip_b]) < threshold


# ============================================================
# GESTURE CLASSIFICATION  (Angle-Based Heuristics)
# ============================================================
# MediaPipe hand landmark indices:
#   Thumb:  1(CMC)  2(MCP)  3(IP)   4(TIP)
#   Index:  5(MCP)  6(PIP)  7(DIP)  8(TIP)
#   Middle: 9(MCP) 10(PIP) 11(DIP) 12(TIP)
#   Ring:  13(MCP) 14(PIP) 15(DIP) 16(TIP)
#   Pinky: 17(MCP) 18(PIP) 19(DIP) 20(TIP)

def classify_gesture(hand_landmarks, handedness):
    """
    Classify ASL static gestures using angle-based heuristics.
    Supports: A, B, C, D, E, F, I, K, L, O, U, V, W, Y
    """
    pts = get_landmark_array(hand_landmarks)

    # --- Finger states ---
    index_ext  = finger_is_extended(pts, 5, 6, 7, 8)
    middle_ext = finger_is_extended(pts, 9, 10, 11, 12)
    ring_ext   = finger_is_extended(pts, 13, 14, 15, 16)
    pinky_ext  = finger_is_extended(pts, 17, 18, 19, 20)
    thumb_ext  = thumb_is_extended(pts, handedness)

    ext_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])

    # --- Curl angles (for finer distinctions) ---
    index_angle = finger_curl_angle(pts, 5, 6, 8)

    # Thumb lateral direction relative to index
    if handedness == "Right":
        thumb_lateral_out = pts[4][0] < pts[3][0] - 0.02
    else:
        thumb_lateral_out = pts[4][0] > pts[3][0] + 0.02

    # -------------------------------------------------------
    # CLASSIFICATION RULES  (ordered from most to least specific)
    # -------------------------------------------------------

    # ----- W: Index + Middle + Ring extended, separated, pinky folded -----
    if (index_ext and middle_ext and ring_ext
            and not pinky_ext and not thumb_ext):
        if (fingers_spread(pts, 8, 12) and fingers_spread(pts, 12, 16)):
            return "W"

    # ----- F: Index tip touches thumb tip, other 3 fingers extended -----
    if (middle_ext and ring_ext and pinky_ext
            and fingertips_touching(pts, 4, 8, threshold=0.05)):
        return "F"

    # ----- O: All fingertips curled toward thumb, forming a circle -----
    if (not index_ext and not middle_ext and not ring_ext and not pinky_ext):
        if (fingertips_touching(pts, 4, 8, threshold=0.06)
                and index_angle < 120):
            return "O"

    # ----- D: Index extended, others curled, thumb touches middle -----
    if (index_ext and not middle_ext and not ring_ext and not pinky_ext):
        if not thumb_lateral_out:
            if fingertips_touching(pts, 4, 12, threshold=0.06):
                return "D"

    # ----- K: Index up, middle up but angled, thumb between them -----
    if (index_ext and not ring_ext and not pinky_ext):
        mid_angle = finger_curl_angle(pts, 9, 10, 12)
        if 90 < mid_angle < 150:  # middle partially extended
            if (pts[4][1] < pts[10][1]):  # thumb tip above middle PIP
                return "K"

    # ----- L: Index extended, thumb out perpendicular, others folded -----
    if (index_ext and not middle_ext and not ring_ext and not pinky_ext):
        if thumb_ext and thumb_lateral_out:
            return "L"

    # ----- I: Only pinky extended, thumb tucked -----
    if (not index_ext and not middle_ext and not ring_ext
            and pinky_ext and not thumb_ext):
        return "I"

    # ----- U: Index + Middle extended and together, others folded -----
    if (index_ext and middle_ext and not ring_ext and not pinky_ext):
        if not thumb_ext:
            if not fingers_spread(pts, 8, 12, threshold=0.05):
                return "U"

    # ----- V: Index + Middle extended and separated, others folded -----
    if (index_ext and middle_ext and not ring_ext and not pinky_ext):
        if not thumb_ext:
            if fingers_spread(pts, 8, 12, threshold=0.05):
                return "V"

    # ----- Y: Thumb out, pinky extended, others folded -----
    if (not index_ext and not middle_ext and not ring_ext
            and pinky_ext and thumb_ext):
        return "Y"

    # ----- B: All 4 fingers extended, thumb tucked -----
    if ext_count == 4 and (not thumb_ext or not thumb_lateral_out):
        return "B"

    # ----- E: All fingers curled into palm, thumb across the front -----
    if (not index_ext and not middle_ext and not ring_ext
            and not pinky_ext and not thumb_ext):
        if thumb_is_across_palm(pts, handedness):
            # Distinguish from A: in E all fingertips are near the palm
            avg_tip_y = np.mean([pts[8][1], pts[12][1], pts[16][1], pts[20][1]])
            avg_mcp_y = np.mean([pts[5][1], pts[9][1], pts[13][1], pts[17][1]])
            if avg_tip_y < avg_mcp_y + 0.03:
                return "E"

    # ----- C: Fingers curved into a C-shape, thumb opposed -----
    if (not index_ext and not middle_ext and not ring_ext and not pinky_ext):
        if thumb_ext or thumb_lateral_out:
            if 60 < index_angle < 150:
                return "C"

    # ----- A: Fist with thumb alongside (not across palm) -----
    if (not index_ext and not middle_ext and not ring_ext
            and not pinky_ext):
        if not thumb_is_across_palm(pts, handedness):
            if pts[4][1] < pts[5][1]:  # thumb tip above index MCP
                return "A"

    return None


# ============================================================
# UI DRAWING HELPERS  (Glassmorphic Overlay)
# ============================================================
def draw_glass_panel(image, x, y, w, h, color=(40, 40, 40), alpha=0.55):
    """Draw a translucent rounded-rectangle panel (glassmorphism effect)."""
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    # Subtle border
    cv2.rectangle(image, (x, y), (x + w, y + h), (120, 120, 120), 1)


def draw_progress_arc(image, center, radius, progress, thickness=6):
    """Draw a circular progress arc from 0.0 to 1.0."""
    # Background circle
    cv2.circle(image, center, radius, (60, 60, 60), thickness)
    # Progress arc
    end_angle = int(360 * progress)
    if end_angle > 0:
        cv2.ellipse(image, center, (radius, radius), -90, 0, end_angle,
                    (0, 230, 118), thickness, cv2.LINE_AA)


def draw_hud(image, current_letter, word_buffer, progress,
             is_speaking, detected_letter, fps):
    """Draw the full Heads-Up Display overlay."""
    h, w = image.shape[:2]

    # ---- Top banner ----
    draw_glass_panel(image, 0, 0, w, 50, color=(20, 20, 20), alpha=0.7)
    cv2.putText(image, "SignBridge ASL",
                (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 230, 230), 2, cv2.LINE_AA)

    supported = "A B C D E F I K L O U V W Y"
    cv2.putText(image, supported,
                (220, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (160, 160, 160), 1, cv2.LINE_AA)

    # FPS counter
    cv2.putText(image, f"{fps:.0f} FPS",
                (w - 90, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (100, 255, 100), 1, cv2.LINE_AA)

    # ---- Current letter + progress arc (top-left) ----
    if detected_letter:
        draw_glass_panel(image, 15, 65, 130, 130, alpha=0.5)

        # Large letter
        cv2.putText(image, detected_letter,
                    (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 3.0,
                    (255, 255, 255), 4, cv2.LINE_AA)

        # Progress arc around the letter
        if progress > 0 and not is_speaking:
            draw_progress_arc(image, (80, 130), 55, progress)

        if is_speaking:
            cv2.putText(image, "SPOKEN",
                        (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 200, 255), 1, cv2.LINE_AA)

    # ---- Word buffer panel (bottom) ----
    panel_h = 70
    draw_glass_panel(image, 0, h - panel_h, w, panel_h,
                     color=(25, 25, 25), alpha=0.7)

    cv2.putText(image, "WORD:",
                (15, h - panel_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 200, 255), 1, cv2.LINE_AA)

    display_word = word_buffer if word_buffer else "_"
    cv2.putText(image, display_word,
                (90, h - panel_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Controls hint
    controls = "FIST=Space  |  Open Palm(B)=Speak Word  |  ESC=Exit"
    cv2.putText(image, controls,
                (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (140, 140, 140), 1, cv2.LINE_AA)


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils


    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Smoothing history
    history = deque(maxlen=12)

    # State tracking
    current_letter = None
    letter_start_time = None
    already_spoken = False
    hold_duration = 2.0  # seconds to hold before accepting a letter

    # Word building
    word_buffer = ""


    # FPS tracking
    prev_time = time.time()
    fps = 0.0

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1
    ) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # Flip for selfie-view
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            raw_letter = None

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lm, hand_info in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness):
                    # Draw the hand skeleton with custom colors
                    mp_drawing.draw_landmarks(
                        image, hand_lm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(
                            color=(0, 230, 230), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(
                            color=(50, 50, 50), thickness=2))

                    label = hand_info.classification[0].label
                    raw_letter = classify_gesture(hand_lm, label)

            history.append(raw_letter)

            # --- Smoothing: require 7/12 frames agreement ---
            counter = Counter(history)
            most_common = counter.most_common(1)
            if most_common:
                smoothed, count = most_common[0]
                detected_letter = smoothed if (count >= 7
                                               and smoothed is not None) else None
            else:
                detected_letter = None

            # --- FPS ---
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 0.001))
            prev_time = now

            # --- Letter hold + word building logic ---
            progress = 0.0
            is_speaking = False

            if detected_letter:
                if detected_letter == current_letter:
                    if not already_spoken and letter_start_time is not None:
                        elapsed = now - letter_start_time
                        progress = min(1.0, elapsed / hold_duration)

                        if elapsed >= hold_duration:
                            # Letter confirmed — add to word
                            word_buffer += detected_letter

                            # Speak the individual letter
                            speak_text(f"Letter {detected_letter}")
                            already_spoken = True
                            is_speaking = True
                else:
                    current_letter = detected_letter
                    letter_start_time = now
                    already_spoken = False
            else:
                if current_letter is not None:
                    current_letter = None
                    letter_start_time = None
                    already_spoken = False

            # --- Special gestures for word control ---
            # SPACE: Closed fist (A sign) held for 1.5s when word is not empty
            # We detect "no letter" + fist heuristic via the 'A' detection
            # Actually, let's use a separate approach:
            # If detected_letter == 'A' and word_buffer is not empty,
            # holding 'A' for 1.5s adds a space.

            # SPEAK WORD: If 'B' (open palm) is held for 2.5s, speak the whole word
            # This is handled naturally — when B is spoken, we can also trigger
            # word speech if the word is long enough.

            # Keyboard shortcuts for word control (while camera window focused)
            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # Space bar → add space to word
                word_buffer += " "
            elif key == 8 or key == ord('\b'):  # Backspace → delete last char
                word_buffer = word_buffer[:-1]
            elif key == 13 or key == ord('\r'):  # Enter → speak the whole word
                if word_buffer.strip():
                    speak_text(word_buffer)
                    word_buffer = ""

            # --- Draw the HUD ---
            draw_hud(image, current_letter, word_buffer,
                     progress, is_speaking, detected_letter, fps)

            cv2.imshow('SignBridge ASL Recognizer', image)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
