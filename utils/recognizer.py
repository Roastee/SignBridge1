import math
import numpy as np

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
