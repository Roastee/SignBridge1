import cv2
import mediapipe as mp
import csv
import os

# ==========================================
# 1. INITIALIZE MEDIAPIPE & CAMERA
# ==========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

DATA_FILE = "landmark_dataset.csv"

# ==========================================
# 2. CREATE CSV SKELETON
# ==========================================
# We record 21 landmarks. Each has an X, Y, and Z coordinate.
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)

print("\n" + "="*50)
print(" 🚀 SIGNBRIDGE AI DATA COLLECTOR STUDIO 🚀")
print("="*50)
print("INSTRUCTIONS:")
print("1. A new window will open showing your webcam.")
print("2. Hold your hand up and make an ASL letter gesture.")
print("3. Press that specific letter key (A-Z) on your keyboard.")
print("4. Press it multiple times while slightly moving your hand")
print("   to capture different angles for the AI.")
print("5. Aim for ~50-100 captures per letter.")
print("6. Press ESC to exit.")
print("="*50 + "\n")

# Try to find a working camera
cap = cv2.VideoCapture(0)
if not cap.isOpened() or not cap.read()[0]:
    print("Camera 0 failed, trying Camera 1...")
    cap = cv2.VideoCapture(1)

# ==========================================
# 3. MAIN CAPTURE LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret: 
        continue

    # Mirror for selfie-view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process hands
    results = hands.process(rgb)
    landmarks_list = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw skeleton on screen
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- NORMALIZATION ---
            # To ensure the AI isn't confused by where your hand is on the screen,
            # we make all coordinates relative to your wrist (landmark 0)
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z
            
            for lm in hand_landmarks.landmark:
                landmarks_list.extend([
                    lm.x - base_x, 
                    lm.y - base_y, 
                    lm.z - base_z
                ])

    # HUD Elements
    cv2.rectangle(frame, (0, 0), (350, 70), (20, 20, 20), -1)
    cv2.putText(frame, "Press A-Z to capture data", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 118), 2)
    cv2.putText(frame, "Press ESC to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    cv2.imshow("SignBridge Data Studio", frame)
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF

    if key == 27: # ESC key
        break
    elif 97 <= key <= 122: # Lowercase 'a' to 'z'
        if landmarks_list:
            letter = chr(key).upper()
            
            # Save the sequence to CSV
            with open(DATA_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([letter] + landmarks_list)
                
            print(f"✅ Captured 1 sample for: {letter}")
            
            # Visual flash effect for capture
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)
            cv2.imshow("SignBridge Data Studio", frame)
            cv2.waitKey(50)
        else:
            print("⚠️ No hand detected! Please show your hand in the frame.")

cap.release()
cv2.destroyAllWindows()
print("\nData collection session ended. Your data is saved in 'landmark_dataset.csv'.")
