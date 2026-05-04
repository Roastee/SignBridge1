import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore

print("="*50)
print(" 🧠 SIGNBRIDGE AI TRAINING STUDIO 🧠")
print("="*50)

# ==========================================
# 1. LOAD & PREPARE DATA
# ==========================================
print("Loading dataset...")
try:
    df = pd.read_csv("landmark_dataset.csv")
except FileNotFoundError:
    print("❌ Error: 'landmark_dataset.csv' not found. Please run data_collector.py first.")
    exit()

# Features (X) are the 63 mathematical coordinates
X = df.iloc[:, 1:].values  
# Labels (y) are the letters (A, B, C...)
y = df.iloc[:, 0].values   

# ==========================================
# 2. ENCODE LABELS
# ==========================================
# The neural network only understands numbers, not letters.
# We convert ['A', 'B', 'A'] into [0, 1, 0]
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Save the translation dictionary so the main app can decode the numbers back to letters
classes = encoder.classes_.tolist()
with open("classes.json", "w") as f:
    json.dump(classes, f)
print(f"Detected {len(classes)} unique gestures: {classes}")

# Split into 80% training data, 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# ==========================================
# 3. BUILD THE NEURAL NETWORK
# ==========================================
print("Architecting neural network...")
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2), # Prevents the AI from memorizing the data (Overfitting)
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(classes), activation='softmax') # Outputs a percentage probability for each letter
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# ==========================================
# 4. TRAIN THE BRAIN
# ==========================================
print("\nInitiating Deep Learning sequence... 🚀\n")
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# ==========================================
# 5. EVALUATE & EXPORT
# ==========================================
loss, accuracy = model.evaluate(X_test, y_test)
print("\n" + "="*50)
print(f"🎯 FINAL TEST ACCURACY: {accuracy * 100:.2f}%")
print("="*50)

model.save("sign_model.keras")
print("\n✅ AI Model successfully exported as 'sign_model.keras'")
print("✅ Decoder ring successfully exported as 'classes.json'")
print("You are now ready to integrate this brain into app.py!")
