<div align="center">
  
  # 🤟 SignBridge Web-Pro
  **Advanced AI-Powered American Sign Language (ASL) Recognition Engine**

  [![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-WebRTC-FF4B4B.svg)](https://streamlit.io/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00.svg)](https://tensorflow.org)
  [![MediaPipe](https://img.shields.io/badge/Google-MediaPipe-00bfa5.svg)](https://mediapipe.dev)

  *Bridging the communication gap between the Deaf community and the hearing world through real-time AI computer vision and speech synthesis.*

</div>

## 🌟 Overview
SignBridge is a production-grade web application that leverages a custom-trained Deep Learning neural network to translate American Sign Language gestures into text and spoken English in real-time directly through the browser.

Moving beyond simple heuristic math rules, this "Web-Pro" edition introduces a state-of-the-art **TensorFlow Neural Network** built entirely from scratch, utilizing Google's modern MediaPipe Tasks API for flawless, low-latency 3D hand tracking.

## 🚀 Key Features
* **Custom AI Brain**: Powered by a custom-trained TensorFlow Keras model capable of learning and adapting to specific hand shapes.
* **Browser-Native Video (WebRTC)**: High-performance, low-latency video streaming utilizing `streamlit-webrtc`—completely removing the need for local desktop windows.
* **Modular Architecture**: Professionally structured codebase cleanly separating frontend UI, backend ML inference, and utility scripts.
* **Text-to-Speech Engine**: Automatically converts successfully constructed words and phrases into audible speech using an asynchronous text-to-speech engine.
* **Data Collection Studio**: Includes a built-in Data Studio (`data_collector.py`) allowing anyone to capture their own hand data and instantly retrain the AI model for infinite scalability.
* **Premium Glassmorphic UI**: A highly polished, dynamic, and responsive dark-mode user interface designed for accessibility.

## 🛠️ Architecture & Tech Stack
* **Frontend**: Streamlit, Custom HTML/CSS (Glassmorphism)
* **Backend Inference**: TensorFlow / Keras (Sequential Dense Neural Network)
* **Computer Vision**: OpenCV (`cv2`), MediaPipe Tasks API (Vision)
* **Real-Time Streaming**: WebRTC (`streamlit-webrtc`, `aiortc`)
* **Audio Synthesis**: `pyttsx3` (Text-to-Speech)

## 📁 Project Structure
```text
SignBridge/
├── app.py                   # Main Application Orchestrator
├── frontend/                # UI Components & CSS Styling
├── backend/                 # AI Inference & MediaPipe Video Processor
├── utils/                   # Helper scripts (Text-to-Speech)
├── models/                  # Custom Trained .keras Neural Networks
├── data_collector.py        # Studio to capture custom ASL datasets
└── train_model.py           # Deep Learning training pipeline
```

## 💻 Getting Started
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Roastee/SignBridge1.git
   cd SignBridge1
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install tensorflow scikit-learn pandas
   ```
3. **Launch the Engine:**
   ```bash
   streamlit run app.py
   ```

## 🧠 Training Your Own AI
Want to teach the AI a new gesture?
1. Run `python data_collector.py` to capture 3D landmarks of your hand.
2. Run `python train_model.py` to automatically compile a new Keras Neural Network.
3. Refresh the web app to instantly see your new AI in action!
