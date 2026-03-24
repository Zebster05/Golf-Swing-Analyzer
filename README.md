# 🏌️‍♂️ AI Golf Swing Analyzer

[![Deployed on Railway](https://railway.app/button.svg)](https://railway.app)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose_Estimation-orange.svg)](https://developers.google.com/mediapipe)
[![Gemini API](https://img.shields.io/badge/AI-Google_Gemini_Pro-teal.svg)](https://deepmind.google/technologies/gemini/)

A full-stack computer vision and generative AI web application that analyzes amateur golf swings and provides PGA-style, frame-by-frame coaching feedback. 

**[👉 Try the Live Demo Here](https://golf-swing-analyzer.up.railway.app/)**

---

## ✨ App Showcase

### 1. 🎥 Video Analysis
Upload your swing and watch the AI instantly map 33 skeletal landmarks. Compare your original video side-by-side with the biomechanical overlay, complete with slow-motion playback capabilities.
![Swing Analysis Demo](analysis.gif)
*(Note: Replace `analysis.gif` with a 3-5 second clip of the video playback tab)*

### 2. 📊 Biomechanical Metrics
Dive into the hard data. This section calculates real-time vector geometry to display your lead arm extension, trail arm bend, and head sway across every phase of your swing.
![Metrics Demo](metrics.gif)
*(Note: Replace `metrics.gif` with a 3-5 second clip of your data/charts tab)*

### 3. 🤖 AI Coaching Report
Raw data meets Generative AI. The Google Gemini model ingests your specific swing metrics and generates a personalized, natural-language coaching report highlighting your strengths and areas for improvement.
![AI Coach Demo](ai-coach.gif)
*(Note: Replace `ai-coach.gif` with a 3-5 second clip of the AI generating the text report)*

---

## 🚀 Core Features
* **Biomechanical Tracking:** Utilizes Google's MediaPipe Pose to track 33 distinct skeletal landmarks at 30+ FPS.
* **Dynamic Angle Calculation:** Employs vector mathematics to compute real-time metrics during the backswing, impact, and follow-through.
* **Optimized Video Pipeline:** Features a custom OpenCV transcoding engine that resizes inputs (e.g., 1080p to 640p) and encodes to the open-source VP8/WebM format, reducing rendering latency by ~85% and ensuring cross-browser compatibility.
* **Session State Caching:** Built with Streamlit's session state to cache ML inference results, preventing redundant processing and improving UI responsiveness.

## 🏗️ System Architecture & Deployment
1. **Input:** User uploads an MP4/MOV video file via the Streamlit UI.
2. **Vision Processing:** OpenCV extracts frames; MediaPipe analyzes pose landmarks and draws skeletal overlays.
3. **Data Extraction:** NumPy calculates geometric angles (dot products of vectors) at key swing phases.
4. **LLM Synthesis:** The numerical data is packaged into an intelligent prompt and sent to Google Gemini Pro.
5. **Output:** The app renders a side-by-side WebM video comparison alongside the AI-generated coaching report.

**Deployment (Railway):** This app is containerized and deployed on Railway. It utilizes an `apt.txt` file to install system-level Linux video drivers (`libgl1`, `libglib2.0-0`) required by OpenCV in a headless server environment, and a `Procfile` to dynamically bind Streamlit to Railway's assigned `$PORT`.

## 💻 Tech Stack
* **Frontend/UI:** Streamlit
* **Computer Vision:** OpenCV (`opencv-python-headless`), MediaPipe Pose
* **Data & Math:** NumPy, Pandas
* **Generative AI:** Google GenAI SDK (Gemini)
* **Cloud Infrastructure:** Railway (PaaS)

## 🛠️ Local Installation & Setup

Want to run this project locally? Follow these steps:
**1. Clone the repository**
```bash
git clone [https://github.com/](https://github.com/)[YOUR-GITHUB-USERNAME]/[YOUR-REPO-NAME].git
cd [YOUR-REPO-NAME]
```
2. Create a virtual environment (Requires Python 3.11)

```Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies
```Bash
pip install -r requirements.txt
```
4. Set up your Gemini API Key
Create a .streamlit/secrets.toml file in the root directory and add your API key. (Note: In production on Railway, this is handled via Environment Variables).

```Ini, TOML
GEMINI_API_KEY = "your_actual_api_key_here"
```

5. Run the app

```Bash
streamlit run app.py
```



