import streamlit as st
import cv2
import subprocess
import streamlit.components.v1 as components
import numpy as np
import tempfile
import os
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks.python import vision
import json
import time
import warnings
from google import genai
from google.genai import types

# --- SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# --- FIX: DISABLE MEDIAPIPE GPU (Prevents EGL/OpenGL Errors in Cloud) ---
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

# Try to use legacy solutions API if available, otherwise use tasks
try:
    from mediapipe import solutions

    USE_LEGACY_API = True
except:
    USE_LEGACY_API = False

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Fallback: Try st.secrets if env var is missing (for local dev)
if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except:
        pass

# ===== CONFIG & SETUP =====
st.set_page_config(
    page_title="Golf Swing Analyzer",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===== CUSTOM CSS - PREMIUM DARK THEME WITH NEON ACCENTS =====
st.markdown(
    """
<style>
    :root {
        --bg-dark: #0a0d12;
        --bg-darker: #050609;
        --accent-blue: #3b82f6;
        --accent-teal: #14b8a6;
        --accent-gold: #d97706;
        --text-primary: #f5f5f5;
        --text-secondary: #d1d5db;
    }
    
    * {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main container */
    .stApp {
        background: linear-gradient(135deg, #0a0d12 0%, #1a1f2e 100%);
    }
    
    /* Sidebar */
    .st-emotion-cache-zt5igj {
        background-color: #050609;
        border-right: 1px solid rgba(59, 130, 246, 0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    h1 {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #3b82f6, #14b8a6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* =======================================
       COMPACT TAB NAVIGATION
       ======================================= */
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #0e121b;
        padding: 4px 4px;
        border-radius: 8px;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 35px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        color: #9ca3af;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        border: 1px solid transparent;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        flex-grow: 1;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(59, 130, 246, 0.1);
        color: #3b82f6;
        border-color: rgba(59, 130, 246, 0.2);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(20, 184, 166, 0.2) 100%);
        color: #ffffff;
        border: 1px solid rgba(59, 130, 246, 0.5);
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #14b8a6);
        color: #ffffff;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 12px 24px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9rem;
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
    }
    
    /* Metrics */
    .metric-box {
        background: rgba(59, 130, 246, 0.05);
        border: 2px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        border-color: #3b82f6;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #d97706;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #d1d5db;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Info boxes */
    .stInfo, .st-emotion-cache-1y4p8pa {
        background: rgba(59, 130, 246, 0.1) !important;
        border-left: 4px solid #3b82f6 !important;
        border-radius: 4px !important;
    }
    
    .stSuccess {
        background: rgba(20, 184, 166, 0.1) !important;
        border-left: 4px solid #14b8a6 !important;
        border-radius: 4px !important;
    }
    
    .stWarning {
        background: rgba(217, 119, 6, 0.1) !important;
        border-left: 4px solid #d97706 !important;
        border-radius: 4px !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border-left: 4px solid #ef4444 !important;
        border-radius: 4px !important;
    }
    
    /* File uploader */
    .uploadedFile {
        background-color: rgba(59, 130, 246, 0.05);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 4px;
    }
    
    /* Video Container Styling */
    .video-card {
        background: #0f1219;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .video-header {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
        border-bottom: 1px solid #1f2937;
        padding-bottom: 10px;
    }
    
    .video-badge {
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 700;
        margin-left: auto;
        letter-spacing: 1px;
    }
    
    .badge-original {
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .badge-ai {
        background: rgba(20, 184, 166, 0.2);
        color: #2dd4bf;
        border: 1px solid rgba(20, 184, 166, 0.3);
    }

    /* Custom Progress Bar Color */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #3b82f6, #14b8a6);
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style='text-align: center; margin-bottom: 2rem; margin-top: -3rem;'>
    <h1 style='margin: 0; font-size: 3rem;'>⛳ GOLF SWING ANALYZER</h1>
    <p style='color: #3b82f6; font-size: 0.9rem; letter-spacing: 2px; text-transform: uppercase; margin-top: 0.5rem;'>TOUR-LEVEL BIOMECHANICAL ANALYSIS</p>
</div>
""",
    unsafe_allow_html=True,
)

# ===== HELPER FUNCTIONS =====


def calculate_angle(a, b, c):
    """Calculate angle between three points (in degrees)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle


def analyze_head_stability(landmarks, frame_width, frame_height):
    """Track head position to detect swaying or dipping"""
    nose = landmarks[0]
    left_ear = landmarks[9]
    right_ear = landmarks[10]

    head_x = (nose.x + left_ear.x + right_ear.x) / 3
    head_y = (nose.y + left_ear.y + right_ear.y) / 3

    horizontal_position = head_x * frame_width

    return {
        "head_x": head_x,
        "head_y": head_y,
        "horizontal_position": horizontal_position,
        "nose": [nose.x, nose.y, nose.z],
    }


def analyze_lead_arm_angle(landmarks):
    """Measure lead arm angle at top of backswing"""
    left_shoulder = [landmarks[11].x, landmarks[11].y, landmarks[11].z]
    left_elbow = [landmarks[13].x, landmarks[13].y, landmarks[13].z]
    left_wrist = [landmarks[15].x, landmarks[15].y, landmarks[15].z]

    right_shoulder = [landmarks[12].x, landmarks[12].y, landmarks[12].z]
    right_elbow = [landmarks[14].x, landmarks[14].y, landmarks[14].z]
    right_wrist = [landmarks[16].x, landmarks[16].y, landmarks[16].z]

    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    return {
        "left_arm_angle": left_arm_angle,
        "right_arm_angle": right_arm_angle,
    }


def trace_swing_path(landmarks, frame):
    """Trace the path of hands during swing"""
    left_wrist = [landmarks[15].x, landmarks[15].y, landmarks[15].z]
    right_wrist = [landmarks[16].x, landmarks[16].y, landmarks[16].z]

    h, w = frame.shape[:2]

    left_wrist_pos = (int(left_wrist[0] * w), int(left_wrist[1] * h))
    right_wrist_pos = (int(right_wrist[0] * w), int(right_wrist[1] * h))

    return {
        "left_wrist_pos": left_wrist_pos,
        "right_wrist_pos": right_wrist_pos,
    }


# --- NEW HELPER FOR RAILWAY COMPATIBILITY ---
def convert_to_h264(input_path, output_path):
    """
    Converts a video file to H.264 format using FFmpeg.
    This ensures the video is playable in web browsers.
    """
    try:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vcodec",
            "libx264",
            "-acodec",
            "aac",
            output_path,
        ]
        # Run ffmpeg, suppressing output unless there is an error
        subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Video conversion failed. Ensure FFmpeg is installed. Error: {e}")
        return False
    except FileNotFoundError:
        st.error("FFmpeg not found. Please install FFmpeg on the server.")
        return False


def process_video(input_path, output_path, progress_callback=None):
    """Process video, save to file, and return metrics. Supports progress callback."""
    cap = cv2.VideoCapture(input_path)

    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 1. Calculate New Dimensions (Max 640px width for speed/memory)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_width = 640
    if orig_width > target_width:
        scale = target_width / orig_width
        width = target_width
        height = int(orig_height * scale)
    else:
        width = orig_width
        height = orig_height

    # 2. Setup Video Writer
    # FIX: Use 'mp4v' for backend processing (works on Linux/Railway without HW accel)
    # We will write to a temporary file first, then convert it.
    temp_raw_path = output_path.replace(".mp4", "_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_raw_path, fourcc, fps, (width, height))

    # MediaPipe Setup
    if USE_LEGACY_API:
        mp_pose = solutions.pose
        mp_drawing = solutions.drawing_utils
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    else:
        raise Exception("Legacy MediaPipe required for this implementation")

    metrics_list = []
    frame_count = 0
    landmarks_detected_count = 0
    path_history = {"left": [], "right": []}

    with pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Update Progress Bar
            if progress_callback and total_frames > 0:
                progress_val = min(frame_count / total_frames, 1.0)
                progress_callback(progress_val)

            # Resize
            frame = cv2.resize(frame, (width, height))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Analyze
            results = pose.process(frame_rgb)
            analyzed_frame = frame.copy()

            if results.pose_landmarks:
                landmarks_detected_count += 1

                # Draw Skeleton
                mp_drawing.draw_landmarks(
                    analyzed_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                # --- METRICS CALCULATION ---
                class SimpleLandmark:
                    def __init__(self, lm):
                        self.x, self.y, self.z = lm.x, lm.y, lm.z

                landmarks_list = [
                    SimpleLandmark(lm) for lm in results.pose_landmarks.landmark
                ]

                # 1. Arm Angles
                arm_data = analyze_lead_arm_angle(landmarks_list)

                # 2. Head Stability
                head_data = analyze_head_stability(landmarks_list, width, height)

                # 3. Path Tracing
                path_data = trace_swing_path(landmarks_list, analyzed_frame)
                path_history["left"].append(path_data["left_wrist_pos"])
                path_history["right"].append(path_data["right_wrist_pos"])

                # Draw Paths
                if len(path_history["left"]) > 1:
                    cv2.polylines(
                        analyzed_frame,
                        [np.array(path_history["left"])],
                        False,
                        (0, 255, 0),
                        2,
                    )
                    cv2.polylines(
                        analyzed_frame,
                        [np.array(path_history["right"])],
                        False,
                        (255, 0, 0),
                        2,
                    )

                # Save Metrics
                metrics_list.append(
                    {
                        "frame": frame_count,
                        "left_arm_angle": arm_data["left_arm_angle"],
                        "right_arm_angle": arm_data["right_arm_angle"],
                        "head_x": head_data["head_x"],
                        "head_y": head_data["head_y"],
                    }
                )

            # Write frame to file
            out.write(analyzed_frame)

    cap.release()
    out.release()

    # --- FIX: CONVERT TO BROWSER COMPATIBLE FORMAT ---
    # Convert the raw 'mp4v' file to 'H.264' so it plays in Chrome/Safari
    convert_to_h264(temp_raw_path, output_path)

    # Clean up the temporary raw file
    if os.path.exists(temp_raw_path):
        os.remove(temp_raw_path)

    return {
        "metrics": metrics_list,
        "total_frames": frame_count,
        "fps": fps,
        "landmarks_detected_count": landmarks_detected_count,
    }


def get_ai_coaching(metrics, user_context):
    """Generate AI coaching insights using Gemini API."""
    try:
        if not metrics:
            return None

        # --- PRE-PROCESSING METRICS ---
        left_arm_angles = [m["left_arm_angle"] for m in metrics]
        right_arm_angles = [m["right_arm_angle"] for m in metrics]

        min_left_arm = np.min(left_arm_angles) if left_arm_angles else 0
        head_x_std = np.std([m["head_x"] for m in metrics])
        head_y_std = np.std([m["head_y"] for m in metrics])

        severity_flags = []
        if min_left_arm < 135:
            severity_flags.append("CRITICAL: Lead Arm Collapse (Chicken Wing)")
        elif min_left_arm < 155:
            severity_flags.append("MODERATE: Lead Arm Soft")

        if head_x_std > 0.08:
            severity_flags.append("CRITICAL: Excessive Lateral Sway")
        elif head_x_std > 0.04:
            severity_flags.append("MODERATE: Minor Head Sway")

        analysis_data = {
            "biometrics": {
                "top_of_backswing_lead_arm_angle": f"{min_left_arm:.1f} degrees (Ideal: >160)",
                "avg_trail_arm_angle": f"{np.mean(right_arm_angles):.1f} degrees",
                "head_sway_factor": f"{head_x_std:.4f} (Lower is better)",
                "head_dip_factor": f"{head_y_std:.4f} (Lower is better)",
            },
            "severity_assessment": severity_flags,
            "user_profile": user_context,
        }

        analysis_json = json.dumps(analysis_data)

        prompt = f"""
        You are an elite PGA Tour Biomechanics Coach. Analyze this golfer's data.
        
        INPUT DATA:
        {analysis_json}

        INSTRUCTIONS:
        1. Compare the golfer's metrics to PGA Tour averages adjusted for their handicap.
        2. Identify the 1–3 most damaging swing faults ("Swing Killers").
        3. CRITICAL REQUIREMENT: Prioritize drills by SEVERITY. If a "CRITICAL" fault is detected, that drill MUST be first.

        RESPONSE RULES:
        - Return VALID JSON ONLY.
        - Do NOT include markdown, explanations, or extra text.

        RESPONSE FORMAT:
        Return valid JSON ONLY with this structure:
        {{
            "summary": "1-sentence executive summary highlighting the primary swing fault.",
            "positives": ["point 1", "point 2"],
            "negatives": ["point 1", "point 2"],
            "drills": [
            {{
                "problem": "Specific swing fault (e.g. Critical Sway)",
                "name": "Drill Name",
                "why": "Why this specific drill fixes the biomechanical issue.",
                "steps": ["Step 1", "Step 2"]
            }}
            ],
            "pro_tip": "One short advanced tip related to the #1 fault"
        }}
        """

        client = genai.Client(api_key=GEMINI_API_KEY)
        max_retries = 3
        base_delay = 2

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    ),
                )
                return json.loads(response.text)

            except Exception as e:
                if "429" in str(e) or "resource exhausted" in str(e).lower():
                    if attempt < max_retries - 1:
                        time.sleep(base_delay * (2**attempt))
                        continue
                    else:
                        return {
                            "error": "AI Coach is busy. Please try again in 1 minute."
                        }
                return {"error": str(e)}

    except Exception as e:
        return {"error": str(e)}


# ===== STREAMLIT UI =====

st.sidebar.markdown(
    """
<style>
    .sidebar-title {
        color: #3b82f6;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(59, 130, 246, 0.2);
    }
</style>
<div class='sidebar-title'>⚙️ SETTINGS</div>
""",
    unsafe_allow_html=True,
)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 ANALYSIS", "📈 METRICS", "🤖 AI COACH", "ℹ️ INFO"])

with tab1:
    st.markdown(
        """
    <div style='margin-bottom: 0.5rem;'>
        <h2 style='color: #3b82f6; margin-bottom: 0; font-size: 1.5rem;'>UPLOAD SWING</h2>
        <p style='color: #d1d5db; margin: 0; font-size: 0.9rem;'>Import your golf swing video for AI-powered analysis</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # --- RECRUITER / DEMO BUTTON ---
    col_demo, col_or = st.columns([1, 0.2])
    if st.button("🚀 LOAD DEMO SWING FILE", width="stretch"):
        st.session_state["use_demo"] = True
        st.rerun()

    uploaded_file = st.file_uploader(
        "Drop your video file here",
        type=["mp4", "mov", "avi", "mkv"],
        help="Supported formats: MP4, MOV, AVI, MKV",
        label_visibility="collapsed",
    )

    # Logic to determine which file to process
    active_video_path = None
    active_filename = "demo_swing.mp4"
    is_demo = False

    if uploaded_file is not None:
        st.session_state["use_demo"] = False  # Override demo mode
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file.flush()
                active_video_path = tmp_file.name
                active_filename = uploaded_file.name
        except Exception as e:
            st.error(f"Upload error: {str(e)}")

    # 2. Check if Demo Mode is active
    elif st.session_state.get("use_demo"):
        if os.path.exists("demo_swing.mp4"):
            active_video_path = "demo_swing.mp4"
            is_demo = True
        else:
            st.error("⚠️ 'demo_swing.mp4' not found in project folder.")

    # --- PROCESS THE VIDEO ---
    if active_video_path:
        # Create a container for the action area
        action_area = st.empty()

        # Render the initial state (Ready message + Button)
        with action_area.container():
            st.markdown(
                f"""
            <div style='background: rgba(20, 184, 166, 0.1); border-left: 4px solid #14b8a6; 
                        padding: 1rem; border-radius: 4px; margin-bottom: 1rem;'>
                <span style='color: #14b8a6; font-weight: 600;'>✓ Video Ready: {active_filename}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )
            analyze_pressed = st.button(
                "▶ ANALYZE SWING", width="stretch", key="analyze_btn"
            )

        if analyze_pressed:
            if not active_video_path:
                st.error("Please upload a video or load the demo first.")
            else:
                # Clear the "Ready" UI immediately
                action_area.empty()

                # Render the Loading UI in the same container space
                with action_area.container():
                    st.markdown(
                        "<div style='margin-bottom: 10px; color: #d1d5db; font-weight: 600;'>INITIALIZING COMPUTER VISION...</div>",
                        unsafe_allow_html=True,
                    )
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        # Create a temp file for the OUTPUT video
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        output_video_path = tfile.name
                        tfile.close()

                        # Define the callback to update the bar
                        def update_progress(val):
                            progress_bar.progress(val)
                            pct = int(val * 100)
                            if pct < 30:
                                msg = f"Detecting Pose Landmarks... {pct}%"
                            elif pct < 60:
                                msg = f"Computing Biomechanics... {pct}%"
                            elif pct < 90:
                                msg = f"Rendering Visual Overlays... {pct}%"
                            else:
                                msg = f"Finalizing... {pct}%"
                            status_text.markdown(
                                f"<span style='color: #3b82f6; font-weight: bold;'>{msg}</span>",
                                unsafe_allow_html=True,
                            )

                        # Process video with callback
                        results = process_video(
                            active_video_path, output_video_path, update_progress
                        )

                        # Store in session state
                        st.session_state.results = results
                        st.session_state.analyzed_video_path = output_video_path
                        st.session_state.uploaded_filename = active_filename
                        st.session_state.coach_cache = None
                        st.session_state.last_context = None

                        st.session_state.should_scroll = True
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")

    # Display results if available
    if "results" in st.session_state:
        # --- AUTO-SCROLL LOGIC ---
        st.markdown("<div id='analysis_results'></div>", unsafe_allow_html=True)

        if st.session_state.get("should_scroll", False):
            components.html(
                """
                <script>
                    setTimeout(function() {
                        const element = window.parent.document.getElementById('analysis_results');
                        if (element) {
                            element.scrollIntoView({behavior: 'smooth', block: 'start'});
                        }
                    }, 100);
                </script>
                """,
                height=0,
                width=0,
            )
            st.session_state.should_scroll = False
        results = st.session_state.results

        # Status Bar
        st.markdown(
            f"""
        <div style='background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3);
                    padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem;'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <p style='color: #d1d5db; font-size: 0.85rem; text-transform: uppercase; margin: 0 0 0.5rem 0; letter-spacing: 1px;'>
                        Analysis Results
                    </p>
                    <p style='color: #f5f5f5; font-size: 1.2rem; margin: 0; font-weight: 600;'>
                        {results.get('total_frames', 0)} Frames Analyzed
                    </p>
                </div>
                <div style='text-align: right;'>
                    <p style='color: #d97706; font-size: 2.5rem; margin: 0; font-weight: 700;'>
                        {results.get('landmarks_detected_count', 0)}
                    </p>
                    <p style='color: #d1d5db; font-size: 0.85rem; margin: 0.5rem 0 0 0; text-transform: uppercase; letter-spacing: 1px;'>
                        Frames with Pose
                    </p>
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # === THEMED VIDEO PLAYBACK SECTION ===
        st.markdown(
            """
        <div style='margin-bottom: 1.5rem;'>
            <h3 style='color: #f5f5f5; margin-top: 0; border-left: 4px solid #3b82f6; padding-left: 10px;'>🎥 SWING REPLAY</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            <div class='video-card'>
                <div class='video-header'>
                     <span style='color: #e5e7eb; font-weight: 600; font-size: 0.9rem;'>RAW FOOTAGE</span>
                     <span class='video-badge badge-original'>SOURCE</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.video(active_video_path)

        with col2:
            st.markdown(
                """
            <div class='video-card'>
                <div class='video-header'>
                     <span style='color: #e5e7eb; font-weight: 600; font-size: 0.9rem;'>COMPUTER VISION OVERLAY</span>
                     <span class='video-badge badge-ai'>AI PROCESSED</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            if "analyzed_video_path" in st.session_state:
                st.video(st.session_state.analyzed_video_path)
            else:
                st.write("Analysis not available")

        # Key Metrics Section
        st.markdown(
            """
        <h2 style='color: #3b82f6; margin-top: 2rem; margin-bottom: 1.5rem;'>KEY METRICS</h2>
        """,
            unsafe_allow_html=True,
        )

        metrics = results["metrics"]

        # Initialize variables with defaults
        avg_left_arm = 0
        avg_right_arm = 0
        head_x_std = 0

        if metrics:
            avg_left_arm = np.mean([m["left_arm_angle"] for m in metrics])
            avg_right_arm = np.mean([m["right_arm_angle"] for m in metrics])
            head_x_std = np.std([m["head_x"] for m in metrics])

            # Display metrics in a grid
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(
                    f"""
                <div class='metric-box'>
                    <div class='metric-label'>Lead Arm Angle</div>
                    <div class='metric-value'>{avg_left_arm:.1f}°</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                <div class='metric-box'>
                    <div class='metric-label'>Trail Arm Angle</div>
                    <div class='metric-value'>{avg_right_arm:.1f}°</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f"""
                <div class='metric-box'>
                    <div class='metric-label'>Head Stability (σ)</div>
                    <div class='metric-value'>{head_x_std:.4f}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col4:
                st.markdown(
                    f"""
                <div class='metric-box'>
                    <div class='metric-label'>Total Frames</div>
                    <div class='metric-value'>{results["total_frames"]}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Recommendations Section
        st.markdown(
            """
        <h2 style='color: #14b8a6; margin-top: 2rem; margin-bottom: 1.5rem;'>INSIGHTS</h2>
        """,
            unsafe_allow_html=True,
        )

        recommendations = []

        if avg_left_arm < 150:
            recommendations.append(
                (
                    "⚠️ LEAD ARM EXTENSION",
                    "Extend your lead arm more at the top of the backswing to maximize power transfer.",
                    "warning",
                )
            )
        else:
            recommendations.append(
                (
                    "✓ LEAD ARM EXTENSION",
                    "Excellent arm extension at the top of the backswing.",
                    "success",
                )
            )

        if head_x_std > 0.05:
            recommendations.append(
                (
                    "⚠️ HEAD STABILITY",
                    "Reduce lateral head movement (swaying) to improve consistency.",
                    "warning",
                )
            )
        else:
            recommendations.append(
                (
                    "✓ HEAD STABILITY",
                    "Great head position stability throughout the swing.",
                    "success",
                )
            )

        if avg_right_arm < 90:
            recommendations.append(
                (
                    "⚠️ TRAIL ARM ANGLE",
                    "Your trail arm appears overly extended. Work on maintaining proper angles.",
                    "warning",
                )
            )
        else:
            recommendations.append(
                (
                    "✓ TRAIL ARM ANGLE",
                    "Good trail arm positioning and control.",
                    "success",
                )
            )

        for title, msg, rec_type in recommendations:
            if rec_type == "warning":
                st.markdown(
                    f"""
                <div style='background: rgba(217, 119, 6, 0.1); border-left: 4px solid #d97706;
                            padding: 1rem; border-radius: 4px; margin-bottom: 1rem;'>
                    <p style='color: #d97706; font-weight: 600; margin: 0 0 0.5rem 0; text-transform: uppercase; letter-spacing: 0.5px;'>
                        {title}
                    </p>
                    <p style='color: #d1d5db; margin: 0;'>{msg}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div style='background: rgba(20, 184, 166, 0.1); border-left: 4px solid #14b8a6;
                            padding: 1rem; border-radius: 4px; margin-bottom: 1rem;'>
                    <p style='color: #14b8a6; font-weight: 600; margin: 0 0 0.5rem 0; text-transform: uppercase; letter-spacing: 0.5px;'>
                        {title}
                    </p>
                    <p style='color: #d1d5db; margin: 0;'>{msg}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

with tab2:
    st.markdown(
        """
    <h2 style='color: #3b82f6; margin-bottom: 1.5rem;'>DETAILED METRICS</h2>
    """,
        unsafe_allow_html=True,
    )

    if "results" in st.session_state:
        metrics = st.session_state.results["metrics"]

        if metrics:
            import pandas as pd

            df_metrics = pd.DataFrame(metrics)

            st.markdown(
                """
            <h3 style='color: #d1d5db; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 1rem;'>
                FRAME-BY-FRAME ANALYSIS
            </h3>
            """,
                unsafe_allow_html=True,
            )
            st.dataframe(df_metrics, width="stretch")

            st.markdown(
                """
            <h3 style='color: #d1d5db; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin: 2rem 0 1rem 0;'>
                TREND ANALYSIS
            </h3>
            """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
                <p style='color: #d1d5db; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;'>
                    ARM ANGLES OVER TIME
                </p>
                """,
                    unsafe_allow_html=True,
                )
                st.line_chart(df_metrics[["left_arm_angle", "right_arm_angle"]])

            with col2:
                st.markdown(
                    """
                <p style='color: #d1d5db; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;'>
                    HEAD POSITION OVER TIME
                </p>
                """,
                    unsafe_allow_html=True,
                )
                st.line_chart(df_metrics[["head_x", "head_y"]])
        else:
            st.markdown(
                """
            <div style='background: rgba(217, 119, 6, 0.1); border-left: 4px solid #d97706;
                        padding: 1.5rem; border-radius: 4px; text-align: center;'>
                <p style='color: #d97706; font-weight: 600; margin: 0 0 0.5rem 0; text-transform: uppercase; letter-spacing: 0.5px;'>
                    NO DATA DETECTED
                </p>
                <p style='color: #d1d5db; margin: 0;'>No pose landmarks detected. Try with a clearer video or better lighting.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
        <div style='background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6;
                    padding: 1.5rem; border-radius: 4px; text-align: center;'>
            <p style='color: #3b82f6; font-weight: 600; margin: 0 0 0.5rem 0; text-transform: uppercase; letter-spacing: 0.5px;'>
                WAITING FOR INPUT
            </p>
            <p style='color: #d1d5db; margin: 0;'>Upload and analyze a video in the Analysis tab to view metrics.</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

with tab3:
    st.markdown(
        """
        <div style='margin-bottom: 2rem;'>
            <h2 style='color: #14b8a6; margin-bottom: 0.5rem;'>🤖 AI SWING COACH</h2>
            <p style='color: #d1d5db; margin: 0;'>PGA-level insights powered by Gemini 2.5 Flash</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize Session State for Cache
    if "coach_cache" not in st.session_state:
        st.session_state.coach_cache = None
    if "last_context" not in st.session_state:
        st.session_state.last_context = None
    if "should_scroll_coach" not in st.session_state:
        st.session_state.should_scroll_coach = False

    # --- USER CONTEXT INPUTS ---
    with st.container():
        st.markdown(
            "<div style='background: rgba(59, 130, 246, 0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(59, 130, 246, 0.2); margin-bottom: 20px;'>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            handicap = st.selectbox(
                "Your Handicap",
                ["Pro (+5 to 0)", "Low (1-9)", "Mid (10-19)", "High (20+)", "Beginner"],
                index=2,
            )
        with c2:
            miss_type = st.selectbox(
                "Common Miss",
                [
                    "Slice (Right)",
                    "Hook (Left)",
                    "Fat/Chunk",
                    "Thin/Top",
                    "Inconsistent",
                ],
                index=0,
            )
        with c3:
            club_used = st.selectbox("Club Used", ["Driver", "Iron", "Wedge"], index=1)
        st.markdown("</div>", unsafe_allow_html=True)

    # Generate Button
    if st.button("✨ GENERATE COACHING PLAN", width="stretch"):

        if "results" not in st.session_state:
            st.warning("Analyze a swing first.")
        else:
            # Create a unique context key to check if inputs changed
            current_context = f"{handicap}-{miss_type}-{club_used}-{st.session_state.results.get('landmarks_detected_count', 0)}"

            # Check if we can use the cache
            if (
                st.session_state.coach_cache
                and st.session_state.last_context == current_context
            ):
                st.success("Loaded from cache (No API usage)")
                st.session_state.should_scroll_coach = (
                    True  # Trigger scroll on cache hit too
                )
            else:
                # No cache or new context -> Call API
                with st.spinner("Consulting PGA biomechanics database..."):
                    st.session_state.coach_response = get_ai_coaching(
                        st.session_state.results["metrics"],
                        {
                            "handicap": handicap,
                            "common_miss": miss_type,
                            "club": club_used,
                        },
                    )
                    st.session_state.coach_cache = st.session_state.coach_response
                    st.session_state.last_context = current_context
                    st.session_state.should_scroll_coach = (
                        True  # Trigger scroll on new gen
                    )

    # --- DISPLAY RESULTS (Outside button so it persists) ---
    if st.session_state.coach_cache:
        coach_response = st.session_state.coach_cache

        # 1. THE ANCHOR for scrolling
        st.markdown("<div id='coach_results'></div>", unsafe_allow_html=True)

        # 2. AUTO-SCROLL LOGIC
        if st.session_state.should_scroll_coach:
            components.html(
                """
                <script>
                    setTimeout(function() {
                        const element = window.parent.document.getElementById('coach_results');
                        if (element) {
                            element.scrollIntoView({behavior: 'smooth', block: 'start'});
                        }
                    }, 100);
                </script>
                """,
                height=0,
                width=0,
            )
            st.session_state.should_scroll_coach = False

        if "error" in coach_response:
            st.error(f"AI Error: {coach_response['error']}")
        else:
            # 1. Executive Summary
            st.markdown(
                f"""
            <div style='background: linear-gradient(90deg, rgba(20, 184, 166, 0.2), rgba(59, 130, 246, 0.2)); 
                        padding: 20px; border-radius: 12px; border-left: 5px solid #14b8a6; margin-bottom: 25px;'>
                <h3 style='margin:0; color: #f5f5f5; font-size: 1.2rem;'>🏌️ COACH'S VERDICT</h3>
                <p style='margin: 10px 0 0 0; color: #d1d5db; font-size: 1.1rem; font-style: italic;'>
                    "{coach_response.get('summary', 'Analysis complete.')}"
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            col_good, col_bad = st.columns(2)

            # 2. What you did well
            with col_good:
                st.markdown(
                    "<h4 style='color: #14b8a6;'>✅ STRENGTHS</h4>",
                    unsafe_allow_html=True,
                )
                for item in coach_response.get("positives", []):
                    st.markdown(
                        f"<div style='background: rgba(20, 184, 166, 0.1); padding: 10px; border-radius: 6px; margin-bottom: 8px; border: 1px solid rgba(20, 184, 166, 0.2);'>{item}</div>",
                        unsafe_allow_html=True,
                    )

            # 3. What needs work
            with col_bad:
                st.markdown(
                    "<h4 style='color: #ef4444;'>⚠️ OPPORTUNITIES</h4>",
                    unsafe_allow_html=True,
                )
                for item in coach_response.get("negatives", []):
                    st.markdown(
                        f"<div style='background: rgba(239, 68, 68, 0.1); padding: 10px; border-radius: 6px; margin-bottom: 8px; border: 1px solid rgba(239, 68, 68, 0.2);'>{item}</div>",
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

            # 4. The Drill Card
            drills = coach_response.get("drills", [])[:3]
            if drills:
                drill = drills[0]  # Display the top priority drill
                drill_name = drill.get("name", "Custom Drill")
                drill_why = drill.get("why", "Improves swing mechanics.")
                drill_steps = drill.get("steps", [])
                drill_problem = drill.get("problem", "Swing fault")

                steps_html = ""
                for i, step in enumerate(drill_steps):
                    steps_html += f"<li style='color: #d1d5db; margin-bottom: 10px; display: flex; align-items: flex-start;'><span style='background-color: #3b82f6; color: white; border-radius: 50%; width: 20px; height: 20px; display: flex; justify-content: center; align-items: center; font-size: 0.8rem; font-weight: bold; margin-right: 12px; flex-shrink: 0;'>{i+1}</span><span style='line-height: 1.5; margin-top: -2px;'>{step}</span></li>"

                raw_html = f"""
                <div style='background-color: #0f172a; border: 1px solid #3b82f6; border-radius: 12px; overflow: hidden; margin-top: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5); font-family: sans-serif;'>
                    <div style='background-color: #3b82f6; padding: 15px 20px; border-bottom: 1px solid #2563eb;'>
                        <div style='display: flex; align-items: center;'>
                            <span style='font-size: 1.2rem; margin-right: 10px;'>🛠️</span>
                            <h4 style='margin: 0; color: white; font-size: 1.1rem; font-weight: 600; letter-spacing: 0.5px;'>RECOMMENDED DRILL: {drill_name}</h4>
                        </div>
                         <div style='font-size: 0.8rem; color: rgba(255,255,255,0.8); margin-top: 4px;'>Addressing: {drill_problem}</div>
                    </div>
                    <div style='padding: 20px;'>
                        <div style='margin-bottom: 25px; background: rgba(59, 130, 246, 0.1); padding: 15px; border-radius: 8px; border-left: 3px solid #3b82f6;'>
                            <div style='color: #93c5fd; font-weight: 700; font-size: 0.8rem; letter-spacing: 1px; margin-bottom: 5px; text-transform: uppercase;'>Why This Works</div>
                            <div style='color: #e2e8f0; font-size: 0.95rem; line-height: 1.6;'>{drill_why}</div>
                        </div>
                        <div>
                            <div style='color: #93c5fd; font-weight: 700; font-size: 0.8rem; letter-spacing: 1px; margin-bottom: 15px; text-transform: uppercase;'>Instructions</div>
                            <ul style='list-style-type: none; padding-left: 0; margin: 0;'>
                                {steps_html}
                            </ul>
                        </div>
                    </div>
                </div>
                """

                clean_html = raw_html.replace("\n", "")
                st.markdown(clean_html, unsafe_allow_html=True)

            st.markdown(
                f"""
            <div style='margin-top: 20px; text-align: center; color: #d97706; font-weight: bold; font-size: 0.9rem; letter-spacing: 1px; border: 1px dashed #d97706; padding: 10px; border-radius: 8px;'>
                PRO TIP: {coach_response.get('pro_tip', '')}
            </div>
            """,
                unsafe_allow_html=True,
            )

with tab4:
    st.markdown(
        """
    <h2 style='color: #3b82f6; margin-bottom: 1.5rem;'>ABOUT THIS ANALYZER</h2>
    <div style='margin-bottom: 2rem;'>
        <h3 style='color: #d97706; font-size: 1.1rem; margin-bottom: 1rem;'>CAPABILITIES</h3>
        <div style='background: rgba(59, 130, 246, 0.05); border-left: 4px solid #3b82f6; padding: 1rem; margin-bottom: 1rem; border-radius: 4px;'>
            <p style='color: #3b82f6; font-weight: 600; margin: 0 0 0.5rem 0; text-transform: uppercase; letter-spacing: 0.5px;'>
                📊 SKELETON TRACKING
            </p>
            <p style='color: #d1d5db; margin: 0;'>Real-time pose detection with 33 body landmarks tracked using MediaPipe's advanced computer vision algorithms.</p>
        </div>
        <div style='background: rgba(217, 119, 6, 0.05); border-left: 4px solid #d97706; padding: 1rem; margin-bottom: 1rem; border-radius: 4px;'>
            <p style='color: #d97706; font-weight: 600; margin: 0 0 0.5rem 0; text-transform: uppercase; letter-spacing: 0.5px;'>
                👤 HEAD STABILITY
            </p>
            <p style='color: #d1d5db; margin: 0;'>Detects lateral head movement and vertical dipping throughout your swing. Measures position variance and consistency.</p>
        </div>
        <div style='background: rgba(20, 184, 166, 0.05); border-left: 4px solid #14b8a6; padding: 1rem; margin-bottom: 1rem; border-radius: 4px;'>
            <p style='color: #14b8a6; font-weight: 600; margin: 0 0 0.5rem 0; text-transform: uppercase; letter-spacing: 0.5px;'>
                💪 ARM ANGLE ANALYSIS
            </p>
            <p style='color: #d1d5db; margin: 0;'>Measures lead and trail arm angles at key positions. Identifies power leaks from improper arm mechanics.</p>
        </div>
        <div style='background: rgba(59, 130, 246, 0.05); border-left: 4px solid #3b82f6; padding: 1rem; margin-bottom: 1rem; border-radius: 4px;'>
            <p style='color: #3b82f6; font-weight: 600; margin: 0 0 0.5rem 0; text-transform: uppercase; letter-spacing: 0.5px;'>
                🎯 SWING PATH TRACING
            </p>
            <p style='color: #d1d5db; margin: 0;'>Visualizes hand trajectory throughout the swing. Shows swing plane and movement patterns.</p>
        </div>
    </div>
    <h3 style='color: #d97706; font-size: 1.1rem; margin-bottom: 1rem;'>TECHNOLOGY STACK</h3>
    <div style='background: rgba(10, 13, 18, 0.8); border: 1px solid rgba(59, 130, 246, 0.2); padding: 1.5rem; border-radius: 8px; font-family: monospace;'>
        <p style='color: #3b82f6; margin: 0.5rem 0;'>• <span style='color: #d1d5db;'>Framework:</span> Streamlit 1.28+</p>
        <p style='color: #d97706; margin: 0.5rem 0;'>• <span style='color: #d1d5db;'>Vision:</span> OpenCV 4.8+</p>
        <p style='color: #14b8a6; margin: 0.5rem 0;'>• <span style='color: #d1d5db;'>Pose Detection:</span> MediaPipe 0.10.5</p>
        <p style='color: #3b82f6; margin: 0.5rem 0;'>• <span style='color: #d1d5db;'>Computation:</span> NumPy, Pandas</p>
    </div>
    <h3 style='color: #d97706; font-size: 1.1rem; margin: 2rem 0 1rem 0;'>DATA PRIVACY</h3>
    <p style='color: #d1d5db;'>All video processing happens locally on your machine. No data is sent to external servers. Your swing analysis remains private.</p>
    <h3 style='color: #d97706; font-size: 1.1rem; margin: 2rem 0 1rem 0;'>GETTING STARTED</h3>
    <ol style='color: #d1d5db;'>
        <li>Record your golf swing from a side angle</li>
        <li>Upload the video in the Analysis tab</li>
        <li>Click ANALYZE SWING to process</li>
        <li>Review metrics and insights</li>
        <li>Use data to refine your technique</li>
    </ol>
    """,
        unsafe_allow_html=True,
    )

# Footer
st.markdown(
    """
<div style='margin-top: 3rem; padding-top: 2rem; border-top: 1px solid rgba(59, 130, 246, 0.1); text-align: center;'>
    <p style='color: #d1d5db; font-size: 0.8rem; margin: 0;'>
        ⛳ GOLF SWING ANALYZER v1.0 | Powered by AI
    </p>
    <p style='color: #6b7280; font-size: 0.75rem; margin: 0.5rem 0 0 0;'>
        Local processing • Privacy-first • Tour-level analytics
    </p>
</div>
""",
    unsafe_allow_html=True,
)
