import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks.python import vision
import json

# Try to use legacy solutions API if available, otherwise use tasks
try:
    from mediapipe import solutions

    USE_LEGACY_API = True
except:
    USE_LEGACY_API = False

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
    
    /* Tabs - Professional Tab Bar */
    .st-emotion-cache-13ln4kf {
        gap: 1rem;
        background: linear-gradient(90deg, rgba(59, 130, 246, 0.05) 0%, rgba(20, 184, 166, 0.05) 100%);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid rgba(59, 130, 246, 0.15);
        margin-bottom: 2rem;
    }
    
    .st-emotion-cache-6qob1r button[kind="secondary"] {
        background-color: transparent;
        color: #d1d5db;
        border: none;
        border-bottom: 2px solid transparent;
        padding: 12px 20px;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .st-emotion-cache-6qob1r button[kind="secondary"]:hover {
        color: #3b82f6;
        background-color: rgba(59, 130, 246, 0.1);
        border-radius: 8px;
    }
    
    .st-emotion-cache-6qob1r button[kind="secondary"][aria-selected="true"] {
        color: #ffffff;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(20, 184, 166, 0.15));
        border-bottom: 2px solid #3b82f6;
        box-shadow: inset 0 -2px 0 0 #3b82f6;
        border-radius: 8px;
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
    
    /* Divider */
    .st-emotion-cache-z5fcqf {
        border-color: rgba(59, 130, 246, 0.2);
    }
    
    /* Text */
    body {
        color: #f5f5f5;
        background-color: #0a0d12;
    }
    
    /* Dataframe */
    .st-emotion-cache-1wiy60d {
        background-color: rgba(59, 130, 246, 0.05);
    }
    
    /* Sliders */
    .st-emotion-cache-16idsys p {
        color: #d1d5db;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style='text-align: center; margin-bottom: 2rem;'>
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


def draw_landmarks(image, landmarks):
    """Draw pose landmarks on image using OpenCV"""
    h, w = image.shape[:2]

    # Draw connections
    connections = [
        (11, 13),
        (13, 15),  # Left arm
        (12, 14),
        (14, 16),  # Right arm
        (11, 12),  # Shoulders
        (11, 23),
        (12, 24),  # Torso
        (23, 24),  # Hips
        (23, 25),
        (24, 26),  # Legs
        (25, 27),
        (26, 28),  # Lower legs
    ]

    # Convert landmarks to pixel coordinates
    points = {}
    for idx, landmark in enumerate(landmarks):
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        points[idx] = (x, y)
        # Draw landmark circles
        cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

    # Draw connections
    for start, end in connections:
        if start in points and end in points:
            cv2.line(image, points[start], points[end], (0, 255, 0), 2)

    return image, points


def analyze_head_stability(landmarks, frame_width, frame_height):
    """Track head position to detect swaying or dipping"""
    # Landmark indices: 0=nose, 9=left_ear, 10=right_ear
    nose = landmarks[0]
    left_ear = landmarks[9]
    right_ear = landmarks[10]

    # Calculate head center
    head_x = (nose.x + left_ear.x + right_ear.x) / 3
    head_y = (nose.y + left_ear.y + right_ear.y) / 3

    # Check for excessive lateral movement (sway)
    horizontal_position = head_x * frame_width

    return {
        "head_x": head_x,
        "head_y": head_y,
        "horizontal_position": horizontal_position,
        "nose": [nose.x, nose.y, nose.z],
        "left_ear": [left_ear.x, left_ear.y, left_ear.z],
        "right_ear": [right_ear.x, right_ear.y, right_ear.z],
    }


def analyze_lead_arm_angle(landmarks):
    """Measure lead arm angle at top of backswing"""
    # Landmark indices for arms
    # Left: 11=shoulder, 13=elbow, 15=wrist
    # Right: 12=shoulder, 14=elbow, 16=wrist

    left_shoulder = [landmarks[11].x, landmarks[11].y, landmarks[11].z]
    left_elbow = [landmarks[13].x, landmarks[13].y, landmarks[13].z]
    left_wrist = [landmarks[15].x, landmarks[15].y, landmarks[15].z]

    right_shoulder = [landmarks[12].x, landmarks[12].y, landmarks[12].z]
    right_elbow = [landmarks[14].x, landmarks[14].y, landmarks[14].z]
    right_wrist = [landmarks[16].x, landmarks[16].y, landmarks[16].z]

    # Calculate arm angles
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    return {
        "left_arm_angle": left_arm_angle,
        "right_arm_angle": right_arm_angle,
        "left_shoulder": left_shoulder,
        "left_elbow": left_elbow,
        "left_wrist": left_wrist,
        "right_shoulder": right_shoulder,
        "right_elbow": right_elbow,
        "right_wrist": right_wrist,
    }


def trace_swing_path(landmarks, frame):
    """Trace the path of hands during swing"""
    # Landmark indices: 15=left_wrist, 16=right_wrist
    left_wrist = [landmarks[15].x, landmarks[15].y, landmarks[15].z]
    right_wrist = [landmarks[16].x, landmarks[16].y, landmarks[16].z]

    h, w = frame.shape[:2]

    left_wrist_pos = (int(left_wrist[0] * w), int(left_wrist[1] * h))
    right_wrist_pos = (int(right_wrist[0] * w), int(right_wrist[1] * h))

    return {
        "left_wrist": left_wrist,
        "right_wrist": right_wrist,
        "left_wrist_pos": left_wrist_pos,
        "right_wrist_pos": right_wrist_pos,
    }


def process_video(video_path):
    """Process video and return analyzed frames with metrics"""
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Validate video properties
    if width == 0 or height == 0:
        raise Exception("Invalid video dimensions")
    if fps == 0:
        fps = 30  # Default fallback

    print(f"[DEBUG] Video loaded: {total_frames} frames, {width}x{height} @ {fps} fps")

    original_frames = []
    analyzed_frames = []
    metrics_list = []

    frame_count = 0
    path_history = {"left": [], "right": []}
    landmarks_detected_count = 0

    # Use legacy API if available
    if USE_LEGACY_API:
        mp_pose = solutions.pose
        mp_drawing = solutions.drawing_utils

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        ) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Convert to RGB (don't flip yet, process original orientation)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process pose
                results = pose.process(frame_rgb)

                # Store original frame
                original_frames.append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

                # Create analyzed frame
                analyzed_frame = frame_rgb.copy()

                # Check if landmarks were detected (match reference code exactly)
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    landmarks_detected_count += 1

                    # Draw skeleton using MediaPipe's drawing utilities
                    mp_drawing.draw_landmarks(
                        analyzed_frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

                    # Convert landmarks to match our expected format
                    class SimpleLandmark:
                        def __init__(self, lm):
                            self.x = lm.x
                            self.y = lm.y
                            self.z = lm.z

                    landmarks_list = [SimpleLandmark(lm) for lm in landmarks]

                    # Analyze head stability
                    head_data = analyze_head_stability(landmarks_list, width, height)

                    # Draw head tracking circle
                    head_pos = (
                        int(head_data["head_x"] * width),
                        int(head_data["head_y"] * height),
                    )
                    cv2.circle(analyzed_frame, head_pos, 10, (0, 255, 255), 2)
                    cv2.putText(
                        analyzed_frame,
                        "HEAD",
                        (head_pos[0] - 20, head_pos[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )

                    # Analyze lead arm angle
                    arm_data = analyze_lead_arm_angle(landmarks_list)

                    # Draw arm angle on frame
                    left_angle_text = f"L-Arm: {arm_data['left_arm_angle']:.1f}°"
                    right_angle_text = f"R-Arm: {arm_data['right_arm_angle']:.1f}°"

                    cv2.putText(
                        analyzed_frame,
                        left_angle_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 100, 100),
                        2,
                    )
                    cv2.putText(
                        analyzed_frame,
                        right_angle_text,
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (100, 100, 255),
                        2,
                    )

                    # Trace swing path
                    path_data = trace_swing_path(landmarks_list, analyzed_frame)
                    path_history["left"].append(path_data["left_wrist_pos"])
                    path_history["right"].append(path_data["right_wrist_pos"])

                    # Draw hand paths
                    if len(path_history["left"]) > 1:
                        for i in range(1, len(path_history["left"])):
                            cv2.line(
                                analyzed_frame,
                                path_history["left"][i - 1],
                                path_history["left"][i],
                                (0, 255, 0),
                                2,
                            )

                    if len(path_history["right"]) > 1:
                        for i in range(1, len(path_history["right"])):
                            cv2.line(
                                analyzed_frame,
                                path_history["right"][i - 1],
                                path_history["right"][i],
                                (255, 0, 0),
                                2,
                            )

                    # Draw hand position circles
                    cv2.circle(
                        analyzed_frame, path_data["left_wrist_pos"], 5, (0, 255, 0), -1
                    )
                    cv2.circle(
                        analyzed_frame, path_data["right_wrist_pos"], 5, (255, 0, 0), -1
                    )

                    # Compile metrics
                    metrics = {
                        "frame": frame_count,
                        "head_x": head_data["head_x"],
                        "head_y": head_data["head_y"],
                        "left_arm_angle": arm_data["left_arm_angle"],
                        "right_arm_angle": arm_data["right_arm_angle"],
                        "left_wrist_x": path_data["left_wrist"][0],
                        "left_wrist_y": path_data["left_wrist"][1],
                        "right_wrist_x": path_data["right_wrist"][0],
                        "right_wrist_y": path_data["right_wrist"][1],
                    }
                    metrics_list.append(metrics)

                analyzed_frames.append(cv2.cvtColor(analyzed_frame, cv2.COLOR_RGB2BGR))
    else:
        raise Exception(
            "Legacy MediaPipe API not available. Please downgrade to mediapipe<0.10"
        )

    cap.release()

    print(
        f"[DEBUG] Video processing complete: {landmarks_detected_count}/{frame_count} frames with landmarks"
    )

    return {
        "original_frames": original_frames,
        "analyzed_frames": analyzed_frames,
        "metrics": metrics_list,
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "landmarks_detected_count": landmarks_detected_count,
    }


def create_video_file(frames, fps, width, height, output_path):
    """Create a video file from frames"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()


# ===== STREAMLIT UI =====

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
tab1, tab2, tab3 = st.tabs(["📊 ANALYSIS", "📈 METRICS", "ℹ️ INFO"])

with tab1:
    st.markdown(
        """
    <div style='margin-bottom: 2rem;'>
        <h2 style='color: #3b82f6; margin-bottom: 0.5rem;'>UPLOAD SWING</h2>
        <p style='color: #d1d5db; margin: 0;'>Import your golf swing video for AI-powered analysis</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drop your video file here",
        type=["mp4", "mov", "avi", "mkv"],
        help="Supported formats: MP4, MOV, AVI, MKV",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_video_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file.flush()
                temp_video_path = tmp_file.name

            st.markdown(
                """
            <div style='background: rgba(20, 184, 166, 0.1); border-left: 4px solid #14b8a6; 
                        padding: 1rem; border-radius: 4px; margin-bottom: 1rem;'>
                <span style='color: #14b8a6; font-weight: 600;'>✓ Video Ready</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Process button
            if st.button(
                "▶ ANALYZE SWING", use_container_width=True, key="analyze_btn"
            ):
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.markdown(
                    """
                <div style='color: #3b82f6; text-align: center; font-weight: 600;'>
                    Processing video...
                </div>
                """,
                    unsafe_allow_html=True,
                )

                try:
                    # Process video
                    results = process_video(temp_video_path)

                    status_text.markdown(
                        """
                    <div style='color: #14b8a6; text-align: center; font-weight: 600;'>
                        ✓ Analysis Complete
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                    progress_bar.progress(100)

                    # Store in session state
                    st.session_state.results = results
                    st.session_state.uploaded_filename = uploaded_file.name

                    st.rerun()

                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                finally:
                    # Clean up temp file with error handling
                    if temp_video_path and os.path.exists(temp_video_path):
                        try:
                            os.remove(temp_video_path)
                        except PermissionError:
                            pass
        except Exception as e:
            st.error(f"Upload error: {str(e)}")

    # Display results if available
    if "results" in st.session_state:
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

        # Video Playback Section
        st.markdown(
            """
        <h2 style='color: #3b82f6; margin-bottom: 1.5rem;'>SWING REPLAY</h2>
        """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            <div style='background: rgba(59, 130, 246, 0.05); border: 1px solid rgba(59, 130, 246, 0.2);
                        padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
                <p style='color: #3b82f6; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; 
                          font-size: 0.85rem; margin: 0 0 0.5rem 0;'>Original</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            original_frame_display = st.empty()
            playback_speed = st.slider(
                "Speed",
                0.1,
                3.0,
                1.0,
                key="original_speed",
                help="Playback speed multiplier",
            )

            if st.button(
                "▶ PLAY ORIGINAL", use_container_width=True, key="play_original"
            ):
                for frame in results["original_frames"]:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    original_frame_display.image(frame_rgb)
                    import time

                    time.sleep(1 / (results["fps"] * playback_speed))

        with col2:
            st.markdown(
                """
            <div style='background: rgba(217, 119, 6, 0.05); border: 1px solid rgba(217, 119, 6, 0.2);
                        padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
                <p style='color: #d97706; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
                          font-size: 0.85rem; margin: 0 0 0.5rem 0;'>AI Analyzed</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            analyzed_frame_display = st.empty()
            analyzed_speed = st.slider(
                "Speed",
                0.1,
                3.0,
                1.0,
                key="analyzed_speed",
                help="Playback speed multiplier",
            )

            if st.button(
                "▶ PLAY ANALYZED", use_container_width=True, key="play_analyzed"
            ):
                for frame in results["analyzed_frames"]:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    analyzed_frame_display.image(frame_rgb)
                    import time

                    time.sleep(1 / (results["fps"] * analyzed_speed))

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
            st.dataframe(df_metrics, use_container_width=True)

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
