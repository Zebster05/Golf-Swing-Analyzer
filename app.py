import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import numpy as np

# --- 1. SETUP PAGE CONFIG ---
st.set_page_config(page_title="Golf AI Analyzer", page_icon="🏌️")

st.title("🏌️ AI Golf Swing Analyzer")

# --- 2. UPLOAD VIDEO ---
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- 3. SETTINGS ---
    with st.expander("⚙️ Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            # Slider to skip the "waggle"
            start_frame = st.slider("Start at Frame:", 0, total_frames, 0)
        with col2:
            # Checkbox to enable/disable the "Pause" feature
            pause_on_fault = st.checkbox("Pause video when head moves?", value=True)

        # Sensitivity slider (Make the box tighter or looser)
        strictness = st.slider(
            "Strictness (Gap Size)", 0.1, 1.0, 0.5, help="Lower number = Tighter lines"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # --- 4. SETUP MEDIAPIPE ---
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    stframe = st.empty()

    initial_nose_y = None

    # --- 5. MAIN LOOP ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        h, w, c = frame.shape

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Key Points
            nose = landmarks[0]
            left_eye = landmarks[2]
            right_eye = landmarks[5]

            # 1. ESTABLISH THE REFERENCE LINES (On first frame only)
            if initial_nose_y is None:
                initial_nose_y = nose.y

            # Calculate Face Width (to keep the scale relative)
            face_width = abs(left_eye.x - right_eye.x)

            # Calculate the Gap (using the Strictness slider) - controls line distance
            gap = (
                strictness * 0.1
            )  # Converts strictness to a proportion of screen height

            # Define the Top and Bottom Line positions (in pixels)
            # Lines are stationary based on initial nose position
            top_line_y = int((initial_nose_y - gap) * h)
            bottom_line_y = int((initial_nose_y + gap) * h)

            nose_y_pixel = int(nose.y * h)
            nose_x_pixel = int(nose.x * w)

            # 2. CHECK FOR MOVEMENT
            # Logic: Is the current nose position ABOVE the top line OR BELOW the bottom line?
            if nose.y < (initial_nose_y - gap) or nose.y > (initial_nose_y + gap):
                status = "FAULT DETECTED"
                color = (255, 0, 0)  # Red
                is_fault = True
            else:
                status = "GOOD"
                color = (0, 255, 0)  # Green
                is_fault = False

            # 3. DRAW THE UI
            # Draw Top Line
            cv2.line(frame, (0, top_line_y), (w, top_line_y), (0, 255, 255), 2)
            # Draw Bottom Line
            cv2.line(frame, (0, bottom_line_y), (w, bottom_line_y), (0, 255, 255), 2)

            # Draw Nose Dot
            cv2.circle(frame, (nose_x_pixel, nose_y_pixel), 8, color, -1)

            # Draw Status Text
            cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # 4. DISPLAY THE FRAME
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb)

            # 5. PAUSE LOGIC
            # If we found a fault AND the user wants to pause:
            if is_fault and pause_on_fault:
                st.error("🛑 Movement Detected! Video Paused.")
                st.info("Uncheck 'Pause video when head moves?' to continue playing.")
                break  # This stops the loop, freezing the video on the current frame.

    cap.release()
