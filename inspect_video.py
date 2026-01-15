#!/usr/bin/env python3
"""Extract first few frames from a video to inspect"""
import cv2
import sys

video_path = "20250819_134559.mp4" if len(sys.argv) < 2 else sys.argv[1]

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"ERROR: Could not open {video_path}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height} @ {fps}fps, {total} frames")
print(f"Video path exists: {True}")

# Read first 5 frames
for i in range(min(5, total)):
    ret, frame = cap.read()
    if not ret:
        print(f"  Frame {i}: FAILED to read")
        break
    print(f"  Frame {i}: Read OK, shape={frame.shape}")

cap.release()
print("Done")
