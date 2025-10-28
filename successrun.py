import cv2

import mediapipe as mp
import numpy as np
import time # Added for potential frame rate control if needed

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# --- Helper Functions ---

def calculate_angle(a, b, c):
    """Calculates the angle between three points (in degrees)."""
    a = np.array(a) # First point
    b = np.array(b) # Mid point (vertex)
    c = np.array(c) # End point

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle # Adjust reflex angle
    return angle

def rescale_frame(frame, percent=50):
    """Resizes a frame to a specific percentage."""
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# --- Configuration ---
VIDEO_SOURCE = r"C:\Users\Atharva\OneDrive\Desktop\demo\aies mini\videos\sample3.mp4" # Path to video file or 0 for webcam
OUTPUT_VIDEO_NAME = 'output_squat_analysis.mp4'
RESCALE_PERCENTAGE = 65 # Adjust resizing percentage (e.g., 50 for 50%)
MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF = 0.5
POSE_SIDE = 'RIGHT' # Choose 'LEFT' or 'RIGHT' side for calculations

# --- Main Processing ---

# Determine which landmarks to use based on POSE_SIDE
if POSE_SIDE.upper() == 'LEFT':
    shoulder_lm = mp_pose.PoseLandmark.LEFT_SHOULDER
    hip_lm = mp_pose.PoseLandmark.LEFT_HIP
    knee_lm = mp_pose.PoseLandmark.LEFT_KNEE
    ankle_lm = mp_pose.PoseLandmark.LEFT_ANKLE
else: # Default to RIGHT
    shoulder_lm = mp_pose.PoseLandmark.RIGHT_SHOULDER
    hip_lm = mp_pose.PoseLandmark.RIGHT_HIP
    knee_lm = mp_pose.PoseLandmark.RIGHT_KNEE
    ankle_lm = mp_pose.PoseLandmark.RIGHT_ANKLE

# Video Input
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Could not open video source: {VIDEO_SOURCE}")
    exit()

# Video Output Setup (Infer size after first frame resize)
frame_width = None
frame_height = None
out = None

# Squat analysis variables
counter = 0
stage = None
min_angle_knee_rep = 0
min_angle_hip_rep = 0
current_rep_knee_angles = []
current_rep_hip_angles = []

# Initialize MediaPipe Pose instance
with mp_pose.Pose(min_detection_confidence=MIN_DETECTION_CONF,
                  min_tracking_confidence=MIN_TRACKING_CONF) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Reached end of video or cannot read frame.")
            break

        # Resize frame
        frame = rescale_frame(frame, percent=RESCALE_PERCENTAGE)

        # Setup VideoWriter on first valid frame
        if out is None:
            frame_height, frame_width, _ = frame.shape
            size = (frame_width, frame_height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
            out = cv2.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, 24, size) # Adjust FPS if needed
            if not out.isOpened():
                print(f"Error: Could not open video writer for {OUTPUT_VIDEO_NAME}")
                cap.release()
                exit()


        # Recolor image to RGB for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # Improve performance

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks and calculate angles
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[shoulder_lm.value].x, landmarks[shoulder_lm.value].y]
            hip = [landmarks[hip_lm.value].x, landmarks[hip_lm.value].y]
            knee = [landmarks[knee_lm.value].x, landmarks[knee_lm.value].y]
            ankle = [landmarks[ankle_lm.value].x, landmarks[ankle_lm.value].y]

            # Calculate angles
            angle_knee = calculate_angle(hip, knee, ankle) # Knee joint angle
            angle_hip = calculate_angle(shoulder, hip, knee) # Hip joint angle

            # Store angles for current rep analysis
            current_rep_knee_angles.append(angle_knee)
            current_rep_hip_angles.append(angle_hip)

            # --- Squat Counter Logic ---
            if angle_knee > 160: # Threshold for 'up' stage (adjust as needed)
                stage = "up"
            if angle_knee <= 90 and stage == 'up': # Threshold for 'down' stage and completing a rep
                stage = "down"
                counter += 1
                # Analyze angles for the completed rep
                if current_rep_knee_angles:
                    min_angle_knee_rep = round(min(current_rep_knee_angles), 1)
                if current_rep_hip_angles:
                    min_angle_hip_rep = round(min(current_rep_hip_angles), 1)

                # Reset angle lists for the next rep
                current_rep_knee_angles = []
                current_rep_hip_angles = []
                print(f"Rep: {counter}, Min Knee Angle: {min_angle_knee_rep}, Min Hip Angle: {min_angle_hip_rep}")


            # --- Visualization ---
            # Display angles on joints
            cv2.putText(image, f"{angle_knee:.1f}",
                           tuple(np.multiply(knee, [frame_width, frame_height]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA) # Cyan for knee

            cv2.putText(image, f"{angle_hip:.1f}",
                           tuple(np.multiply(hip, [frame_width, frame_height]).astype(int)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA) # Red for hip


        except Exception as e:
            # print(f"Error processing landmarks: {e}") # Uncomment for debugging
            pass # Continue if landmarks are not detected in a frame

        # --- Render Status Box ---
        box_x, box_y, box_w, box_h = 10, 10, 280, 100
        cv2.rectangle(image, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1) # Black box

        # Repetition Data
        cv2.putText(image, f"REPS: {counter}",
                    (box_x + 10, box_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Min Knee Angle Data (from last rep)
        cv2.putText(image, f"Knee Angle (min): {min_angle_knee_rep}",
                    (box_x + 10, box_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA) # Cyan

        # Min Hip Angle Data (from last rep)
        cv2.putText(image, f"Hip Angle (min): {min_angle_hip_rep}",
                    (box_x + 10, box_y + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA) # Red


        # --- Render Pose Detections ---
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2), # Black joints
                                mp_drawing.DrawingSpec(color=(203, 17, 17), thickness=2, circle_radius=2) # Red connections
                                 )

        # --- Output ---
        out.write(image) # Write frame to output video
        cv2.imshow('Squat Analysis', image) # Display processed frame

        # Exit condition
        if cv2.waitKey(5) & 0xFF == ord('q'): # Press 'q' to quit
            break

# --- Cleanup ---
cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()

print("Processing finished.")