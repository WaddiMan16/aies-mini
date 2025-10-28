import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- Angle Calculation Function ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


# --- Core Squat Analysis Function ---
def analyze_squat(video_path):
    cap = cv2.VideoCapture(video_path)
    counter = 0
    stage = None
    feedback_list = []

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                # Extract keypoints
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                # Calculate angles
                knee_angle = calculate_angle(hip, knee, ankle)
                hip_angle = calculate_angle(shoulder, hip, knee)

                # Squat counting logic
                if knee_angle > 160:
                    stage = "up"
                if knee_angle < 90 and stage == "up":
                    stage = "down"
                    counter += 1

                # Feedback logic
                feedback = ""
                if knee_angle < 80:
                    feedback = "Too low – avoid over-flexing knees"
                elif hip_angle > 170:
                    feedback = "Keep your back straight"
                elif knee_angle > 150 and stage == "up":
                    feedback = "Good posture – ready to squat"
                else:
                    feedback = "Keep knees aligned with toes"

                feedback_list.append(feedback)

                # Display overlays
                cv2.putText(image, f'Count: {counter}', (30,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(image, f'Knee: {int(knee_angle)}°', (30,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(image, f'Hip: {int(hip_angle)}°', (30,140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(image, f'{feedback}', (30,180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            except:
                pass

            # Write frame to temporary output video
            if 'out' not in locals():
                temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_video_path, fourcc, 20.0, (640,480))
            out.write(image)

        cap.release()
        out.release()

    final_feedback = list(set(feedback_list))
    return temp_video_path, f"Total squats: {counter}\n\nKey feedback:\n- " + "\n- ".join(final_feedback)


# --- Gradio Interface ---
demo = gr.Interface(
    fn=analyze_squat,
    inputs=[
        gr.Video(label="Upload or record your squat video")
    ],
    outputs=[
        gr.Video(label="Processed Video with Analysis"),
        gr.Textbox(label="Performance Feedback")
    ],
    title="AI Squat Form Analyzer 🏋️‍♂️",
    description="Upload your squat video to get real-time biomechanical feedback on form, angles, and posture."
)

if __name__ == "__main__":
    demo.launch()
