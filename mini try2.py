import cv2, os, sys, math, json, time, joblib, itertools
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

VIDEO_PATH = r"C:\Users\Atharva\OneDrive\Desktop\demo\aies mini\videos\sample3.mp4"

EXERCISE = "squat"   
OUTPUT_DIR = "./artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

ANNOTATED_VIDEO = os.path.join(OUTPUT_DIR, f"annotated_{EXERCISE}.mp4")
LANDMARKS_CSV   = os.path.join(OUTPUT_DIR, f"landmarks_{EXERCISE}.csv")
FEATURES_CSV    = os.path.join(OUTPUT_DIR, f"features_{EXERCISE}.csv")
MODEL_PATH      = os.path.join(OUTPUT_DIR, f"{EXERCISE}_rf_model.pkl")
SESSION_LOG     = os.path.join(OUTPUT_DIR, f"{EXERCISE}_session_log.csv")

def calc_angle(a, b, c):
    """
    a,b,c = (x, y) points. Returns angle ABC in degrees.
    """
    a = np.array(a); b = np.array(b); c = np.array(c)
    ang = np.degrees(np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]))
    ang = np.abs(ang)
    if ang > 180: ang = 360 - ang
    return ang

def smooth_series(x, window=5):
    if len(x) < window: return x
    return pd.Series(x).rolling(window, min_periods=1, center=True).mean().values

def landmark_row(landmarks, frame_idx, W, H):
    """
    Return flat vector [frame, x0,y0,z0, vis0, x1,y1,z1,vis1, ...]
    pixel coords (x*W, y*H) for reproducibility.
    """
    row = [frame_idx]
    for lm in landmarks:
        row += [lm.x*W, lm.y*H, lm.z, lm.visibility]
    return row

def ensure_cap(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap


def extract_landmarks_to_csv(video_path, csv_out):
    cap = ensure_cap(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_landmarks = 33
    
    header = ["frame"]
    for i in range(n_landmarks):
        header += [f"x_{i}", f"y_{i}", f"z_{i}", f"vis_{i}"]
    
    rows = []
    with mp_pose.Pose(static_image_mode=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        f = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if res.pose_landmarks:
                rows.append(landmark_row(res.pose_landmarks.landmark, f, W, H))
            else:
                rows.append([f] + [np.nan]*(4*n_landmarks))
            f += 1
    cap.release()
    
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(csv_out, index=False)
    return df

df_landmarks = extract_landmarks_to_csv(VIDEO_PATH, LANDMARKS_CSV)
print("Saved:", LANDMARKS_CSV, df_landmarks.shape)
df_landmarks.head()

def compute_key_angles(df, exercise):
    # indices
    L_HIP, L_KNEE, L_ANKLE = 23, 25, 27
    L_SH,  L_ELB,  L_WRIST = 11, 13, 15
    
    angles = []
    for _, row in df.iterrows():
        def P(i):
            return (row[f"x_{i}"], row[f"y_{i}"])
        if exercise == "squat":
            if pd.isna(row[f"x_{L_HIP}"]): angles.append(np.nan); continue
            ang = calc_angle(P(L_HIP), P(L_KNEE), P(L_ANKLE))
        elif exercise == "pushup":
            if pd.isna(row[f"x_{L_SH}"]): angles.append(np.nan); continue
            ang = calc_angle(P(L_SH), P(L_ELB), P(L_WRIST))
        else:
            raise ValueError("exercise must be 'squat' or 'pushup'")
        angles.append(ang)
    return np.array(angles)

angles = compute_key_angles(df_landmarks, EXERCISE)
angles_s = smooth_series(angles, window=7)

# Auto-labeling thresholds
if EXERCISE == "squat":
    # More tolerant squat detection
    # Accept even shallow or imperfect squats by widening the threshold range
    up_thr   = 150   # was 160 → accept less straight legs
    down_thr = 110   # was 90  → accept higher (shallower) squats



labels = []
stage = None
reps = 0
for a in angles_s:
    if np.isnan(a):
        labels.append("none")
        continue
    if a > up_thr:
        stage = "up"
        labels.append("up")
    elif a < down_thr and stage == "up":
        stage = "down"
        reps += 1
        labels.append("down")
    else:
        labels.append(stage if stage is not None else "none")

feature_df = df_landmarks[["frame"]].copy()
feature_df["angle"] = angles_s
feature_df["label"] = labels

# basic temporal features
feature_df["angle_slope"] = feature_df["angle"].diff().fillna(0.0)
feature_df["angle_slope2"] = feature_df["angle_slope"].diff().fillna(0.0)
feature_df["angle_min3"] = pd.Series(angles_s).rolling(3, min_periods=1).min()
feature_df["angle_max3"] = pd.Series(angles_s).rolling(3, min_periods=1).max()

feature_df.to_csv(FEATURES_CSV, index=False)
print(f"Auto labels done. Estimated reps: {reps}")
print("Saved:", FEATURES_CSV, feature_df.shape)
feature_df.head()

# Prepare data (drop NaNs and rare none)
train_df = feature_df.copy()
train_df = train_df.dropna(subset=["angle"])
train_df = train_df[train_df["label"].isin(["up","down","none"])]

X = train_df[["angle","angle_slope","angle_slope2","angle_min3","angle_max3"]].values
y = train_df["label"].values

# If few 'none', keep them; else can downsample. We'll proceed directly.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_tr, y_tr)
joblib.dump(clf, MODEL_PATH)

y_pred = clf.predict(X_te)
print("Accuracy:", accuracy_score(y_te, y_pred))
print(classification_report(y_te, y_pred))

cm = confusion_matrix(y_te, y_pred, labels=["up","down","none"])
print("Confusion matrix (up,down,none):\n", cm)

# Simple confusion-matrix plot (single chart)
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0,1,2], ["up","down","none"])
plt.yticks([0,1,2], ["up","down","none"])
for i in range(3):
    for j in range(3):
        plt.text(j, i, int(cm[i,j]), ha="center", va="center")
plt.show()

def write_annotated_video(video_path, out_path, model, exercise):
    cap = ensure_cap(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS if FPS>0 else 25, (W,H))
    
    # For dynamic smoothing during inference
    angle_hist = []

    stage = None
    reps = 0
    frame_idx = 0

    with mp_pose.Pose(static_image_mode=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            angle = np.nan
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                if exercise == "squat":
                    hip,knee,ankle = (23,25,27)
                    a = (lms[hip].x*W, lms[hip].y*H)
                    b = (lms[knee].x*W, lms[knee].y*H)
                    c = (lms[ankle].x*W, lms[ankle].y*H)
                    angle = calc_angle(a,b,c)
                else:
                    sh,el,wr = (11,13,15)
                    a = (lms[sh].x*W, lms[sh].y*H)
                    b = (lms[el].x*W, lms[el].y*H)
                    c = (lms[wr].x*W, lms[wr].y*H)
                    angle = calc_angle(a,b,c)

            angle_hist.append(angle)
            if len(angle_hist) > 7:
                angle_s = np.mean(angle_hist[-7:])
            else:
                angle_s = np.mean([v for v in angle_hist if not np.isnan(v)]) if len(angle_hist)>0 else np.nan

            # build feature vector on the fly
            if len(angle_hist) < 3:
                slope = 0.0
                slope2 = 0.0
            else:
                slope  = (angle_hist[-1] - angle_hist[-2]) if not np.isnan(angle_hist[-1]) and not np.isnan(angle_hist[-2]) else 0.0
                slope2 = slope - ((angle_hist[-2]-angle_hist[-3]) if not np.isnan(angle_hist[-3]) and not np.isnan(angle_hist[-2]) else 0.0)

            win = angle_hist[-3:]
            win = [w for w in win if not np.isnan(w)]
            amin = np.min(win) if len(win)>0 else np.nan
            amax = np.max(win) if len(win)>0 else np.nan

            feat = np.array([[angle_s, slope, slope2, amin, amax]])
            # if NaN present, replace with 0 (model trained on real numbers)
            feat = np.nan_to_num(feat, nan=0.0)
            pred = model.predict(feat)[0]

            # Rep counting (robust—use thresholds as guard)
            if EXERCISE == "squat":
    # More tolerant squat detection
    # Accept even shallow or imperfect squats by widening the threshold range
              up_thr   = 150   # was 160 → accept less straight legs
              down_thr = 110   # was 90  → accept higher (shallower) squats


            if not np.isnan(angle_s):
                if angle_s > up_thr:
                    stage = "up"
                elif angle_s < down_thr and stage == "up":
                    stage = "down"
                    reps += 1

            # Draw pose + HUD
            if res.pose_landmarks:
                mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.rectangle(frame, (10,10), (330,100), (0,0,0), -1)
            cv2.putText(frame, f"EXERCISE: {exercise.upper()}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame, f"MODEL: {pred}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.rectangle(frame, (W-200,10), (W-10,100), (0,0,0), -1)
            cv2.putText(frame, "REPS", (W-190,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame, f"{reps}", (W-150,90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255,255,255), 3)

            if not np.isnan(angle_s):
                cv2.putText(frame, f"ANGLE: {int(angle_s)}", (20, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            writer.write(frame)
            frame_idx += 1

    cap.release(); writer.release()
    return reps

# Load trained model & run
model = joblib.load(MODEL_PATH)
final_reps = write_annotated_video(VIDEO_PATH, ANNOTATED_VIDEO, model, EXERCISE)

print(f"Annotated video saved to: {ANNOTATED_VIDEO}")
print(f"Final counted reps: {final_reps}")

# Save session log
pd.DataFrame([{
    "exercise": EXERCISE,
    "video": os.path.basename(VIDEO_PATH),
    "reps": final_reps,
    "model": os.path.basename(MODEL_PATH),
    "features_csv": os.path.basename(FEATURES_CSV),
    "landmarks_csv": os.path.basename(LANDMARKS_CSV),
    "annotated_video": os.path.basename(ANNOTATED_VIDEO),
    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
}]).to_csv(SESSION_LOG, index=False)

print("Session log saved:", SESSION_LOG)

# Angle over time
plt.figure()
plt.plot(feature_df["frame"], feature_df["angle"])
plt.title("Angle over time")
plt.xlabel("Frame")
plt.ylabel("Angle")
plt.show()

# Stage (numeric) over time
label_map = {"up":1, "down":-1, "none":0}
plt.figure()
plt.plot(feature_df["frame"], feature_df["label"].map(label_map))
plt.title("Stage over time")
plt.xlabel("Frame")
plt.ylabel("Stage (up=1, down=-1, none=0)")
plt.show()



# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

video_path = video_path = r"C:\Users\Atharva\OneDrive\Desktop\demo\aies mini\videos\sample3.mp4"

cap = cv2.VideoCapture(video_path)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        h, w, _ = frame.shape
        
        # Select important joints
        key_points = {
            "Head": results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE],
            "Right Shoulder": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            "Left Shoulder": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
            "Right Elbow": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW],
            "Left Elbow": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW],
            "Right Knee": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE],
            "Left Knee": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE],
            "Right Ankle": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE],
            "Left Ankle": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE],
        }
        
        # Draw circles and labels
        for name, lm in key_points.items():
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_rgb, (x, y), 10, (255, 0, 0), -1)  # Blue circle
            cv2.putText(frame_rgb, name, (x+10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if frame_count % 150 == 0:
        plt.imshow(frame_rgb)
        plt.axis("off")
        plt.show()
    
    frame_count += 1

cap.release()



import numpy as np
import math
import os
try:
    from google.colab.patches import cv2_imshow as _cv2_imshow  # type: ignore
except Exception:
    _cv2_imshow = None

def safe_imshow(img, title=None):
    if _cv2_imshow is not None:
        _cv2_imshow(img)
        return
    try:
        import matplotlib.pyplot as _plt
        if getattr(img, "ndim", 0) == 3 and img.shape[2] == 3:
            _plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            _plt.imshow(img, cmap="gray")
        _plt.axis("off")
        if title:
            _plt.title(title)
        _plt.show()
    except Exception:
        # fallback to blocking cv2 window (only works with GUI)
        cv2.imshow(title or "frame", img)
        cv2.waitKey(1)

# --- Ensure model exists before loading for annotation ---
if __name__ == "__main__":
    # choose a single, correct video path for your environment
    VIDEO_PATH = r"C:\Users\Atharva\OneDrive\Desktop\demo\aies mini\videos\yoga.mp4"
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    # load model if present
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        final_reps = write_annotated_video(VIDEO_PATH, ANNOTATED_VIDEO, model, EXERCISE)
        print(f"Annotated video saved to: {ANNOTATED_VIDEO}")
        print(f"Final counted reps: {final_reps}")
    else:
        print(f"Model not found at {MODEL_PATH}. Skipping annotated video generation.")




cap = cv2.VideoCapture("/kaggle/input/yoga00/yoga.mp4")  

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape

            # ---- Right Elbow ----
            shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)
            elbow = (landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h)
            wrist = (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h)

            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            cv2.putText(image, f"Elbow: {elbow_angle} deg",
                        (int(elbow[0]), int(elbow[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # ---- Right Knee ----
            hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h)
            knee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h)
            ankle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h)

            knee_angle = calculate_angle(hip, knee, ankle)
            cv2.putText(image, f"Knee: {knee_angle} deg",
                        (int(knee[0]), int(knee[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # Draw skeleton
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show one frame every 50 to avoid too much output
        if frame_count % 50 == 0:
            cv2_imshow(image)

        frame_count += 1

cap.release()
