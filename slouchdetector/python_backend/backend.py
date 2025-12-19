import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from flask import Flask, Response, jsonify
from enum import Enum

# --- Configuration ---
CALIBRATION_FRAMES = 60
PORT = 5001

# --- Global State ---
output_frame = None
lock = threading.Lock()

system_state = {
    "is_slouching": False,
    "message": "Waiting...",
}

app = Flask(__name__)

class AppState(Enum):
    WAITING_FOR_PERSON = 0
    CALIBRATING = 1
    MONITORING = 2

class PostureEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(image_rgb)

    def draw_landmarks(self, frame, results, color=(0, 255, 0)):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
            )

class BiometricsCalculator:
    @staticmethod
    def calculate_metrics(landmarks):
        nose = landmarks[0]
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]
        
        def to_np(lm): return np.array([lm.x, lm.y])
        
        nose_pt = to_np(nose)
        l_sh_pt = to_np(l_shoulder)
        r_sh_pt = to_np(r_shoulder)

        shoulder_width = np.linalg.norm(l_sh_pt - r_sh_pt)
        if shoulder_width < 0.01: return None

        shoulder_midpoint = (l_sh_pt + r_sh_pt) / 2.0
        torso_y = shoulder_midpoint[1]
        neck_dist = np.linalg.norm(nose_pt - shoulder_midpoint)
        neck_ratio = neck_dist / shoulder_width
        centroid_x = (nose_pt[0] + l_sh_pt[0] + r_sh_pt[0]) / 3.0
        
        points = [nose, l_shoulder, r_shoulder]
        out_of_bounds = any(p.x < 0.01 or p.x > 0.99 or p.y < 0.01 or p.y > 0.99 for p in points)

        return {
            "shoulder_width": shoulder_width,
            "neck_ratio": neck_ratio,
            "torso_y": torso_y,
            "centroid_x": centroid_x,
            "out_of_bounds": out_of_bounds
        }

def vision_loop():
    global output_frame, system_state
    
    estimator = PostureEstimator()
    biometrics = BiometricsCalculator()
    
    current_state = AppState.WAITING_FOR_PERSON
    calibration_samples = []
    baseline = None
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
            
        frame = cv2.flip(frame, 1) # Mirror effect

        results = estimator.process_frame(frame)
        metrics = None

        if results.pose_landmarks:
            metrics = biometrics.calculate_metrics(results.pose_landmarks.landmark)

        status_text = "Init..."
        color = (200, 200, 200)

        if current_state == AppState.WAITING_FOR_PERSON:
            status_text = "Waiting for a person..."
            system_state["is_slouching"] = False
            system_state["message"] = status_text
            if metrics:
                current_state = AppState.CALIBRATING
                calibration_samples = []

        elif current_state == AppState.CALIBRATING:
            progress = int((len(calibration_samples) / CALIBRATION_FRAMES) * 100)
            status_text = f"Sit straight! Calibration {progress}%"
            color = (0, 255, 255) # Yellow
            system_state["message"] = status_text

            if metrics:
                calibration_samples.append(metrics)
                estimator.draw_landmarks(frame, results, color=color)
            
            if len(calibration_samples) >= CALIBRATION_FRAMES:
                baseline = {
                    "neck_ratio": np.mean([s["neck_ratio"] for s in calibration_samples]),
                    "torso_y": np.mean([s["torso_y"] for s in calibration_samples]),
                }
                current_state = AppState.MONITORING
                print(f"Calibration Done. Baseline: {baseline}")

        elif current_state == AppState.MONITORING:
            if not metrics:
                status_text = "No person"
                system_state["is_slouching"] = False
                system_state["message"] = status_text
            else:
                is_bad = False
                msg = "Correct posture"
                
                if metrics["out_of_bounds"] or abs(metrics["centroid_x"] - 0.5) > 0.25:
                    is_bad = True
                    msg = "Please sit in the middle"
                elif metrics["neck_ratio"] < (baseline["neck_ratio"] * 0.80):
                    is_bad = True
                    msg = "Do not hunch over!"
                elif metrics["torso_y"] > (baseline["torso_y"] * 1.10):
                    is_bad = True
                    msg = "Straighten your back!"

                system_state["is_slouching"] = is_bad
                system_state["message"] = msg

                if is_bad:
                    color = (0, 0, 255) # Red
                    status_text = f"ERROR: {msg}"
                else:
                    color = (0, 255, 0) # Green
                    status_text = "OK"

                estimator.draw_landmarks(frame, results, color=color)

        # Draw status on video for visual debug
        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        with lock:
            output_frame = frame.copy()

    cap.release()

def generate_frames():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    return jsonify(system_state)

@app.route('/current_frame')
def current_frame():
    global output_frame
    with lock:
        if output_frame is None:
            return "Camera initializing...", 404
            
        # Encode frame to JPEG
        (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
        if not flag:
            return "Encoding error", 500
        
        # Return bytes
        return Response(encodedImage.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    print("Starting Python Backend...")
    
    t = threading.Thread(target=vision_loop, daemon=True)
    t.start()
    print(f"Vision loop started. Starting Flask on port {PORT}...")
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
    print(f"CRITICAL ERROR: {e}")
    print("Backend exiting.")
