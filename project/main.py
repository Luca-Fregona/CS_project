import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import threading
import sys
import os
from plyer import notification
from datetime import datetime
from enum import Enum
from collections import deque
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# --- Constants & Configuration ---
WINDOW_NAME = "Slouch Detector"
CALIBRATION_FRAMES = 60
NOTIFICATION_COOLDOWN = 10  # Seconds between notifications
BAD_POSTURE_TRIGGER_DURATION = 5.0  # Seconds of bad posture to trigger alert
BREAK_INTERVAL_SEC = 10  # 5 minutes break reminder

class AppState(Enum):
    WAITING_FOR_CALIBRATION = 0
    SAMPLING = 1
    MONITORING = 2

class PostureStatus(Enum):
    GOOD = "Good Posture"
    ERR_SLOUCH = "Slouching (Neck Flexion)"
    ERR_SLUMP = "Slumping (Torso Drop)"
    ERR_CENTER = "Off-Center / Out of Frame"
    UNCALIBRATED = "Uncalibrated"

# --- Posture Estimator ---
class PostureEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results

    def draw_landmarks(self, frame, results, color=(0, 255, 0)):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
            )

# --- Biometrics Calculator ---
class BiometricsCalculator:
    @staticmethod
    def calculate_metrics(landmarks):
        nose = landmarks[0]
        l_shoulder = landmarks[11]
        r_shoulder = landmarks[12]

        def to_np(lm):
            return np.array([lm.x, lm.y])

        nose_pt = to_np(nose)
        l_sh_pt = to_np(l_shoulder)
        r_sh_pt = to_np(r_shoulder)

        shoulder_width = np.linalg.norm(l_sh_pt - r_sh_pt)
        if shoulder_width < 0.01:
            return None

        shoulder_midpoint = (l_sh_pt + r_sh_pt) / 2.0
        neck_dist = np.linalg.norm(nose_pt - shoulder_midpoint)
        neck_ratio = neck_dist / shoulder_width
        torso_y = shoulder_midpoint[1]
        upper_body_centroid = (nose_pt + l_sh_pt + r_sh_pt) / 3.0
        centroid_x = upper_body_centroid[0]

        points = [nose, l_shoulder, r_shoulder]
        out_of_bounds = any(p.x < 0.01 or p.x > 0.99 or p.y < 0.01 or p.y > 0.99 for p in points)

        return {
            "shoulder_width": shoulder_width,
            "neck_ratio": neck_ratio,
            "torso_y": torso_y,
            "centroid_x": centroid_x,
            "out_of_bounds": out_of_bounds,
            "raw_landmarks": {
                "nose": (nose.x, nose.y),
                "l_shoulder": (l_shoulder.x, l_shoulder.y),
                "r_shoulder": (r_shoulder.x, r_shoulder.y)
            }
        }

# --- Calibration Manager ---
class CalibrationManager:
    def __init__(self):
        self.state = AppState.WAITING_FOR_CALIBRATION
        self.samples = []
        self.baseline = None

    def start_sampling(self):
        self.state = AppState.SAMPLING
        self.samples = []
        print("Starting Calibration, stay still...")

    def process_sample(self, metrics):
        if metrics:
            self.samples.append(metrics)
        if len(self.samples) >= CALIBRATION_FRAMES:
            self.finalize_calibration()

    def finalize_calibration(self):
        if not self.samples:
            return
        avg_neck_ratio = np.mean([s["neck_ratio"] for s in self.samples])
        avg_torso_y = np.mean([s["torso_y"] for s in self.samples])
        avg_shoulder_width = np.mean([s["shoulder_width"] for s in self.samples])
        self.baseline = {
            "neck_ratio": avg_neck_ratio,
            "torso_y": avg_torso_y,
            "shoulder_width": avg_shoulder_width
        }
        self.state = AppState.MONITORING
        print(f"Calibration Complete. Baseline: {self.baseline}")

# --- Notification Manager ---
class NotificationManager:
    def __init__(self):
        self.bad_posture_start_time = None
        self.last_notification_time = 0

    def update_status(self, is_bad_posture, status_text):
        if not is_bad_posture:
            self.bad_posture_start_time = None
            return

        if self.bad_posture_start_time is None:
            self.bad_posture_start_time = time.time()
        
        duration = time.time() - self.bad_posture_start_time
        if duration > BAD_POSTURE_TRIGGER_DURATION:
            self.trigger_notification(status_text)

    def trigger_notification(self, message):
        now = time.time()
        if now - self.last_notification_time > NOTIFICATION_COOLDOWN:
            self.last_notification_time = now
            threading.Thread(target=self._show_toast, args=(message,), daemon=True).start()

    def _show_toast(self, message):
        try:
            notification.notify(
                title="Slouch Detector Alert",
                message=f"Correct your posture: {message}",
                app_name="Slouch Detector",
                timeout=3
            )
        except Exception as e:
            print(f"Notification failed: {e}")

# --- Data Logger ---
class DataLogger:
    def __init__(self):
        self.log_buffer = []

    def log_frame(self, frame_idx, metrics, status):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "frame_index": frame_idx,
            "status": status.value
        }
        if metrics:
            entry.update({
                "neck_ratio": metrics["neck_ratio"],
                "torso_y": metrics["torso_y"],
                "centroid_x": metrics["centroid_x"],
                "shoulder_width": metrics["shoulder_width"]
            })
        self.log_buffer.append(entry)

    def save_to_csv(self, filename="posture_session_log.csv"):
        if not self.log_buffer:
            print("No data to save.")
            return
        df = pd.DataFrame(self.log_buffer)
        try:
            df.to_csv(filename, index=False)
            print(f"Session log saved to {filename}")
        except Exception as e:
            print(f"Failed to save log: {e}")

# --- Main App ---
class SlouchDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Slouch Detector")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Style
        style = ttk.Style()
        style.theme_use('clam')

        # Logic Components
        self.estimator = PostureEstimator()
        self.calib_manager = CalibrationManager()
        self.biometrics = BiometricsCalculator()
        self.notifier = NotificationManager()
        self.logger = DataLogger()
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open webcam")
            
        self.frame_count = 0
        self.running = True

        # UI Layout
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(self.main_frame)
        self.video_label.pack(padx=10, pady=10)

        self.controls_frame = ttk.Frame(self.main_frame)
        self.controls_frame.pack(fill=tk.X, padx=10, pady=5)

        self.btn_calibrate = ttk.Button(self.controls_frame, text="Calibrate", command=self.start_calibration)
        self.btn_calibrate.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.btn_quit = ttk.Button(self.controls_frame, text="Quit", command=self.on_close)
        self.btn_quit.pack(side=tk.RIGHT, padx=5, expand=True, fill=tk.X)

        # Start loops
        self.update_frame()
        self.schedule_break_reminder()

    # --- Calibration ---
    def start_calibration(self):
        self.calib_manager.start_sampling()

    # --- Break Reminder ---
    def schedule_break_reminder(self):
        """每30分鐘提醒一次休息"""
        def show_break_notification():
            try:
                notification.notify(
                    title="Slouch Detector Break Reminder",
                    message="已經 30 分鐘了，請起身伸展一下！",
                    app_name="Slouch Detector",
                    timeout=5
                )
            except Exception as e:
                print(f"Break notification failed: {e}")
            # 再次安排下一次提醒
            self.root.after(BREAK_INTERVAL_SEC * 1000, show_break_notification)

        self.root.after(BREAK_INTERVAL_SEC * 1000, show_break_notification)

    # --- Frame Update ---
    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            results = self.estimator.process_frame(frame)
            current_status = PostureStatus.UNCALIBRATED
            metrics = None

            if results.pose_landmarks:
                metrics = self.biometrics.calculate_metrics(results.pose_landmarks.landmark)

            # State Machine
            if self.calib_manager.state == AppState.WAITING_FOR_CALIBRATION:
                cv2.putText(frame, "Sit Up Straight and Still & Press 'Calibrate'", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                if metrics:
                    self.estimator.draw_landmarks(frame, results, color=(200, 200, 200))

            elif self.calib_manager.state == AppState.SAMPLING:
                cv2.putText(frame, f"Calibrating... {len(self.calib_manager.samples)}/{CALIBRATION_FRAMES}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                if metrics:
                    self.calib_manager.process_sample(metrics)
                    self.estimator.draw_landmarks(frame, results, color=(0, 255, 255))

            elif self.calib_manager.state == AppState.MONITORING:
                if metrics:
                    baseline = self.calib_manager.baseline
                    if metrics["out_of_bounds"] or abs(metrics["centroid_x"] - 0.5) > 1.5:
                        current_status = PostureStatus.ERR_CENTER
                    elif metrics["neck_ratio"] < (baseline["neck_ratio"] * 0.9):
                        current_status = PostureStatus.ERR_SLOUCH
                    elif metrics["torso_y"] > (baseline["torso_y"] * 1.05):
                        current_status = PostureStatus.ERR_SLUMP
                    else:
                        current_status = PostureStatus.GOOD

                    color = (0, 255, 0) if current_status == PostureStatus.GOOD else (0, 0, 255)
                    self.estimator.draw_landmarks(frame, results, color=color)
                    cv2.putText(frame, f"Status: {current_status.value}", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    self.notifier.update_status(current_status != PostureStatus.GOOD, current_status.value)
                else:
                    cv2.putText(frame, "No Person Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Logging
            self.logger.log_frame(self.frame_count, metrics, current_status)

            # Convert for Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Next frame
        self.root.after(30, self.update_frame)

    # --- Close App ---
    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.logger.save_to_csv()
        self.root.destroy()
        sys.exit(0)

# --- Main ---
if __name__ == "__main__":
    root = tk.Tk()
    app = SlouchDetectorApp(root)
    root.mainloop()
