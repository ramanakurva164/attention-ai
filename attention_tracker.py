"""
Attention Tracker — OpenCV + Mediapipe 0.10 (Tasks API)
=========================================================
Tracks:
  • Blink detection  via Eye Aspect Ratio (EAR)
  • Gaze direction   (left / right / centre) via iris landmarks
  • Head pose        (pitch / yaw / roll) via solvePnP
  • Attention score  (0-100) combining all signals
  • Session stats    and CSV log

Model downloaded automatically on first run (~4 MB).

Usage:
  python attention_tracker.py
  python attention_tracker.py --camera 1
  python attention_tracker.py --no-display
  python attention_tracker.py --no-csv

Dependencies:
  pip install opencv-python mediapipe numpy
"""

import argparse
import csv
import math
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker

# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
MODEL_PATH = Path(__file__).parent / "face_landmarker.task"


def ensure_model():
    if not MODEL_PATH.exists():
        print(f"Downloading face landmarker model (~4 MB) …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EAR_BLINK_THRESHOLD    = 0.20
EAR_CONSEC_FRAMES      = 2
YAW_ATTENTION_LIMIT    = 25.0
PITCH_ATTENTION_LIMIT  = 20.0
GAZE_DEVIATION_LIMIT   = 0.30
SCORE_DECAY_RATE       = 20.0
SCORE_RECOVER_RATE     = 10.0
SCORE_START            = 100.0
SMOOTH_WINDOW          = 8

CLR_GREEN  = (0,   220,  80)
CLR_YELLOW = (0,   200, 255)
CLR_RED    = (30,   30, 240)
CLR_CYAN   = (230, 200,  20)
CLR_WHITE  = (255, 255, 255)
CLR_DARK   = ( 20,  20,  20)
CLR_IRIS   = (255, 140,   0)

# ── FaceMesh 478-landmark indices ────────────────────────────────────────────

# 6-point EAR layout (same in 0.9 and 0.10)
LEFT_EAR_IDS   = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_IDS  = [33,  160, 158, 133, 144, 153]

# Iris landmarks (indices 468-477 in the 478-point mesh)
LEFT_IRIS_IDS   = [474, 475, 476, 477]
RIGHT_IRIS_IDS  = [469, 470, 471, 472]
LEFT_EYE_CORNERS  = [362, 263]
RIGHT_EYE_CORNERS = [33,  133]

# solvePnP: MediaPipe canonical 3D reference + matching landmark indices
MODEL_POINTS_3D = np.array([
    [ 0.0,    0.0,    0.0   ],   # Nose tip         1
    [ 0.0,  -330.0,  -65.0  ],   # Chin             152
    [-225.0,  170.0, -135.0 ],   # Left eye corner  263
    [ 225.0,  170.0, -135.0 ],   # Right eye corner 33
    [-150.0, -150.0, -125.0 ],   # Left mouth       287
    [ 150.0, -150.0, -125.0 ],   # Right mouth      57
], dtype=np.float64)

POSE_LM_IDS = [1, 152, 263, 33, 287, 57]

# Subset used for drawing contours (eye + lips + face oval)
FACE_OVAL      = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
LEFT_EYE_CON   = [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]
RIGHT_EYE_CON  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
LEFT_BROW      = [276,283,282,295,285,300,293,334,296,336]
RIGHT_BROW     = [46,53,52,65,55,70,63,105,66,107]
LIPS_CON       = [61,146,91,181,84,17,314,405,321,375,61, 291,409,270,269,267,0,37,39,40,185]


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def lm_xy(lm_list, idx, w, h):
    lm = lm_list[idx]
    return int(lm.x * w), int(lm.y * h)


def ear(lm_list, ids, w, h):
    pts = np.array([[lm_list[i].x * w, lm_list[i].y * h] for i in ids])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def iris_offset(lm_list, iris_ids, corner_ids, w, h):
    iris_pts = np.array([[lm_list[i].x * w, lm_list[i].y * h] for i in iris_ids])
    cx = iris_pts[:, 0].mean()
    lc = np.array([lm_list[corner_ids[0]].x * w, lm_list[corner_ids[0]].y * h])
    rc = np.array([lm_list[corner_ids[1]].x * w, lm_list[corner_ids[1]].y * h])
    ew = np.linalg.norm(rc - lc) + 1e-6
    return (cx - (lc[0] + rc[0]) / 2.0) / (ew / 2.0)


def rot_to_euler(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z])


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def text(img, t, pos, scale=0.55, colour=CLR_WHITE, thick=1):
    cv2.putText(img, t, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, CLR_DARK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, t, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour,  thick,     cv2.LINE_AA)


def draw_bar(img, score, x, y, bw=160, bh=18):
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (50, 50, 50), -1)
    fill   = int(bw * score / 100.0)
    colour = CLR_GREEN if score >= 70 else (CLR_YELLOW if score >= 40 else CLR_RED)
    cv2.rectangle(img, (x, y), (x + fill, y + bh), colour, -1)
    cv2.rectangle(img, (x, y), (x + bw,   y + bh), CLR_WHITE, 1)
    text(img, f"{score:.0f}%", (x + bw + 6, y + bh - 2), 0.5, colour)


def draw_contour(img, lm_list, ids, w, h, colour, thickness=1):
    pts = np.array([[int(lm_list[i].x * w), int(lm_list[i].y * h)]
                    for i in ids if i < len(lm_list)], dtype=np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts], False, colour, thickness, cv2.LINE_AA)


def draw_axes(img, rvec, tvec, cam, dist):
    axis     = np.float32([[80,0,0],[0,80,0],[0,0,80]])
    origin   = np.float32([[0,0,0]])
    o2d, _   = cv2.projectPoints(origin, rvec, tvec, cam, dist)
    a2d, _   = cv2.projectPoints(axis,   rvec, tvec, cam, dist)
    o  = tuple(o2d[0].ravel().astype(int))
    cv2.line(img, o, tuple(a2d[0].ravel().astype(int)), (0,   0,   255), 2)
    cv2.line(img, o, tuple(a2d[1].ravel().astype(int)), (0,   255,  0 ), 2)
    cv2.line(img, o, tuple(a2d[2].ravel().astype(int)), (255,  0,   0 ), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Main tracker
# ─────────────────────────────────────────────────────────────────────────────

class AttentionTracker:
    def __init__(self, camera_index=0, show_display=True, log_csv=True):
        self.camera_index = camera_index
        self.show_display = show_display
        self.log_csv      = log_csv

        # Build FaceLandmarker (Tasks API)
        base_opts = mp_tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
        opts = FaceLandmarkerOptions(
            base_options             = base_opts,
            output_face_blendshapes  = False,
            output_facial_transformation_matrixes = False,
            num_faces                = 1,
            min_face_detection_confidence = 0.6,
            min_face_presence_confidence  = 0.6,
            min_tracking_confidence       = 0.5,
            running_mode             = mp_vision.RunningMode.VIDEO,
        )
        self.detector = FaceLandmarker.create_from_options(opts)

        # State
        self.blink_counter   = 0
        self.total_blinks    = 0
        self.attention_score = SCORE_START
        self.session_scores  = []
        self.distraction_log = []

        self.pitch_buf = deque(maxlen=SMOOTH_WINDOW)
        self.yaw_buf   = deque(maxlen=SMOOTH_WINDOW)
        self.roll_buf  = deque(maxlen=SMOOTH_WINDOW)

        # Eye-closure timer: how long eyes have been continuously shut
        self._eyes_closed_since : float | None = None
        # Low-attention timer: when score first dropped below 40
        self._low_attn_since    : float | None = None
        # Alert flash phase (for pulsing border)
        self._alert_phase       = 0.0

        self._last_time = time.time()
        self._frame_ts  = 0          # mediapipe timestamp_ms

        if log_csv:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_path = Path(f"attention_log_{ts}.csv")
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp","attention_score","pitch","yaw","roll",
                    "ear_l","ear_r","gaze_l","gaze_r","blinks","status"
                ])
        else:
            self.csv_path = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cam_mat(self, w, h):
        f = w
        return np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float64)

    def _head_pose(self, lm_list, w, h):
        img_pts = np.array(
            [[lm_list[i].x * w, lm_list[i].y * h] for i in POSE_LM_IDS],
            dtype=np.float64
        )
        cam  = self._cam_mat(w, h)
        dist = np.zeros((4, 1))
        ok, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_3D, img_pts, cam, dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        p, y, r = rot_to_euler(R)
        return p, y, r, rvec, tvec, cam, dist

    def _update_score(self, distracted, dt):
        if distracted:
            self.attention_score -= SCORE_DECAY_RATE * dt
        else:
            self.attention_score += SCORE_RECOVER_RATE * dt
        self.attention_score = max(0.0, min(100.0, self.attention_score))

    # ── Frame processing ──────────────────────────────────────────────────────

    def process_frame(self, frame):
        h, w  = frame.shape[:2]
        now   = time.time()
        dt    = now - self._last_time
        self._last_time = now

        # Convert to mediapipe image
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts += max(1, int(dt * 1000))

        result = self.detector.detect_for_video(mp_image, self._frame_ts)

        pitch = yaw = roll = 0.0
        ear_l = ear_r = gaze_l = gaze_r = 0.0
        rvec = tvec = cam = dist_ = None
        reasons = []

        if not result.face_landmarks:
            reasons.append("NO FACE")
            self._update_score(distracted=True, dt=dt)
            self._draw_no_face(frame)
        else:
            lm = result.face_landmarks[0]   # list of NormalizedLandmark

            # — EAR / blink ——————————————————————————————————————————————
            ear_l = ear(lm, LEFT_EAR_IDS,  w, h)
            ear_r = ear(lm, RIGHT_EAR_IDS, w, h)
            avg_ear     = (ear_l + ear_r) / 2.0
            eyes_closed = avg_ear < EAR_BLINK_THRESHOLD

            # Track continuous eye-closure duration
            if eyes_closed:
                self.blink_counter += 1
                if self._eyes_closed_since is None:
                    self._eyes_closed_since = now
            else:
                if self.blink_counter >= EAR_CONSEC_FRAMES:
                    self.total_blinks += 1
                self.blink_counter      = 0
                self._eyes_closed_since = None

            # Eyes closed for > 0.5 s counts as inattentive
            EYE_CLOSE_GRACE = 0.5   # seconds — normal blinks are < 0.4 s
            eyes_distracted = (
                eyes_closed and
                self._eyes_closed_since is not None and
                (now - self._eyes_closed_since) > EYE_CLOSE_GRACE
            )

            # — Gaze (skip when eyes are closed — iris landmarks are unreliable) ——
            gaze_dist = False
            if not eyes_closed:
                gaze_l = iris_offset(lm, LEFT_IRIS_IDS,  LEFT_EYE_CORNERS,  w, h)
                gaze_r = iris_offset(lm, RIGHT_IRIS_IDS, RIGHT_EYE_CORNERS, w, h)
                avg_gaze  = (abs(gaze_l) + abs(gaze_r)) / 2.0
                gaze_dist = avg_gaze > GAZE_DEVIATION_LIMIT

            # — Head pose ——————————————————————————————————————————————
            pose = self._head_pose(lm, w, h)
            head_dist = False
            if pose:
                pitch, yaw, roll, rvec, tvec, cam, dist_ = pose
                self.pitch_buf.append(pitch)
                self.yaw_buf.append(yaw)
                self.roll_buf.append(roll)
                sp = float(np.mean(self.pitch_buf))
                sy = float(np.mean(self.yaw_buf))
                if abs(sy) > YAW_ATTENTION_LIMIT:
                    head_dist = True
                    reasons.append(f"HEAD {'LEFT' if sy < 0 else 'RIGHT'} {abs(sy):.0f}\u00b0")
                if abs(sp) > PITCH_ATTENTION_LIMIT:
                    head_dist = True
                    reasons.append(f"HEAD {'DOWN' if sp > 0 else 'UP'} {abs(sp):.0f}\u00b0")

            if gaze_dist:
                side = "RIGHT" if (gaze_l + gaze_r) > 0 else "LEFT"
                reasons.append(f"GAZE {side}")

            # Combine all distraction signals including sustained eye closure
            any_distracted = head_dist or gaze_dist or eyes_distracted
            self._update_score(distracted=any_distracted, dt=dt)
            if reasons:
                self.distraction_log.append((now, " | ".join(reasons)))

            # — Draw ———————————————————————————————————————————————————
            if self.show_display:
                self._draw_mesh(frame, lm, w, h, eyes_closed)
                if rvec is not None:
                    draw_axes(frame, rvec, tvec, cam, dist_)

        # — Low-attention alert timer (< 40 for 5+ seconds) ————————————————————
        LOW_ATTN_THRESHOLD = 40.0
        LOW_ATTN_DURATION  = 5.0
        if self.attention_score < LOW_ATTN_THRESHOLD:
            if self._low_attn_since is None:
                self._low_attn_since = now
        else:
            self._low_attn_since = None

        alert_active = (
            self._low_attn_since is not None and
            (now - self._low_attn_since) >= LOW_ATTN_DURATION
        )

        # — HUD & logging ————————————————————————————————————————————————————————
        if self.show_display:
            ec = (avg_ear < EAR_BLINK_THRESHOLD) if result.face_landmarks else False
            self._draw_hud(frame, pitch, yaw, roll,
                           ear_l, ear_r, gaze_l, gaze_r, reasons, w, h, ec)
            if alert_active:
                self._draw_alert(frame, now)

        status = "DISTRACTED" if reasons else "ATTENTIVE"
        self.session_scores.append(self.attention_score)
        if self.csv_path:
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.now().isoformat(),
                    f"{self.attention_score:.1f}",
                    f"{pitch:.1f}", f"{yaw:.1f}", f"{roll:.1f}",
                    f"{ear_l:.3f}", f"{ear_r:.3f}",
                    f"{gaze_l:.3f}", f"{gaze_r:.3f}",
                    self.total_blinks, status,
                ])
        return frame

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_mesh(self, frame, lm, w, h, eyes_closed=False):
        """
        Clean minimal overlay:
          - Fitted ellipse around each eye (open=teal, closed=dim)
          - Iris circle + pupil dot (only when eyes open)
          - Small dots on nose tip, chin, mouth corners
        """
        # ── Eye ellipses ────────────────────────────────────────────────
        eye_open_clr   = (0, 210, 180)
        eye_closed_clr = (80, 80, 80)

        for ear_ids in [LEFT_EAR_IDS, RIGHT_EAR_IDS]:
            pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)]
                            for i in ear_ids], dtype=np.int32)
            # Fit a rotated rectangle to the 6 eye points
            rect = cv2.minAreaRect(pts)
            box  = cv2.boxPoints(rect).astype(np.int32)
            clr  = eye_closed_clr if eyes_closed else eye_open_clr
            cv2.drawContours(frame, [box], 0, clr, 1, cv2.LINE_AA)

        # ── Iris + pupil (only when eyes open) ──────────────────────────
        if not eyes_closed:
            for iris_ids in [LEFT_IRIS_IDS, RIGHT_IRIS_IDS]:
                pts = np.array([[lm[i].x * w, lm[i].y * h]
                                for i in iris_ids])
                cx = int(pts[:, 0].mean())
                cy = int(pts[:, 1].mean())
                r  = max(2, int(np.linalg.norm(pts[0] - pts[2]) / 2))
                cv2.circle(frame, (cx, cy), r,  CLR_IRIS, 1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 2,  CLR_IRIS, -1, cv2.LINE_AA)

        # ── Sparse key-point dots (nose, chin, mouth corners) ───────────
        key_ids = [1, 152, 57, 287, 4]   # nose-tip, chin, mouth corners, nose-bridge
        for i in key_ids:
            px = int(lm[i].x * w)
            py = int(lm[i].y * h)
            cv2.circle(frame, (px, py), 2, (120, 120, 120), -1, cv2.LINE_AA)

    def _draw_alert(self, frame, now):
        """Pulsing red border + warning text when attention stays < 40 for 5 s."""
        h, w = frame.shape[:2]
        # Pulse alpha between 0.35 and 0.65 at ~1 Hz
        pulse = 0.5 + 0.25 * math.sin(now * math.pi * 2)
        border_t = 18
        overlay  = frame.copy()
        # Top / Bottom / Left / Right red strips
        for rect in [
            (0, 0, w, border_t),
            (0, h - border_t, w, h),
            (0, 0, border_t, h),
            (w - border_t, 0, w, h),
        ]:
            cv2.rectangle(overlay, rect[:2], rect[2:], (0, 0, 220), -1)
        cv2.addWeighted(overlay, pulse, frame, 1 - pulse, 0, frame)

        # Central warning banner
        bx, by, bw2, bh2 = w//2 - 240, h//2 - 38, 480, 76
        banner = frame.copy()
        cv2.rectangle(banner, (bx, by), (bx + bw2, by + bh2), (0, 0, 180), -1)
        cv2.addWeighted(banner, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame,  (bx, by), (bx + bw2, by + bh2), (0, 0, 255), 2)

        text(frame, "LOW ATTENTION WARNING",
             (bx + 42, by + 28), scale=0.80, colour=CLR_WHITE, thick=2)
        text(frame, "Focus on screen!",
             (bx + 105, by + 58), scale=0.60, colour=CLR_YELLOW, thick=1)

    def _draw_no_face(self, frame):
        h, w   = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h//2-30), (w, h//2+30), (0,0,180), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        text(frame, "NO FACE DETECTED",
             (w//2 - 160, h//2 + 10), 0.9, CLR_WHITE, 2)

    def _draw_hud(self, frame, pitch, yaw, roll,
                  ear_l, ear_r, gaze_l, gaze_r, reasons, w, h,
                  eyes_closed=False):
        pw, ph = 310, 280
        ol = frame.copy()
        cv2.rectangle(ol, (0, 0), (pw, ph), (15,15,15), -1)
        cv2.addWeighted(ol, 0.65, frame, 0.35, 0, frame)

        text(frame, "ATTENTION",                     (12, 22),  0.58, CLR_CYAN)
        draw_bar(frame, self.attention_score,         12,  30)

        # Status line — eyes-closed gets its own label, not mixed into reasons
        if eyes_closed:
            text(frame, "EYES CLOSED",               (12, 68),  0.45, CLR_YELLOW)
        elif reasons:
            sc = CLR_RED
            st = " | ".join(reasons)[:44]
            text(frame, st,                          (12, 68),  0.45, sc)
        else:
            text(frame, "ATTENTIVE",                 (12, 68),  0.45, CLR_GREEN)

        cv2.line(frame, (8, 76), (pw - 8, 76), (80,80,80), 1)

        text(frame, "HEAD POSE",                     (12, 95),  0.46, CLR_CYAN)
        text(frame, f"  Pitch : {pitch:+.1f}\u00b0", (12,114),  0.46)
        text(frame, f"  Yaw   : {yaw:+.1f}\u00b0",  (12,132),  0.46)
        text(frame, f"  Roll  : {roll:+.1f}\u00b0",  (12,150),  0.46)

        gl = lambda v: "R" if v > GAZE_DEVIATION_LIMIT else ("L" if v < -GAZE_DEVIATION_LIMIT else "C")
        text(frame, "GAZE",                          (12,170),  0.46, CLR_CYAN)
        if eyes_closed:
            text(frame, "  (suppressed — eyes closed)", (12,188), 0.40, (120,120,120))
            text(frame, "",                              (12,205), 0.46)
        else:
            text(frame, f"  Left  : {gaze_l:+.2f} ({gl(gaze_l)})", (12,188), 0.46)
            text(frame, f"  Right : {gaze_r:+.2f} ({gl(gaze_r)})", (12,205), 0.46)

        text(frame, "EYES",                          (12,224),  0.46, CLR_CYAN)
        eye_lbl = "CLOSED" if eyes_closed else "open"
        text(frame, f"  State : {eye_lbl}",          (12,242),  0.46,
             CLR_YELLOW if eyes_closed else CLR_GREEN)
        text(frame, f"  EAR  L:{ear_l:.3f}  R:{ear_r:.3f}", (12,258), 0.44)
        text(frame, f"  Blinks: {self.total_blinks}", (12,274), 0.46)

        if self.session_scores:
            avg = sum(self.session_scores) / len(self.session_scores)
            text(frame, f"Session avg: {avg:.0f}%",    (w-190, 22), 0.5, CLR_CYAN)
        text(frame, f"Distractions: {len(self.distraction_log)}", (w-190, 42), 0.5, CLR_YELLOW)

    # ── Summary ───────────────────────────────────────────────────────────────

    def print_summary(self):
        print("\n" + "="*55)
        print("  ATTENTION TRACKING SESSION SUMMARY")
        print("="*55)
        if self.session_scores:
            avg  = sum(self.session_scores) / len(self.session_scores)
            mn   = float(np.min(self.session_scores))
            mx   = float(np.max(self.session_scores))
            print(f"  Average score  : {avg:.1f}%")
            print(f"  Min / Max      : {mn:.1f}% / {mx:.1f}%")
        print(f"  Total blinks   : {self.total_blinks}")
        print(f"  Distractions   : {len(self.distraction_log)}")
        if self.distraction_log:
            print("\n  Recent distractions:")
            for ts, reason in self.distraction_log[-5:]:
                t = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                print(f"    [{t}] {reason}")
        if self.csv_path:
            print(f"\n  CSV log : {self.csv_path}")
        print("="*55)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        cap.set(cv2.CAP_PROP_FPS,           30)

        print("Attention Tracker running. Press Q or Esc to quit.")

        fps_cnt = 0
        fps_t   = time.time()
        fps     = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame = cv2.flip(frame, 1)
                frame = self.process_frame(frame)

                fps_cnt += 1
                if time.time() - fps_t >= 1.0:
                    fps     = fps_cnt / (time.time() - fps_t)
                    fps_cnt = 0
                    fps_t   = time.time()

                if self.show_display:
                    h_, w_ = frame.shape[:2]
                    text(frame, f"FPS {fps:.0f}",
                         (w_-90, h_-12), 0.46, (160,160,160))
                    text(frame, "Q / Esc = quit",
                         (w_-150, h_-28), 0.40, (120,120,120))
                    cv2.imshow("Attention Tracker", frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord("q"), ord("Q"), 27):
                        break
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            if self.show_display:
                cv2.destroyAllWindows()
            self.detector.close()
            self.print_summary()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Attention Tracker — OpenCV + Mediapipe")
    p.add_argument("--camera",     type=int, default=0)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--no-csv",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    ensure_model()
    args    = parse_args()
    tracker = AttentionTracker(
        camera_index = args.camera,
        show_display = not args.no_display,
        log_csv      = not args.no_csv,
    )
    tracker.run()
