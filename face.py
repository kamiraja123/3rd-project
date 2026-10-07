"""
=============================================================================
  CYBERNETIC FACE MATRIX // 478-POINT NEURAL MESH & EXPRESSION SYSTEM
  Real-Time Multi-Face Dense Landmark Mesh, Irises & Blendshape Telemetry
  Powered by MediaPipe Tasks API (Vision 1.x) & OpenCV
=============================================================================
"""

import os
import sys
import time
import math
import urllib.request
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarksConnections

# ---------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ---------------------------------------------------------------------------
MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Color Themes (BGR Format)
THEMES = {
    "CYBERPUNK": {
        "name": "Cyberpunk Neon",
        "mesh_wire": (240, 180, 0),        # Cyan wireframe
        "points": (255, 255, 200),          # Bright electric cyan-white dots
        "contours_lips": (120, 0, 255),     # Neon Fuchsia / Magenta
        "contours_eyes": (255, 220, 0),     # Electric Cyan
        "contours_brows": (0, 220, 255),    # Golden Amber
        "contours_iris": (0, 255, 180),     # Neon Mint / Emerald
        "contours_oval": (180, 100, 255),   # Violet
        "box": (255, 242, 0),               # Cyber Cyan reticle
        "bg_glass": (14, 18, 26),
        "hud_border": (255, 242, 0),
        "accent": (255, 100, 0),
    },
    "MATRIX": {
        "name": "Matrix Terminal",
        "mesh_wire": (50, 180, 50),
        "points": (180, 255, 180),
        "contours_lips": (40, 255, 120),
        "contours_eyes": (100, 255, 150),
        "contours_brows": (50, 220, 100),
        "contours_iris": (0, 255, 255),
        "contours_oval": (30, 200, 60),
        "box": (50, 255, 100),
        "bg_glass": (10, 22, 12),
        "hud_border": (50, 255, 100),
        "accent": (20, 200, 60),
    },
    "SYNTHWAVE": {
        "name": "Synthwave Sunset",
        "mesh_wire": (200, 50, 200),
        "points": (255, 180, 255),
        "contours_lips": (0, 70, 255),      # Sunset Orange
        "contours_eyes": (255, 50, 180),    # Neon Hot Pink
        "contours_brows": (0, 180, 255),    # Amber
        "contours_iris": (255, 220, 50),    # Sky Blue
        "contours_oval": (180, 0, 220),     # Deep Purple
        "box": (255, 50, 200),
        "bg_glass": (24, 12, 28),
        "hud_border": (255, 80, 180),
        "accent": (0, 120, 255),
    },
    "DEEP_ICE": {
        "name": "Deep Ice / Arctic",
        "mesh_wire": (240, 160, 60),
        "points": (255, 240, 200),
        "contours_lips": (255, 120, 120),
        "contours_eyes": (255, 220, 100),
        "contours_brows": (240, 180, 120),
        "contours_iris": (200, 255, 255),
        "contours_oval": (200, 140, 80),
        "box": (255, 200, 100),
        "bg_glass": (18, 24, 32),
        "hud_border": (255, 200, 100),
        "accent": (255, 160, 60),
    },
}

THEME_KEYS = list(THEMES.keys())

# Key Landmark Indices for Pose Estimation & Key Features
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468


# ---------------------------------------------------------------------------
# MODEL DOWNLOAD HELPER
# ---------------------------------------------------------------------------
def ensure_face_model_exists(model_path: str = MODEL_PATH):
    """Verifies that the Face Landmarker model asset exists or downloads it."""
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
        return True

    print("=" * 65)
    print(" [DOWNLOAD] Face Landmarker Model not found locally.")
    print(f" [DOWNLOAD] Downloading from: {MODEL_URL}")
    print("=" * 65)

    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = min(int(count * block_size * 100 / total_size), 100)
                sys.stdout.write(
                    f"\rDownloading model asset: {percent}% [{count * block_size // 1024} KB / {total_size // 1024} KB]"
                )
                sys.stdout.flush()

        urllib.request.urlretrieve(MODEL_URL, model_path, reporthook)
        print("\n [SUCCESS] Model asset downloaded successfully!")
        return True
    except Exception as e:
        print(f"\n [ERROR] Failed to download model asset: {e}")
        return False


# ---------------------------------------------------------------------------
# FACIAL EXPRESSION & POSE ANALYZER
# ---------------------------------------------------------------------------
class FacialExpressionAnalyzer:
    """
    Analyzes 52 facial blendshapes and 3D geometric coordinates
    to detect real-time expressions and head orientation angles.
    """

    def __init__(self, history_len: int = 6):
        self.history = {}
        self.history_len = history_len

    def analyze_blendshapes(self, blendshapes_categories, face_id: int):
        """
        Parses blendshape categories into structured metrics and classifies primary expression.
        """
        if not blendshapes_categories:
            return "NEUTRAL", 0.95, {}, (0.0, 0.0)

        # Map categories to lookup dict
        b_dict = {cat.category_name: cat.score for cat in blendshapes_categories}

        # Core Metrics
        smile_l = b_dict.get("mouthSmileLeft", 0.0)
        smile_r = b_dict.get("mouthSmileRight", 0.0)
        smile = (smile_l + smile_r) / 2.0

        blink_l = b_dict.get("eyeBlinkLeft", 0.0)
        blink_r = b_dict.get("eyeBlinkRight", 0.0)

        jaw_open = b_dict.get("jawOpen", 0.0)
        brow_inner = b_dict.get("browInnerUp", 0.0)
        brow_outer_l = b_dict.get("browOuterUpLeft", 0.0)
        brow_outer_r = b_dict.get("browOuterUpRight", 0.0)
        brow_up = (brow_inner + brow_outer_l + brow_outer_r) / 3.0

        frown_l = b_dict.get("mouthFrownLeft", 0.0)
        frown_r = b_dict.get("mouthFrownRight", 0.0)
        frown = (frown_l + frown_r) / 2.0

        pucker = b_dict.get("mouthPucker", 0.0)
        squint_l = b_dict.get("eyeSquintLeft", 0.0)
        squint_r = b_dict.get("eyeSquintRight", 0.0)
        eye_wide = (b_dict.get("eyeWideLeft", 0.0) + b_dict.get("eyeWideRight", 0.0)) / 2.0

        metrics = {
            "Smile": smile,
            "L-Blink": blink_l,
            "R-Blink": blink_r,
            "Jaw Open": jaw_open,
            "Brows Up": brow_up,
            "Frown": frown,
            "Pucker": pucker,
            "Eye Wide": eye_wide,
        }

        # Expression Classification
        expr = "NEUTRAL"
        conf = 0.85

        if blink_l > 0.55 and blink_r > 0.55:
            expr = "EYES CLOSED"
            conf = max(blink_l, blink_r)
        elif blink_l > 0.50 and blink_r < 0.25:
            expr = "LEFT WINK"
            conf = blink_l
        elif blink_r > 0.50 and blink_l < 0.25:
            expr = "RIGHT WINK"
            conf = blink_r
        elif jaw_open > 0.45 and (brow_up > 0.25 or eye_wide > 0.25):
            expr = "SURPRISED"
            conf = jaw_open
        elif smile > 0.42:
            expr = "SMILING"
            conf = smile
        elif jaw_open > 0.35:
            expr = "MOUTH OPEN"
            conf = jaw_open
        elif pucker > 0.45:
            expr = "PUCKER"
            conf = pucker
        elif brow_up > 0.50:
            expr = "BROWS RAISED"
            conf = brow_up
        elif frown > 0.38:
            expr = "FROWNING"
            conf = frown
        else:
            expr = "NEUTRAL"
            conf = 0.90

        # Temporal smoothing to prevent rapid jitter
        if face_id not in self.history:
            self.history[face_id] = deque(maxlen=self.history_len)
        self.history[face_id].append((expr, conf))

        recent_exprs = [item[0] for item in self.history[face_id]]
        smooth_expr = max(set(recent_exprs), key=recent_exprs.count)
        smooth_conf = float(np.mean([item[1] for item in self.history[face_id] if item[0] == smooth_expr]))

        return smooth_expr, smooth_conf, metrics

    @staticmethod
    def estimate_head_pose(lms, w, h):
        """
        Estimates yaw and pitch angles from 3D facial landmarks.
        """
        nose = lms[NOSE_TIP]
        forehead = lms[FOREHEAD]
        chin = lms[CHIN]
        l_cheek = lms[LEFT_CHEEK]
        r_cheek = lms[RIGHT_CHEEK]

        # Yaw: nose horizontal distance relative to cheeks
        cheek_span = abs(r_cheek.x - l_cheek.x) + 1e-6
        cheek_mid_x = (l_cheek.x + r_cheek.x) / 2.0
        yaw_offset = (nose.x - cheek_mid_x) / cheek_span
        yaw_deg = float(np.clip(yaw_offset * 120.0, -90.0, 90.0))

        # Pitch: nose vertical distance relative to forehead and chin
        face_height = abs(chin.y - forehead.y) + 1e-6
        face_mid_y = (forehead.y + chin.y) / 2.0
        pitch_offset = (nose.y - face_mid_y) / face_height
        pitch_deg = float(np.clip(pitch_offset * 100.0, -60.0, 60.0))

        return yaw_deg, pitch_deg


# ---------------------------------------------------------------------------
# FANCY CYBERNETIC MESH & HUD VISUALIZER
# ---------------------------------------------------------------------------
class FancyFaceVisualizer:
    """
    Renders the 478-point dense mesh, facial contours, iris reticles,
    telemetry charts, and cybernetic HUD interface.
    """

    def __init__(self):
        self.theme_idx = 0
        self.toast_text = ""
        self.toast_expires = 0.0

        # Pre-convert connection lists for speed
        self.conn_tessellation = list(FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION)
        self.conn_contours = list(FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS)
        self.conn_lips = list(FaceLandmarksConnections.FACE_LANDMARKS_LIPS)
        self.conn_left_eye = list(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE)
        self.conn_right_eye = list(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE)
        self.conn_left_brow = list(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW)
        self.conn_right_brow = list(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW)
        self.conn_left_iris = list(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS)
        self.conn_right_iris = list(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS)
        self.conn_face_oval = list(FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL)

    @property
    def current_theme(self):
        return THEMES[THEME_KEYS[self.theme_idx]]

    def cycle_theme(self):
        self.theme_idx = (self.theme_idx + 1) % len(THEME_KEYS)
        return self.current_theme["name"]

    def show_toast(self, message: str, duration: float = 2.5):
        self.toast_text = message
        self.toast_expires = time.time() + duration

    @staticmethod
    def draw_glass_rect(img, x1, y1, x2, y2, bg_color=(14, 18, 26), border_color=(255, 242, 0), alpha=0.75):
        """Renders frosted semi-transparent glass panel."""
        h, w = img.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
        if x2 <= x1 or y2 <= y1:
            return

        sub = img[y1:y2, x1:x2]
        rect = np.full(sub.shape, bg_color, dtype=np.uint8)
        cv2.addWeighted(rect, alpha, sub, 1.0 - alpha, 0, sub)
        img[y1:y2, x1:x2] = sub

        if border_color:
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)

    @staticmethod
    def draw_corner_brackets(img, x1, y1, x2, y2, color, length=22, thickness=2):
        """Draws sci-fi targeting brackets around bounding box."""
        length = min(length, (x2 - x1) // 3, (y2 - y1) // 3)
        if length <= 0:
            return

        # Top-Left
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)

    def draw_tessellation_mesh(self, img, coords, color):
        """
        Renders the full 2,556-edge 3D Delaunay tessellation wireframe mesh.
        """
        for conn in self.conn_tessellation:
            p1 = coords[conn.start]
            p2 = coords[conn.end]
            cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)

    def draw_all_landmarks(self, img, lms, coords, base_color):
        """
        Renders every single point of the 478 face landmarks.
        Point brightness modulates based on z-depth for 3D realism.
        """
        for i, (cx, cy) in enumerate(coords):
            # Key feature points are larger
            if i in (NOSE_TIP, FOREHEAD, CHIN, LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER):
                cv2.circle(img, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 5, base_color, 1, cv2.LINE_AA)
            else:
                # Modulate brightness based on landmark z-depth
                z = lms[i].z
                brightness = int(np.clip(255 - (z + 0.1) * 300, 140, 255))
                dot_color = (
                    int(base_color[0] * brightness / 255),
                    int(base_color[1] * brightness / 255),
                    int(base_color[2] * brightness / 255),
                )
                cv2.circle(img, (cx, cy), 1, dot_color, -1)

    def draw_contours(self, img, coords, theme):
        """
        Renders highlighted architectural contours (Lips, Eyes, Brows, Irises, Oval).
        """
        def draw_conn_set(conns, color, thickness=1, glow=True):
            if glow:
                for conn in conns:
                    cv2.line(img, coords[conn.start], coords[conn.end], color, thickness + 2, cv2.LINE_AA)
            for conn in conns:
                cv2.line(img, coords[conn.start], coords[conn.end], color, thickness, cv2.LINE_AA)

        # 1. Lips (Neon Pink/Red)
        draw_conn_set(self.conn_lips, theme["contours_lips"], thickness=2, glow=False)

        # 2. Eyes (Electric Cyan)
        draw_conn_set(self.conn_left_eye, theme["contours_eyes"], thickness=2, glow=False)
        draw_conn_set(self.conn_right_eye, theme["contours_eyes"], thickness=2, glow=False)

        # 3. Eyebrows (Golden Amber)
        draw_conn_set(self.conn_left_brow, theme["contours_brows"], thickness=2, glow=False)
        draw_conn_set(self.conn_right_brow, theme["contours_brows"], thickness=2, glow=False)

        # 4. Irises (Glowing Emerald rings with pupil reticles)
        draw_conn_set(self.conn_left_iris, theme["contours_iris"], thickness=2, glow=False)
        draw_conn_set(self.conn_right_iris, theme["contours_iris"], thickness=2, glow=False)

        # Iris Pupil Center Reticles
        if len(coords) >= 478:
            for iris_center in (LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER):
                cx, cy = coords[iris_center]
                cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.drawMarker(img, (cx, cy), theme["contours_iris"], cv2.MARKER_CROSS, 6, 1)

        # 5. Face Oval (Silhouette boundary)
        draw_conn_set(self.conn_face_oval, theme["contours_oval"], thickness=1, glow=False)

    def draw_face_hud_card(
        self,
        img,
        coords,
        face_idx: int,
        expr_name: str,
        expr_conf: float,
        yaw: float,
        pitch: float,
        theme: dict,
        show_box: bool,
    ):
        """
        Renders sci-fi bounding box, subject badge, and pose orientation tags.
        """
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        pad = 26
        x1, y1 = max(10, min(xs) - pad), max(10, min(ys) - pad)
        x2, y2 = min(img.shape[1] - 10, max(xs) + pad), min(img.shape[0] - 10, max(ys) + pad)

        box_color = theme["box"]

        # 1. Corner Brackets Reticle
        if show_box:
            self.draw_glass_rect(img, x1, y1, x2, y2, bg_color=(10, 14, 20), border_color=None, alpha=0.12)
            self.draw_corner_brackets(img, x1, y1, x2, y2, box_color, length=24, thickness=2)

        # 2. Top Header Badge: [SUBJECT #1 // YAW +5° PITCH -2°]
        badge_text = f"SUBJECT #{face_idx + 1} // YAW:{int(yaw):+d} PITCH:{int(pitch):+d}"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        bx1 = x1
        by1 = max(10, y1 - 42)
        bx2 = bx1 + max(tw + 20, 160)
        by2 = by1 + 18
        self.draw_glass_rect(img, bx1, by1, bx2, by2, bg_color=theme["bg_glass"], border_color=box_color, alpha=0.88)
        cv2.putText(
            img,
            badge_text,
            (bx1 + 8, by2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            box_color,
            1,
            cv2.LINE_AA,
        )

        # 3. Expression Display Badge
        expr_text = f"{expr_name} [{int(expr_conf * 100)}%]"
        (ew, eh), _ = cv2.getTextSize(expr_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        ey1 = by2 + 2
        ey2 = ey1 + 22
        ex2 = bx1 + max(ew + 20, 160)
        self.draw_glass_rect(img, bx1, ey1, ex2, ey2, bg_color=(20, 26, 36), border_color=theme["accent"], alpha=0.90)
        cv2.putText(
            img,
            expr_text,
            (bx1 + 8, ey2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def draw_telemetry_sidebar(self, img, metrics: dict, frame_w: int, frame_h: int, theme: dict):
        """
        Renders right-side futuristic telemetry panel with real-time blendshape meters.
        """
        if not metrics:
            return

        panel_w = 210
        panel_h = len(metrics) * 28 + 48
        x1 = frame_w - panel_w - 14
        y1 = 64
        x2 = frame_w - 14
        y2 = y1 + panel_h

        # Frosted glass panel
        self.draw_glass_rect(img, x1, y1, x2, y2, bg_color=theme["bg_glass"], border_color=theme["hud_border"], alpha=0.85)

        # Header Title
        cv2.putText(
            img,
            "BLENDSHAPE TELEMETRY",
            (x1 + 14, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            theme["hud_border"],
            1,
            cv2.LINE_AA,
        )
        cv2.line(img, (x1 + 10, y1 + 30), (x2 - 10, y1 + 30), (60, 75, 95), 1)

        # Draw each metric bar
        bar_start_y = y1 + 48
        for i, (name, val) in enumerate(metrics.items()):
            row_y = bar_start_y + i * 28
            # Label
            cv2.putText(
                img,
                name[:9],
                (x1 + 14, row_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (200, 220, 240),
                1,
                cv2.LINE_AA,
            )

            # Meter bar background
            bar_x1 = x1 + 80
            bar_w = 80
            bar_x2 = bar_x1 + bar_w
            bar_y1 = row_y + 1
            bar_y2 = row_y + 11

            cv2.rectangle(img, (bar_x1, bar_y1), (bar_x2, bar_y2), (30, 40, 50), -1)

            # Fill bar
            fill_w = int(bar_w * np.clip(val, 0.0, 1.0))
            if fill_w > 0:
                fill_color = (
                    int(theme["accent"][0] * val + theme["box"][0] * (1 - val)),
                    int(theme["accent"][1] * val + theme["box"][1] * (1 - val)),
                    int(theme["accent"][2] * val + theme["box"][2] * (1 - val)),
                )
                cv2.rectangle(img, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), fill_color, -1)

            cv2.rectangle(img, (bar_x1, bar_y1), (bar_x2, bar_y2), (80, 100, 120), 1)

            # Value percentage text
            pct_text = f"{int(val * 100)}%"
            cv2.putText(
                img,
                pct_text,
                (bar_x2 + 6, row_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def draw_top_hud(self, img, fps: float, face_count: int, max_faces: int, frame_w: int, frame_h: int):
        """Renders futuristic header banner with FPS and 478 points telemetry."""
        bar_h = 44
        theme = self.current_theme
        border_col = theme["hud_border"]
        bg_col = theme["bg_glass"]

        # Glass banner
        self.draw_glass_rect(img, 12, 10, frame_w - 12, 10 + bar_h, bg_color=bg_col, border_color=border_col, alpha=0.82)

        # Title
        cv2.putText(
            img,
            "CYBERNETIC FACE MATRIX",
            (24, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            border_col,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "// 478-POINT DENSE MESH",
            (265, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 200, 220),
            1,
            cv2.LINE_AA,
        )

        # FPS meter
        fps_color = (0, 255, 100) if fps >= 28 else ((0, 200, 255) if fps >= 18 else (0, 70, 255))
        cv2.putText(
            img,
            f"FPS: {fps:.1f}",
            (frame_w - 380, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            fps_color,
            2,
            cv2.LINE_AA,
        )

        # Faces Tracked
        cv2.putText(
            img,
            f"FACES: {face_count}/{max_faces}",
            (frame_w - 260, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Theme indicator
        cv2.putText(
            img,
            f"THEME: {theme['name'][:10]}",
            (frame_w - 130, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 180, 200),
            1,
            cv2.LINE_AA,
        )

    def draw_bottom_hud(self, img, frame_w: int, frame_h: int):
        """Renders keyboard shortcut legend."""
        bar_h = 30
        y1 = frame_h - bar_h - 10
        y2 = frame_h - 10
        theme = self.current_theme

        self.draw_glass_rect(img, 12, y1, frame_w - 12, y2, bg_color=theme["bg_glass"], border_color=(60, 70, 90), alpha=0.75)

        help_text = (
            "[Q] Quit  |  [M] Mesh (2556 Lines)  |  [P] 478 Points  |  [C] Contours  |  "
            "[E] Telemetry  |  [B] Box  |  [T] Theme  |  [SPACE] Snapshot"
        )
        cv2.putText(
            img,
            help_text,
            (20, y2 - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (190, 210, 230),
            1,
            cv2.LINE_AA,
        )

    def draw_toast(self, img, frame_w: int, frame_h: int):
        """Displays temporary notification message."""
        if time.time() >= self.toast_expires or not self.toast_text:
            return

        (tw, th), _ = cv2.getTextSize(self.toast_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        box_w = tw + 36
        box_h = 36
        cx = frame_w // 2
        cy = 75
        x1 = cx - box_w // 2
        x2 = cx + box_w // 2
        y1 = cy - box_h // 2
        y2 = cy + box_h // 2

        self.draw_glass_rect(img, x1, y1, x2, y2, bg_color=(15, 35, 20), border_color=(50, 255, 120), alpha=0.92)
        cv2.putText(
            img,
            self.toast_text,
            (x1 + 18, y2 - 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (100, 255, 160),
            2,
            cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# MAIN FACE TRACKER APPLICATION
# ---------------------------------------------------------------------------
class FaceTrackerApp:
    """
    Main real-time Face Mesh & Expression pipeline using MediaPipe Tasks API.
    """

    def __init__(self, max_faces: int = 4, cam_index: int = 0):
        self.max_faces = max_faces
        self.cam_index = cam_index

        # Display Toggles
        self.show_mesh = True          # 2556 Delaunay Wireframe Edges
        self.show_points = True        # All 478 individual landmark dots
        self.show_contours = True      # Lips, Eyes, Eyebrows, Irises, Jawline
        self.show_box = True           # Sci-Fi Corner Brackets
        self.show_telemetry = True     # Blendshape Meter Sidebar
        self.show_hud = True           # Global HUD Overlays

        # Processing & Telemetry
        self.analyzer = FacialExpressionAnalyzer()
        self.visualizer = FancyFaceVisualizer()
        self.landmarker = None

        self.fps_buffer = deque(maxlen=20)
        self.last_time = time.perf_counter()

        # Capture output folder
        self.capture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
        os.makedirs(self.capture_dir, exist_ok=True)

    def initialize_detector(self):
        """Initializes the MediaPipe Tasks FaceLandmarker model."""
        if not ensure_face_model_exists(MODEL_PATH):
            raise RuntimeError(f"Could not load or download {MODEL_PATH}")

        print(" [INIT] Initializing MediaPipe Tasks Face Landmarker (478 Points + Blendshapes)...")
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self.max_faces,
            min_face_detection_confidence=0.50,
            min_face_presence_confidence=0.50,
            min_tracking_confidence=0.50,
            output_face_blendshapes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        print(f" [INIT] Face Landmarker Ready! Configured to track up to {self.max_faces} faces.")

    def run(self):
        """Main camera acquisition and real-time visualization loop."""
        self.initialize_detector()

        print(f" [CAMERA] Opening camera device at index {self.cam_index}...")
        cap = cv2.VideoCapture(self.cam_index)

        # Set 1280x720 resolution for crisp UI
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print(f" [WARNING] Could not open camera {self.cam_index}. Trying camera index 1...")
            cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            print(" [ERROR] No webcam could be opened! Ensure camera is connected.")
            return

        window_name = "CYBERNETIC FACE MATRIX // 478-POINT NEURAL MESH"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("=" * 65)
        print(" SYSTEM ONLINE - Real-time 478-Point Face Tracking Running")
        print(" Press [Q] or [ESC] to quit.")
        print(" Press [M] to toggle 2,556 wireframe mesh lines.")
        print(" Press [P] to toggle 478 dense landmark dots.")
        print(" Press [C] to toggle facial contours (Lips, Eyes, Irises).")
        print(" Press [E] to toggle blendshape telemetry dashboard.")
        print(" Press [T] to cycle color themes.")
        print(" Press [SPACE] to capture high-res snapshot.")
        print("=" * 65)

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    print(" [WARNING] Blank frame received. Retrying...")
                    time.sleep(0.02)
                    continue

                # Mirror frame horizontally for natural user experience
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                # Convert BGR to RGB for MediaPipe Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Inference via Tasks API
                result = self.landmarker.detect(mp_image)

                # FPS calculation
                now = time.perf_counter()
                dt = now - self.last_time
                self.last_time = now
                if dt > 0:
                    self.fps_buffer.append(1.0 / dt)
                fps = float(np.mean(self.fps_buffer)) if self.fps_buffer else 30.0

                face_count = len(result.face_landmarks) if result and result.face_landmarks else 0
                theme = self.visualizer.current_theme
                primary_metrics = {}

                if face_count > 0:
                    for i in range(face_count):
                        lms = result.face_landmarks[i]

                        # Pixel coordinates for all 478 landmarks
                        coords = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

                        # 1. Blendshapes & Expressions Analysis
                        blendshapes = result.face_blendshapes[i] if (result.face_blendshapes and i < len(result.face_blendshapes)) else []
                        expr_name, expr_conf, metrics = self.analyzer.analyze_blendshapes(blendshapes, face_id=i)
                        if i == 0:
                            primary_metrics = metrics

                        # 2. Head Orientation Angles
                        yaw, pitch = self.analyzer.estimate_head_pose(lms, w, h)

                        # 3. Layer 1: Wireframe Tessellation (2,556 Edges)
                        if self.show_mesh:
                            self.visualizer.draw_tessellation_mesh(frame, coords, theme["mesh_wire"])

                        # 4. Layer 2: Contours (Lips, Eyes, Brows, Irises, Oval)
                        if self.show_contours:
                            self.visualizer.draw_contours(frame, coords, theme)

                        # 5. Layer 3: Dense Landmark Points (All 478 Points)
                        if self.show_points:
                            self.visualizer.draw_all_landmarks(frame, lms, coords, theme["points"])

                        # 6. Face Reticle & Card
                        if self.show_hud:
                            self.visualizer.draw_face_hud_card(
                                frame,
                                coords,
                                face_idx=i,
                                expr_name=expr_name,
                                expr_conf=expr_conf,
                                yaw=yaw,
                                pitch=pitch,
                                theme=theme,
                                show_box=self.show_box,
                            )

                # Global HUD Overlays
                if self.show_hud:
                    self.visualizer.draw_top_hud(frame, fps, face_count, self.max_faces, w, h)
                    self.visualizer.draw_bottom_hud(frame, w, h)

                    if self.show_telemetry and primary_metrics:
                        self.visualizer.draw_telemetry_sidebar(frame, primary_metrics, w, h, theme)

                    self.visualizer.draw_toast(frame, w, h)

                cv2.imshow(window_name, frame)

                # Keyboard Controls
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    print(" [EXIT] Exit requested by user.")
                    break
                elif key in (ord("m"), ord("M")):
                    self.show_mesh = not self.show_mesh
                    status = "ENABLED" if self.show_mesh else "DISABLED"
                    self.visualizer.show_toast(f"TESSELLATION MESH: {status}")
                elif key in (ord("p"), ord("P")):
                    self.show_points = not self.show_points
                    status = "ENABLED" if self.show_points else "DISABLED"
                    self.visualizer.show_toast(f"478 POINTS: {status}")
                elif key in (ord("c"), ord("C")):
                    self.show_contours = not self.show_contours
                    status = "ENABLED" if self.show_contours else "DISABLED"
                    self.visualizer.show_toast(f"FACIAL CONTOURS: {status}")
                elif key in (ord("e"), ord("E")):
                    self.show_telemetry = not self.show_telemetry
                    status = "ENABLED" if self.show_telemetry else "DISABLED"
                    self.visualizer.show_toast(f"TELEMETRY PANEL: {status}")
                elif key in (ord("b"), ord("B")):
                    self.show_box = not self.show_box
                    status = "ENABLED" if self.show_box else "DISABLED"
                    self.visualizer.show_toast(f"TARGET BOX: {status}")
                elif key in (ord("t"), ord("T")):
                    new_theme = self.visualizer.cycle_theme()
                    self.visualizer.show_toast(f"THEME: {new_theme}")
                elif key in (ord("h"), ord("H")):
                    self.show_hud = not self.show_hud
                elif key == 32:  # SPACE
                    filename = f"face_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    filepath = os.path.join(self.capture_dir, filename)
                    cv2.imwrite(filepath, frame)
                    self.visualizer.show_toast(f"SNAPSHOT SAVED: {filename}")
                    print(f" [SNAPSHOT] Saved high-res capture to: {filepath}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.landmarker:
                self.landmarker.close()
            print(" [EXIT] Face Tracker clean shutdown complete.")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = FaceTrackerApp(max_faces=4, cam_index=0)
    app.run()
