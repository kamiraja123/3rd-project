"""
=============================================================================
  CYBERNETIC BODY MATRIX // 33-POINT FULL BODY POSE & BIOMECHANICS AI
  Real-Time Full Body Tracking, Joint Angles, Posture & Rep Counter
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
from mediapipe.tasks.python.vision import PoseLandmarksConnections

# ---------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ---------------------------------------------------------------------------
MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

# Color Themes (BGR Format)
THEMES = {
    "CYBERPUNK": {
        "name": "Cyberpunk Neon",
        "primary": (255, 242, 0),        # Electric Cyan
        "secondary": (147, 20, 255),     # Neon Magenta
        "accent": (0, 255, 128),         # Lime
        "torso_fill": (255, 150, 0),     # Cyan-Blue Glow
        "joint_halo": (0, 200, 255),     # Amber Gold
        "bg_glass": (14, 18, 26),
        "hud_border": (255, 242, 0),
        "text": (255, 255, 255),
    },
    "MATRIX": {
        "name": "Matrix Protocol",
        "primary": (50, 255, 100),       # Matrix Emerald
        "secondary": (0, 220, 60),       # Toxic Green
        "accent": (150, 255, 200),       # Mint Glow
        "torso_fill": (30, 180, 50),     # Dark Green Armor
        "joint_halo": (0, 255, 120),
        "bg_glass": (10, 22, 12),
        "hud_border": (50, 255, 100),
        "text": (220, 255, 230),
    },
    "SYNTHWAVE": {
        "name": "Synthwave Sunset",
        "primary": (255, 80, 180),       # Hot Pink
        "secondary": (0, 140, 255),      # Sun Orange
        "accent": (255, 220, 50),        # Cyan Gold
        "torso_fill": (180, 0, 220),     # Violet Armor
        "joint_halo": (0, 180, 255),
        "bg_glass": (24, 12, 28),
        "hud_border": (255, 80, 180),
        "text": (255, 240, 250),
    },
    "DEEP_ICE": {
        "name": "Frost Arctic",
        "primary": (255, 200, 100),      # Sky Blue
        "secondary": (240, 240, 150),    # Ice Cyan
        "accent": (255, 160, 200),       # Ice Violet
        "torso_fill": (200, 150, 80),    # Deep Blue Armor
        "joint_halo": (255, 220, 120),
        "bg_glass": (18, 24, 32),
        "hud_border": (255, 200, 100),
        "text": (240, 250, 255),
    },
}

THEME_KEYS = list(THEMES.keys())

# 33 Standard Body Landmarks
NOSE = 0
LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER = 1, 2, 3
RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER = 4, 5, 6
LEFT_EAR, RIGHT_EAR = 7, 8
MOUTH_LEFT, MOUTH_RIGHT = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_THUMB, RIGHT_THUMB = 21, 22
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

# Body Connection Sets
LIMBS_LEFT = [
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (LEFT_WRIST, LEFT_THUMB), (LEFT_WRIST, LEFT_INDEX), (LEFT_WRIST, LEFT_PINKY),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (LEFT_ANKLE, LEFT_HEEL), (LEFT_HEEL, LEFT_FOOT_INDEX), (LEFT_ANKLE, LEFT_FOOT_INDEX),
]

LIMBS_RIGHT = [
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (RIGHT_WRIST, RIGHT_THUMB), (RIGHT_WRIST, RIGHT_INDEX), (RIGHT_WRIST, RIGHT_PINKY),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    (RIGHT_ANKLE, RIGHT_HEEL), (RIGHT_HEEL, RIGHT_FOOT_INDEX), (RIGHT_ANKLE, RIGHT_FOOT_INDEX),
]

TORSO_BONES = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
]

HEAD_BONES = [
    (NOSE, LEFT_EYE), (LEFT_EYE, LEFT_EAR),
    (NOSE, RIGHT_EYE), (RIGHT_EYE, RIGHT_EAR),
    (MOUTH_LEFT, MOUTH_RIGHT),
]

TORSO_POLYGON = [LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_HIP, LEFT_HIP]

KEY_JOINTS = [
    LEFT_SHOULDER, RIGHT_SHOULDER,
    LEFT_ELBOW, RIGHT_ELBOW,
    LEFT_WRIST, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
]


# ---------------------------------------------------------------------------
# MODEL DOWNLOAD HELPER
# ---------------------------------------------------------------------------
def ensure_pose_model_exists(model_path: str = MODEL_PATH):
    """Verifies that the Pose Landmarker model asset exists or downloads it."""
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
        return True

    print("=" * 65)
    print(" [DOWNLOAD] Pose Landmarker Model not found locally.")
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
        print(f"\n [ERROR] Failed to download pose model: {e}")
        return False


# ---------------------------------------------------------------------------
# BIOMECHANICS & POSE ANALYZER
# ---------------------------------------------------------------------------
class BiomechanicsAnalyzer:
    """
    Computes 3D joint angles, biomechanical posture alignment,
    action classification, and exercise repetition tracking.
    """

    def __init__(self, history_len: int = 5):
        self.history = deque(maxlen=history_len)
        # Squat Rep Counter State
        self.squat_state = "UP"
        self.squat_reps = 0

    @staticmethod
    def calculate_angle(a, b, c):
        """
        Calculates angle ABC (in degrees) at vertex B.
        Supports 2D coordinate tuples.
        """
        ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32)
        bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float32)
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba == 0 or norm_bc == 0:
            return 180.0
        cos_angle = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def analyze_pose(self, lms, coords):
        """
        Calculates key joint angles, posture metrics, and classifies pose.
        """
        # Joint Angles
        l_elbow_ang = self.calculate_angle(coords[LEFT_SHOULDER], coords[LEFT_ELBOW], coords[LEFT_WRIST])
        r_elbow_ang = self.calculate_angle(coords[RIGHT_SHOULDER], coords[RIGHT_ELBOW], coords[RIGHT_WRIST])
        l_knee_ang = self.calculate_angle(coords[LEFT_HIP], coords[LEFT_KNEE], coords[LEFT_ANKLE])
        r_knee_ang = self.calculate_angle(coords[RIGHT_HIP], coords[RIGHT_KNEE], coords[RIGHT_ANKLE])
        l_shoulder_ang = self.calculate_angle(coords[LEFT_HIP], coords[LEFT_SHOULDER], coords[LEFT_ELBOW])
        r_shoulder_ang = self.calculate_angle(coords[RIGHT_HIP], coords[RIGHT_SHOULDER], coords[RIGHT_ELBOW])

        # Shoulder Tilt (Horizontal deviation)
        ls, rs = coords[LEFT_SHOULDER], coords[RIGHT_SHOULDER]
        dx = rs[0] - ls[0]
        dy = rs[1] - ls[1]
        shoulder_tilt = float(np.degrees(np.arctan2(dy, dx))) if dx != 0 else 0.0

        # Spine Vertical Alignment (midpoint shoulders to midpoint hips)
        mid_sh = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
        lh, rh = coords[LEFT_HIP], coords[RIGHT_HIP]
        mid_hp = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
        spine_dx = mid_sh[0] - mid_hp[0]
        spine_dy = mid_sh[1] - mid_hp[1]
        spine_tilt = float(np.degrees(np.arctan2(spine_dx, abs(spine_dy) + 1e-6)))

        # Relative Positions
        wrists_above_head = (coords[LEFT_WRIST][1] < coords[NOSE][1]) and (coords[RIGHT_WRIST][1] < coords[NOSE][1])
        r_wrist_up = coords[RIGHT_WRIST][1] < coords[NOSE][1]
        l_wrist_up = coords[LEFT_WRIST][1] < coords[NOSE][1]
        t_pose_arms = (
            abs(coords[LEFT_WRIST][1] - coords[LEFT_SHOULDER][1]) < 45
            and abs(coords[RIGHT_WRIST][1] - coords[RIGHT_SHOULDER][1]) < 45
            and l_elbow_ang > 150
            and r_elbow_ang > 150
        )
        arms_crossed = abs(coords[LEFT_WRIST][0] - coords[RIGHT_SHOULDER][0]) < 80 and abs(
            coords[RIGHT_WRIST][0] - coords[LEFT_SHOULDER][0]
        ) < 80

        avg_knee = (l_knee_ang + r_knee_ang) / 2.0

        # Pose Classification
        pose_name = "STANDING"
        confidence = 0.90

        if wrists_above_head:
            pose_name = "HANDS UP // VICTORY"
            confidence = 0.95
        elif t_pose_arms:
            pose_name = "T-POSE"
            confidence = 0.94
        elif arms_crossed:
            pose_name = "ARMS CROSSED"
            confidence = 0.91
        elif r_wrist_up and not l_wrist_up:
            pose_name = "RIGHT HAND RAISED"
            confidence = 0.92
        elif l_wrist_up and not r_wrist_up:
            pose_name = "LEFT HAND RAISED"
            confidence = 0.92
        elif avg_knee < 115:
            pose_name = "SQUAT"
            confidence = 0.94
        elif avg_knee < 140:
            pose_name = "HALF SQUAT / CHAIR"
            confidence = 0.88
        elif abs(spine_tilt) > 18:
            pose_name = "LEANING" if spine_tilt > 0 else "LEANING"
            confidence = 0.85
        else:
            pose_name = "STANDING"
            confidence = 0.92

        # Squat Repetition Counter Logic
        if avg_knee < 110:
            if self.squat_state == "UP":
                self.squat_state = "DOWN"
        elif avg_knee > 155:
            if self.squat_state == "DOWN":
                self.squat_state = "UP"
                self.squat_reps += 1

        # Posture Status
        posture_status = "OPTIMAL"
        if abs(shoulder_tilt) > 8:
            posture_status = "SHOULDER TILT"
        elif abs(spine_tilt) > 12:
            posture_status = "UNALIGNED SPINE"

        angles = {
            "L-Elbow": l_elbow_ang,
            "R-Elbow": r_elbow_ang,
            "L-Knee": l_knee_ang,
            "R-Knee": r_knee_ang,
            "L-Shoulder": l_shoulder_ang,
            "R-Shoulder": r_shoulder_ang,
            "Shoulder Tilt": shoulder_tilt,
            "Spine Tilt": spine_tilt,
        }

        # Temporal smoothing
        self.history.append((pose_name, confidence))
        recent = [item[0] for item in self.history]
        smooth_pose = max(set(recent), key=recent.count)

        return smooth_pose, confidence, angles, posture_status, self.squat_reps

    def reset_reps(self):
        self.squat_reps = 0


# ---------------------------------------------------------------------------
# FANCY CYBERNETIC SKELETON & HUD VISUALIZER
# ---------------------------------------------------------------------------
class FancyBodyVisualizer:
    """
    Renders high-tech Cyberpunk full-body HUD:
    - Double-layer glowing bones & joints
    - Translucent cybernetic torso armor polygon
    - Sci-fi angle badges directly at joint pivots
    - Corner bracket bounding box
    - Biomechanics & rep counter telemetry dashboard
    - Toast notifications and system telemetry
    """

    def __init__(self):
        self.theme_idx = 0
        self.toast_text = ""
        self.toast_expires = 0.0

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
    def draw_corner_brackets(img, x1, y1, x2, y2, color, length=24, thickness=2):
        """Draws futuristic target reticle corner brackets."""
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

    def draw_torso_armor(self, img, coords, theme):
        """Renders semi-transparent cybernetic torso armor polygon and spine line."""
        pts = np.array([coords[idx] for idx in TORSO_POLYGON], np.int32)
        pts = pts.reshape((-1, 1, 2))
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], theme["torso_fill"])
        cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)
        cv2.polylines(img, [pts], True, theme["primary"], 2, cv2.LINE_AA)

        # Spine center line
        mid_sh = (
            (coords[LEFT_SHOULDER][0] + coords[RIGHT_SHOULDER][0]) // 2,
            (coords[LEFT_SHOULDER][1] + coords[RIGHT_SHOULDER][1]) // 2,
        )
        mid_hp = (
            (coords[LEFT_HIP][0] + coords[RIGHT_HIP][0]) // 2,
            (coords[LEFT_HIP][1] + coords[RIGHT_HIP][1]) // 2,
        )
        cv2.line(img, mid_sh, mid_hp, theme["secondary"], 2, cv2.LINE_AA)

    def draw_skeleton(self, img, coords, theme, show_torso_armor: bool):
        """
        Renders full 33-point body skeleton with glowing dual-layer bones.
        """
        primary = theme["primary"]
        secondary = theme["secondary"]
        joint_halo = theme["joint_halo"]

        # 1. Torso Armor Mesh
        if show_torso_armor:
            self.draw_torso_armor(img, coords, theme)

        # 2. Glowing Bones (Underlay Glow)
        glow_layer = img.copy()
        all_segments = LIMBS_LEFT + LIMBS_RIGHT + TORSO_BONES + HEAD_BONES
        for s, e in all_segments:
            cv2.line(glow_layer, coords[s], coords[e], secondary, 7, cv2.LINE_AA)
        cv2.addWeighted(glow_layer, 0.45, img, 0.55, 0, img)

        # 3. Crisp Inner Bones (Left side primary, Right side secondary)
        for s, e in LIMBS_LEFT:
            cv2.line(img, coords[s], coords[e], primary, 2, cv2.LINE_AA)
        for s, e in LIMBS_RIGHT:
            cv2.line(img, coords[s], coords[e], secondary, 2, cv2.LINE_AA)
        for s, e in TORSO_BONES:
            cv2.line(img, coords[s], coords[e], primary, 2, cv2.LINE_AA)
        for s, e in HEAD_BONES:
            cv2.line(img, coords[s], coords[e], (255, 255, 255), 1, cv2.LINE_AA)

        # 4. Joint Nodes & Halos
        for idx in range(33):
            cx, cy = coords[idx]
            if idx in KEY_JOINTS:
                # Major Joint Target Reticle
                cv2.circle(img, (cx, cy), 9, joint_halo, 1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 5, primary, -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)
            elif idx == NOSE:
                cv2.circle(img, (cx, cy), 6, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                # Minor Joint Dot
                cv2.circle(img, (cx, cy), 3, secondary, -1, cv2.LINE_AA)

    def draw_joint_angle_badges(self, img, coords, angles: dict, theme: dict):
        """
        Draws sci-fi angle callouts directly next to elbows and knees.
        """
        targets = [
            (LEFT_ELBOW, "L-Elbow", angles.get("L-Elbow", 0), -45, -10),
            (RIGHT_ELBOW, "R-Elbow", angles.get("R-Elbow", 0), 20, -10),
            (LEFT_KNEE, "L-Knee", angles.get("L-Knee", 0), -45, 10),
            (RIGHT_KNEE, "R-Knee", angles.get("R-Knee", 0), 20, 10),
        ]

        for joint_idx, label, deg, off_x, off_y in targets:
            cx, cy = coords[joint_idx]
            bx, by = cx + off_x, cy + off_y
            text = f"{int(deg)}°"

            # Mini badge
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            self.draw_glass_rect(img, bx - 2, by - th - 3, bx + tw + 4, by + 3, bg_color=(12, 16, 24), border_color=theme["primary"], alpha=0.85)
            cv2.putText(
                img,
                text,
                (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            # Connector tick
            cv2.line(img, (cx, cy), (bx if off_x > 0 else bx + tw, by), theme["secondary"], 1, cv2.LINE_AA)

    def draw_subject_hud_card(
        self,
        img,
        coords,
        subject_idx: int,
        pose_name: str,
        pose_conf: float,
        posture_status: str,
        reps: int,
        theme: dict,
        show_box: bool,
    ):
        """
        Renders bounding box reticle, subject identity badge, and exercise rep counter.
        """
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        pad = 28
        x1, y1 = max(10, min(xs) - pad), max(10, min(ys) - pad)
        x2, y2 = min(img.shape[1] - 10, max(xs) + pad), min(img.shape[0] - 10, max(ys) + pad)

        primary = theme["primary"]
        secondary = theme["secondary"]

        # 1. Sci-Fi Corner Brackets
        if show_box:
            self.draw_glass_rect(img, x1, y1, x2, y2, bg_color=(10, 14, 20), border_color=None, alpha=0.10)
            self.draw_corner_brackets(img, x1, y1, x2, y2, primary, length=28, thickness=2)

        # 2. Top Header Badge: [SUBJECT #1 // REPS: 5]
        badge_text = f"SUBJECT #{subject_idx + 1} // SQUAT REPS: {reps}"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        bx1 = x1
        by1 = max(10, y1 - 42)
        bx2 = bx1 + max(tw + 20, 180)
        by2 = by1 + 18
        self.draw_glass_rect(img, bx1, by1, bx2, by2, bg_color=theme["bg_glass"], border_color=primary, alpha=0.88)
        cv2.putText(
            img,
            badge_text,
            (bx1 + 8, by2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            primary,
            1,
            cv2.LINE_AA,
        )

        # 3. Pose Display Pill
        pose_text = f"{pose_name} [{int(pose_conf * 100)}%]"
        (pw, ph), _ = cv2.getTextSize(pose_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        py1 = by2 + 2
        py2 = py1 + 22
        px2 = bx1 + max(pw + 20, 180)
        self.draw_glass_rect(img, bx1, py1, px2, py2, bg_color=(20, 26, 36), border_color=secondary, alpha=0.90)
        cv2.putText(
            img,
            pose_text,
            (bx1 + 8, py2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def draw_telemetry_sidebar(self, img, angles: dict, posture_status: str, reps: int, frame_w: int, frame_h: int, theme: dict):
        """
        Renders right-side futuristic telemetry panel with biomechanical angles and posture assessment.
        """
        panel_w = 215
        panel_h = 290
        x1 = frame_w - panel_w - 14
        y1 = 64
        x2 = frame_w - 14
        y2 = y1 + panel_h

        self.draw_glass_rect(img, x1, y1, x2, y2, bg_color=theme["bg_glass"], border_color=theme["hud_border"], alpha=0.85)

        # Panel Header
        cv2.putText(
            img,
            "BIOMECHANICS TELEMETRY",
            (x1 + 12, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            theme["hud_border"],
            1,
            cv2.LINE_AA,
        )
        cv2.line(img, (x1 + 10, y1 + 30), (x2 - 10, y1 + 30), (60, 75, 95), 1)

        # Squat Repetition Display Banner
        rep_box_y = y1 + 40
        self.draw_glass_rect(img, x1 + 12, rep_box_y, x2 - 12, rep_box_y + 34, bg_color=(20, 36, 30), border_color=(0, 255, 120), alpha=0.90)
        cv2.putText(
            img,
            f"SQUAT REPS:  {reps}",
            (x1 + 24, rep_box_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (50, 255, 140),
            2,
            cv2.LINE_AA,
        )

        # Posture Status
        status_color = (0, 255, 100) if posture_status == "OPTIMAL" else (0, 180, 255)
        cv2.putText(
            img,
            f"POSTURE: {posture_status}",
            (x1 + 14, y1 + 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            status_color,
            1,
            cv2.LINE_AA,
        )

        # Angle Rows
        metric_items = [
            ("L-Elbow", f"{int(angles.get('L-Elbow', 0))}°"),
            ("R-Elbow", f"{int(angles.get('R-Elbow', 0))}°"),
            ("L-Knee", f"{int(angles.get('L-Knee', 0))}°"),
            ("R-Knee", f"{int(angles.get('R-Knee', 0))}°"),
            ("Shoulder Tilt", f"{angles.get('Shoulder Tilt', 0):+.1f}°"),
            ("Spine Tilt", f"{angles.get('Spine Tilt', 0):+.1f}°"),
        ]

        row_start_y = y1 + 120
        for i, (lbl, val_str) in enumerate(metric_items):
            ry = row_start_y + i * 26
            cv2.putText(
                img,
                lbl,
                (x1 + 14, ry + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (190, 210, 230),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                val_str,
                (x2 - 58, ry + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                theme["primary"],
                1,
                cv2.LINE_AA,
            )
            cv2.line(img, (x1 + 12, ry + 18), (x2 - 12, ry + 18), (35, 45, 60), 1)

    def draw_top_hud(self, img, fps: float, body_count: int, max_bodies: int, frame_w: int, frame_h: int):
        """Renders futuristic header banner with FPS, body count, and system telemetry."""
        bar_h = 44
        theme = self.current_theme
        border_col = theme["hud_border"]
        bg_col = theme["bg_glass"]

        # Glass panel header
        self.draw_glass_rect(img, 12, 10, frame_w - 12, 10 + bar_h, bg_color=bg_col, border_color=border_col, alpha=0.82)

        # Title
        cv2.putText(
            img,
            "CYBERNETIC BODY MATRIX",
            (24, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            border_col,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "// 33-POINT FULL BODY AI",
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

        # Tracked Count
        cv2.putText(
            img,
            f"BODIES: {body_count}/{max_bodies}",
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
            "[Q] Quit  |  [S] Skeleton  |  [A] Joint Angles  |  [T] Torso Armor  |  "
            "[E] Telemetry  |  [R] Reset Reps  |  [C] Theme  |  [SPACE] Snapshot"
        )
        cv2.putText(
            img,
            help_text,
            (18, y2 - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (190, 210, 230),
            1,
            cv2.LINE_AA,
        )

    def draw_toast(self, img, frame_w: int, frame_h: int):
        """Displays temporary notification toast."""
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
# MAIN FULL BODY TRACKER APPLICATION
# ---------------------------------------------------------------------------
class FullBodyTrackerApp:
    """
    Main real-time Full Body Pose Tracking & Biomechanics pipeline.
    """

    def __init__(self, max_bodies: int = 2, cam_index: int = 0):
        self.max_bodies = max_bodies
        self.cam_index = cam_index

        # Display Toggles
        self.show_skeleton = True
        self.show_torso_armor = True
        self.show_angles = True
        self.show_box = True
        self.show_telemetry = True
        self.show_hud = True

        # Pipeline components
        self.analyzer = BiomechanicsAnalyzer()
        self.visualizer = FancyBodyVisualizer()
        self.landmarker = None

        # Telemetry
        self.fps_buffer = deque(maxlen=20)
        self.last_time = time.perf_counter()

        # Capture output folder
        self.capture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
        os.makedirs(self.capture_dir, exist_ok=True)

    def initialize_detector(self):
        """Initializes the MediaPipe Tasks PoseLandmarker model."""
        if not ensure_pose_model_exists(MODEL_PATH):
            raise RuntimeError(f"Could not load or download {MODEL_PATH}")

        print(" [INIT] Initializing MediaPipe Tasks Pose Landmarker (33 Full-Body Points)...")
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=self.max_bodies,
            min_pose_detection_confidence=0.50,
            min_pose_presence_confidence=0.50,
            min_tracking_confidence=0.50,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        print(f" [INIT] Pose Landmarker Ready! Configured to track up to {self.max_bodies} bodies.")

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

        window_name = "CYBERNETIC BODY MATRIX // 33-POINT FULL BODY AI"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("=" * 65)
        print(" SYSTEM ONLINE - Real-time 33-Point Full Body Tracking Running")
        print(" Press [Q] or [ESC] to quit.")
        print(" Press [S] to toggle full-body skeleton.")
        print(" Press [A] to toggle real-time joint angles callout.")
        print(" Press [T] to toggle cybernetic torso armor.")
        print(" Press [E] to toggle biomechanics telemetry panel.")
        print(" Press [R] to reset squat repetition counter.")
        print(" Press [C] to cycle neon color themes.")
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

                body_count = len(result.pose_landmarks) if result and result.pose_landmarks else 0
                theme = self.visualizer.current_theme
                primary_angles = {}
                primary_status = "OPTIMAL"
                primary_reps = self.analyzer.squat_reps

                if body_count > 0:
                    for i in range(body_count):
                        lms = result.pose_landmarks[i]

                        # Pixel coordinates for all 33 body landmarks
                        coords = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

                        # 1. Biomechanics & Pose Analysis
                        pose_name, pose_conf, angles, posture_status, reps = self.analyzer.analyze_pose(lms, coords)
                        if i == 0:
                            primary_angles = angles
                            primary_status = posture_status
                            primary_reps = reps

                        # 2. Skeleton & Torso Armor Rendering
                        if self.show_skeleton:
                            self.visualizer.draw_skeleton(frame, coords, theme, self.show_torso_armor)

                        # 3. Joint Angle Badges at Pivots
                        if self.show_angles:
                            self.visualizer.draw_joint_angle_badges(frame, coords, angles, theme)

                        # 4. Subject HUD Reticle Card
                        if self.show_hud:
                            self.visualizer.draw_subject_hud_card(
                                frame,
                                coords,
                                subject_idx=i,
                                pose_name=pose_name,
                                pose_conf=pose_conf,
                                posture_status=posture_status,
                                reps=reps,
                                theme=theme,
                                show_box=self.show_box,
                            )

                # Global HUD Overlays
                if self.show_hud:
                    self.visualizer.draw_top_hud(frame, fps, body_count, self.max_bodies, w, h)
                    self.visualizer.draw_bottom_hud(frame, w, h)

                    if self.show_telemetry and primary_angles:
                        self.visualizer.draw_telemetry_sidebar(
                            frame, primary_angles, primary_status, primary_reps, w, h, theme
                        )

                    self.visualizer.draw_toast(frame, w, h)

                cv2.imshow(window_name, frame)

                # Keyboard Controls
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    print(" [EXIT] Exit requested by user.")
                    break
                elif key in (ord("s"), ord("S")):
                    self.show_skeleton = not self.show_skeleton
                    status = "ENABLED" if self.show_skeleton else "DISABLED"
                    self.visualizer.show_toast(f"SKELETON: {status}")
                elif key in (ord("a"), ord("A")):
                    self.show_angles = not self.show_angles
                    status = "ENABLED" if self.show_angles else "DISABLED"
                    self.visualizer.show_toast(f"JOINT ANGLES: {status}")
                elif key in (ord("t"), ord("T")):
                    self.show_torso_armor = not self.show_torso_armor
                    status = "ENABLED" if self.show_torso_armor else "DISABLED"
                    self.visualizer.show_toast(f"TORSO ARMOR: {status}")
                elif key in (ord("e"), ord("E")):
                    self.show_telemetry = not self.show_telemetry
                    status = "ENABLED" if self.show_telemetry else "DISABLED"
                    self.visualizer.show_toast(f"TELEMETRY PANEL: {status}")
                elif key in (ord("r"), ord("R")):
                    self.analyzer.reset_reps()
                    self.visualizer.show_toast("SQUAT REPS RESET TO 0")
                elif key in (ord("b"), ord("B")):
                    self.show_box = not self.show_box
                    status = "ENABLED" if self.show_box else "DISABLED"
                    self.visualizer.show_toast(f"TARGET BOX: {status}")
                elif key in (ord("c"), ord("C")):
                    new_theme = self.visualizer.cycle_theme()
                    self.visualizer.show_toast(f"THEME: {new_theme}")
                elif key in (ord("h"), ord("H")):
                    self.show_hud = not self.show_hud
                elif key == 32:  # SPACE
                    filename = f"body_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    filepath = os.path.join(self.capture_dir, filename)
                    cv2.imwrite(filepath, frame)
                    self.visualizer.show_toast(f"SNAPSHOT SAVED: {filename}")
                    print(f" [SNAPSHOT] Saved high-res capture to: {filepath}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.landmarker:
                self.landmarker.close()
            print(" [EXIT] Full Body Tracker clean shutdown complete.")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = FullBodyTrackerApp(max_bodies=2, cam_index=0)
    app.run()
