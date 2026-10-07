"""
=============================================================================
  CYBERNETIC HAND MATRIX // AI VISION SYSTEM
  Real-Time Multi-Hand Skeleton Tracking & Gesture Recognition
  Powered by MediaPipe 1.x & OpenCV
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

# ---------------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ---------------------------------------------------------------------------
MODEL_PATH = "gesture_recognizer.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)

# Color Themes (BGR Format)
THEMES = {
    "CYBERPUNK": {
        "name": "Cyberpunk Neon",
        "hands": [
            {"primary": (255, 242, 0), "secondary": (255, 100, 0), "name": "Cyan"},
            {"primary": (147, 20, 255), "secondary": (255, 0, 128), "name": "Magenta"},
            {"primary": (0, 255, 128), "secondary": (0, 200, 255), "name": "Lime"},
            {"primary": (0, 180, 255), "secondary": (0, 100, 255), "name": "Amber"},
        ],
        "bg_glass": (15, 18, 24),
        "hud_border": (255, 242, 0),
        "text_primary": (255, 255, 255),
    },
    "MATRIX": {
        "name": "Matrix Protocol",
        "hands": [
            {"primary": (50, 255, 100), "secondary": (20, 180, 50), "name": "Emerald"},
            {"primary": (120, 255, 200), "secondary": (50, 200, 150), "name": "Mint"},
            {"primary": (0, 255, 50), "secondary": (0, 150, 30), "name": "Acid"},
            {"primary": (200, 255, 0), "secondary": (100, 200, 0), "name": "Teal"},
        ],
        "bg_glass": (10, 20, 12),
        "hud_border": (50, 255, 100),
        "text_primary": (220, 255, 230),
    },
    "RETRO_WAVE": {
        "name": "Retro Sunset",
        "hands": [
            {"primary": (255, 80, 180), "secondary": (200, 0, 255), "name": "Neon Pink"},
            {"primary": (0, 160, 255), "secondary": (0, 80, 255), "name": "Sun Orange"},
            {"primary": (255, 220, 50), "secondary": (255, 120, 0), "name": "Electric Gold"},
            {"primary": (255, 50, 80), "secondary": (180, 0, 120), "name": "Crimson Glow"},
        ],
        "bg_glass": (22, 12, 28),
        "hud_border": (255, 80, 180),
        "text_primary": (255, 240, 250),
    },
    "DEEP_ICE": {
        "name": "Frost Arctic",
        "hands": [
            {"primary": (255, 200, 100), "secondary": (255, 120, 50), "name": "Sky Blue"},
            {"primary": (240, 240, 150), "secondary": (200, 200, 100), "name": "Glacier"},
            {"primary": (255, 160, 200), "secondary": (200, 100, 255), "name": "Ice Violet"},
            {"primary": (200, 255, 255), "secondary": (120, 200, 255), "name": "Diamond"},
        ],
        "bg_glass": (20, 24, 30),
        "hud_border": (255, 200, 100),
        "text_primary": (240, 250, 255),
    },
}

THEME_KEYS = list(THEMES.keys())

# Hand Landmark Indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# Bone Segments for skeleton rendering
HAND_BONES = [
    # Thumb
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    # Index
    (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    # Middle
    (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    # Ring
    (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    # Pinky
    (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    # Palm Webbing / Metacarpals
    (WRIST, INDEX_MCP), (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, RING_MCP),
    (RING_MCP, PINKY_MCP), (PINKY_MCP, WRIST),
]

PALM_POLYGON = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


# ---------------------------------------------------------------------------
# MODEL DOWNLOAD HELPER
# ---------------------------------------------------------------------------
def ensure_model_exists(model_path: str = MODEL_PATH):
    """Verifies that the model file exists, or downloads it with progress reporting."""
    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
        return True

    print("=" * 65)
    print(" [DOWNLOAD] Gesture Recognizer Model not found locally.")
    print(f" [DOWNLOAD] Fetching from: {MODEL_URL}")
    print("=" * 65)

    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = min(int(count * block_size * 100 / total_size), 100)
                sys.stdout.write(f"\rDownloading model asset: {percent}% [{count * block_size // 1024} KB / {total_size // 1024} KB]")
                sys.stdout.flush()

        urllib.request.urlretrieve(MODEL_URL, model_path, reporthook)
        print("\n [SUCCESS] Model asset downloaded successfully!")
        return True
    except Exception as e:
        print(f"\n [ERROR] Failed to download model: {e}")
        return False


# ---------------------------------------------------------------------------
# GESTURE DETECTOR & HEURISTICS
# ---------------------------------------------------------------------------
class AdvancedGestureDetector:
    """
    Combines MediaPipe Task ML gesture outputs with high-precision
    geometric heuristics on 21 3D landmarks for rich gesture classification.
    """

    def __init__(self, history_len: int = 5):
        # Rolling history per hand for temporal smoothing
        self.history = {}
        self.history_len = history_len

    @staticmethod
    def _dist(p1, p2):
        """Euclidean distance in 2D/3D."""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    @staticmethod
    def _pixel_dist(p1, p2, w, h):
        """Pixel distance between normalized points."""
        return math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h)

    def analyze_fingers(self, lms, w, h):
        """
        Determines extension state (True/False) for all 5 fingers.
        Uses distance from wrist and MCP for rotation invariance.
        """
        wrist = lms[WRIST]
        extended = []

        # 1. Thumb state
        # Distance from thumb tip to pinky MCP compared to thumb MCP to pinky MCP
        d_tip_pinky = self._dist(lms[THUMB_TIP], lms[PINKY_MCP])
        d_mcp_pinky = self._dist(lms[THUMB_MCP], lms[PINKY_MCP])
        d_thumb_extend = self._dist(lms[THUMB_TIP], lms[INDEX_MCP])
        thumb_is_open = (d_tip_pinky > d_mcp_pinky * 1.15) and (d_thumb_extend > 0.12)
        extended.append(thumb_is_open)

        # 2. Index finger
        d_index_tip = self._dist(lms[INDEX_TIP], wrist)
        d_index_pip = self._dist(lms[INDEX_PIP], wrist)
        index_open = d_index_tip > d_index_pip * 1.15
        extended.append(index_open)

        # 3. Middle finger
        d_mid_tip = self._dist(lms[MIDDLE_TIP], wrist)
        d_mid_pip = self._dist(lms[MIDDLE_PIP], wrist)
        mid_open = d_mid_tip > d_mid_pip * 1.15
        extended.append(mid_open)

        # 4. Ring finger
        d_ring_tip = self._dist(lms[RING_TIP], wrist)
        d_ring_pip = self._dist(lms[RING_PIP], wrist)
        ring_open = d_ring_tip > d_ring_pip * 1.15
        extended.append(ring_open)

        # 5. Pinky finger
        d_pinky_tip = self._dist(lms[PINKY_TIP], wrist)
        d_pinky_pip = self._dist(lms[PINKY_PIP], wrist)
        pinky_open = d_pinky_tip > d_pinky_pip * 1.15
        extended.append(pinky_open)

        return extended

    def detect_gesture(self, lms, ml_gesture_name: str, ml_score: float, w, h, hand_id: int):
        """
        Synthesizes ML model gesture prediction with geometric rules.
        """
        extended = self.analyze_fingers(lms, w, h)
        thumb, index, mid, ring, pinky = extended
        open_count = sum(extended)

        # Pinch metric: distance between thumb tip and index tip in pixels
        pinch_dist = self._pixel_dist(lms[THUMB_TIP], lms[INDEX_TIP], w, h)
        is_pinching = pinch_dist < 42

        # Wrist position relative to thumb tip for upright direction
        thumb_above_wrist = lms[THUMB_TIP].y < lms[WRIST].y - 0.08
        thumb_below_wrist = lms[THUMB_TIP].y > lms[WRIST].y + 0.08

        # Rule-based heuristics
        candidate = "UNKNOWN"
        confidence = ml_score

        # 1. OK SIGN: Thumb & Index tip touching, other three extended
        if is_pinching and mid and ring and pinky:
            candidate = "OK SIGN"
            confidence = 0.95

        # 2. PINCH: Thumb & Index close together, fingers generally folded
        elif is_pinching and not (mid and ring and pinky):
            candidate = "PINCH"
            confidence = 0.92

        # 3. PEACE / VICTORY: Index & Middle up, Ring & Pinky down
        elif index and mid and not ring and not pinky:
            candidate = "PEACE / VICTORY"
            confidence = 0.94

        # 4. ROCK ON / HORNS: Index & Pinky up, Middle & Ring folded
        elif index and pinky and not mid and not ring:
            candidate = "ROCK ON"
            confidence = 0.93

        # 5. POINTING: Index up, all others folded
        elif index and not mid and not ring and not pinky:
            candidate = "POINTING"
            confidence = 0.91

        # 6. I LOVE YOU: Thumb, Index, Pinky extended, Middle & Ring folded
        elif thumb and index and pinky and not mid and not ring:
            candidate = "I LOVE YOU"
            confidence = 0.94

        # 7. CALL ME / SHAKA: Thumb & Pinky extended, Middle 3 folded
        elif thumb and pinky and not index and not mid and not ring:
            candidate = "CALL ME"
            confidence = 0.92

        # 8. THUMBS UP: Thumb up, other four fingers folded
        elif thumb and not index and not mid and not ring and not pinky and thumb_above_wrist:
            candidate = "THUMBS UP"
            confidence = 0.92

        # 9. THUMBS DOWN: Thumb down, other four fingers folded
        elif thumb and not index and not mid and not ring and not pinky and thumb_below_wrist:
            candidate = "THUMBS DOWN"
            confidence = 0.92

        # 10. OPEN PALM: All 5 fingers extended
        elif open_count == 5:
            candidate = "OPEN PALM"
            confidence = 0.95

        # 11. CLOSED FIST: All fingers curled
        elif open_count == 0:
            candidate = "CLOSED FIST"
            confidence = 0.95

        # 12. THREE: Index, Middle, Ring extended
        elif not thumb and index and mid and ring and not pinky:
            candidate = "THREE"
            confidence = 0.88

        # 13. FOUR: 4 fingers extended, Thumb tucked
        elif not thumb and index and mid and ring and pinky:
            candidate = "FOUR"
            confidence = 0.90

        # If geometric rule wasn't conclusive, inspect ML Task output
        if candidate == "UNKNOWN" and ml_gesture_name and ml_gesture_name != "None":
            ml_clean = ml_gesture_name.replace("_", " ").upper()
            if "THUMB_UP" in ml_gesture_name or "THUMB UP" in ml_clean:
                candidate = "THUMBS UP"
            elif "THUMB_DOWN" in ml_gesture_name or "THUMB DOWN" in ml_clean:
                candidate = "THUMBS DOWN"
            elif "VICTORY" in ml_clean:
                candidate = "PEACE / VICTORY"
            elif "POINTING" in ml_clean:
                candidate = "POINTING"
            elif "OPEN_PALM" in ml_gesture_name or "OPEN PALM" in ml_clean:
                candidate = "OPEN PALM"
            elif "CLOSED_FIST" in ml_gesture_name or "CLOSED FIST" in ml_clean:
                candidate = "CLOSED FIST"
            elif "ILOVEYOU" in ml_gesture_name:
                candidate = "I LOVE YOU"
            else:
                candidate = ml_clean
            confidence = max(ml_score, 0.75)

        # Fallback to finger count if still unknown
        if candidate == "UNKNOWN":
            candidate = f"COUNT: {open_count}"
            confidence = 0.60

        # Temporal smoothing to prevent flicker
        if hand_id not in self.history:
            self.history[hand_id] = deque(maxlen=self.history_len)
        self.history[hand_id].append((candidate, confidence))

        # Majority vote across history
        recent_gestures = [item[0] for item in self.history[hand_id]]
        final_gesture = max(set(recent_gestures), key=recent_gestures.count)
        avg_conf = np.mean([item[1] for item in self.history[hand_id] if item[0] == final_gesture])

        return final_gesture, float(avg_conf), extended, pinch_dist


# ---------------------------------------------------------------------------
# FANCY HUD & SKELETON VISUALIZER
# ---------------------------------------------------------------------------
class FancyHandVisualizer:
    """
    Renders high-tech, Cyberpunk / Sci-Fi HUD overlays:
    - Multi-layer glowing hand skeleton bones
    - Translucent palm cyber webbing
    - Pulsating fingertip reticles & pinch indicator
    - Sci-fi corner bracket target bounding boxes
    - Handedness & gesture badges with confidence meters
    - Top HUD status bar & bottom control hints
    - Screenshot toast notification
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
    def draw_glass_rect(img, x1, y1, x2, y2, bg_color=(15, 18, 24), border_color=(255, 242, 0), alpha=0.75):
        """Draws a semi-transparent glass panel with a tech border."""
        h, w = img.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
        if x2 <= x1 or y2 <= y1:
            return

        sub_img = img[y1:y2, x1:x2]
        colored_rect = np.full(sub_img.shape, bg_color, dtype=np.uint8)
        blended = cv2.addWeighted(colored_rect, alpha, sub_img, 1.0 - alpha, 0)
        img[y1:y2, x1:x2] = blended

        if border_color:
            cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)

    @staticmethod
    def draw_corner_brackets(img, x1, y1, x2, y2, color, length=18, thickness=2):
        """Draws futuristic target reticle corner brackets around a bounding box."""
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

    def draw_skeleton(self, img, lms, color_palette, w, h):
        """
        Renders full hand skeleton with double-layered neon glow and nodes.
        """
        primary = color_palette["primary"]
        secondary = color_palette["secondary"]

        coords = []
        for lm in lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            coords.append((cx, cy))

        # 1. Subtle Translucent Palm Mesh Polygon
        palm_pts = np.array([coords[idx] for idx in PALM_POLYGON], np.int32)
        palm_pts = palm_pts.reshape((-1, 1, 2))
        overlay = img.copy()
        cv2.fillPoly(overlay, [palm_pts], primary)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
        cv2.polylines(img, [palm_pts], True, secondary, 1, cv2.LINE_AA)

        # 2. Bones (Outer Glow + Inner Crisp Core Line)
        glow_layer = img.copy()
        for start_idx, end_idx in HAND_BONES:
            pt1 = coords[start_idx]
            pt2 = coords[end_idx]
            # Outer glow
            cv2.line(glow_layer, pt1, pt2, secondary, 6, cv2.LINE_AA)
        cv2.addWeighted(glow_layer, 0.45, img, 0.55, 0, img)

        for start_idx, end_idx in HAND_BONES:
            pt1 = coords[start_idx]
            pt2 = coords[end_idx]
            # Inner core
            cv2.line(img, pt1, pt2, primary, 2, cv2.LINE_AA)

        # 3. Landmark Nodes & Joints
        for idx, (cx, cy) in enumerate(coords):
            is_tip = idx in FINGER_TIPS
            is_wrist = idx == WRIST

            if is_tip:
                # Fingertip Target Reticle
                cv2.circle(img, (cx, cy), 8, secondary, 1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 5, primary, -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)
            elif is_wrist:
                # Wrist Anchor
                cv2.circle(img, (cx, cy), 7, secondary, 2, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 4, primary, -1, cv2.LINE_AA)
            else:
                # Joint Dot
                cv2.circle(img, (cx, cy), 4, primary, -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)

        return coords

    def draw_pinch_line(self, img, coords, pinch_dist, primary_color):
        """Highlights pinch interaction with laser line and real-time distance readout."""
        p_thumb = coords[THUMB_TIP]
        p_index = coords[INDEX_TIP]

        if pinch_dist < 42:
            # Active pinch laser
            cv2.line(img, p_thumb, p_index, (0, 255, 255), 2, cv2.LINE_AA)
            mid_x = (p_thumb[0] + p_index[0]) // 2
            mid_y = (p_thumb[1] + p_index[1]) // 2 - 12
            cv2.putText(
                img,
                f"PINCH {int(pinch_dist)}px",
                (mid_x - 36, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def draw_hand_hud_card(
        self,
        img,
        coords,
        hand_idx: int,
        handedness: str,
        hand_conf: float,
        gesture: str,
        gesture_conf: float,
        extended_fingers: list,
        color_palette: dict,
        show_box: bool,
        show_fingers: bool,
    ):
        """
        Renders bounding reticle, gesture badge, handedness tag,
        and finger extension bars.
        """
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        pad = 28
        x1, y1 = max(10, min(xs) - pad), max(10, min(ys) - pad)
        x2, y2 = min(img.shape[1] - 10, max(xs) + pad), min(img.shape[0] - 10, max(ys) + pad)

        primary = color_palette["primary"]
        secondary = color_palette["secondary"]

        # 1. Corner Bracket Bounding Box
        if show_box:
            self.draw_glass_rect(img, x1, y1, x2, y2, bg_color=(10, 15, 22), border_color=None, alpha=0.15)
            self.draw_corner_brackets(img, x1, y1, x2, y2, primary, length=24, thickness=2)

        # 2. Floating Header Badge (Hand ID & Handedness)
        header_text = f"HAND #{hand_idx + 1} // {handedness.upper()} ({int(hand_conf * 100)}%)"
        (tw, th), _ = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        bx1 = x1
        by1 = max(10, y1 - 42)
        bx2 = bx1 + max(tw + 20, 160)
        by2 = by1 + 20
        self.draw_glass_rect(img, bx1, by1, bx2, by2, bg_color=(12, 16, 24), border_color=primary, alpha=0.85)
        cv2.putText(
            img,
            header_text,
            (bx1 + 8, by2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            primary,
            1,
            cv2.LINE_AA,
        )

        # 3. Gesture Display Pill
        gesture_text = f"{gesture}  [{int(gesture_conf * 100)}%]"
        (gw, gh), _ = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        gy1 = by2 + 2
        gy2 = gy1 + 22
        gx2 = bx1 + max(gw + 20, 160)
        self.draw_glass_rect(img, bx1, gy1, gx2, gy2, bg_color=(20, 24, 34), border_color=secondary, alpha=0.88)
        cv2.putText(
            img,
            gesture_text,
            (bx1 + 8, gy2 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # 4. Finger State Status Indicators [T] [I] [M] [R] [P]
        if show_fingers and len(extended_fingers) == 5:
            labels = ["T", "I", "M", "R", "P"]
            fy1 = y2 + 6
            fy2 = fy1 + 20
            block_w = 26
            total_w = block_w * 5 + 16
            fx1 = x1
            fx2 = fx1 + total_w

            self.draw_glass_rect(img, fx1, fy1, fx2, fy2, bg_color=(10, 14, 20), border_color=(80, 90, 110), alpha=0.80)

            for i, (lbl, is_ext) in enumerate(zip(labels, extended_fingers)):
                bx = fx1 + 4 + i * block_w
                by = fy1 + 3
                fill_color = primary if is_ext else (35, 40, 50)
                txt_color = (10, 10, 10) if is_ext else (140, 140, 150)
                cv2.rectangle(img, (bx, by), (bx + block_w - 4, by + 14), fill_color, -1)
                cv2.putText(
                    img,
                    lbl,
                    (bx + 6, by + 11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    txt_color,
                    1,
                    cv2.LINE_AA,
                )

    def draw_top_hud(self, img, fps: float, hand_count: int, max_hands: int, frame_w: int, frame_h: int):
        """Renders futuristic header banner with FPS, detection counts, and telemetry."""
        bar_h = 44
        theme = self.current_theme
        border_col = theme["hud_border"]
        bg_col = theme["bg_glass"]

        # Glass panel header
        self.draw_glass_rect(img, 12, 10, frame_w - 12, 10 + bar_h, bg_color=bg_col, border_color=border_col, alpha=0.82)

        # Title & Branding
        cv2.putText(
            img,
            "CYBERNETIC HAND MATRIX",
            (26, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            border_col,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            "// AI MULTI-TRACKER",
            (260, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 200, 220),
            1,
            cv2.LINE_AA,
        )

        # Telemetry: FPS meter with color gradient
        fps_color = (0, 255, 100) if fps >= 28 else ((0, 200, 255) if fps >= 18 else (0, 70, 255))
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            img,
            fps_text,
            (frame_w - 380, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            fps_color,
            2,
            cv2.LINE_AA,
        )

        # Active Hands count
        hands_text = f"HANDS: {hand_count}/{max_hands}"
        cv2.putText(
            img,
            hands_text,
            (frame_w - 260, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Theme indicator
        theme_tag = f"THEME: {theme['name'][:10]}"
        cv2.putText(
            img,
            theme_tag,
            (frame_w - 130, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 180, 200),
            1,
            cv2.LINE_AA,
        )

    def draw_bottom_hud(self, img, frame_w: int, frame_h: int):
        """Renders futuristic bottom keyboard shortcut legend."""
        bar_h = 30
        y1 = frame_h - bar_h - 10
        y2 = frame_h - 10
        theme = self.current_theme

        self.draw_glass_rect(img, 12, y1, frame_w - 12, y2, bg_color=theme["bg_glass"], border_color=(60, 70, 90), alpha=0.75)

        help_text = (
            "[Q] Quit  |  [S] Skeleton  |  [B] Box  |  [F] Fingers  |  "
            "[C] Color Theme  |  [SPACE] Snapshot"
        )
        cv2.putText(
            img,
            help_text,
            (24, y2 - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (190, 210, 230),
            1,
            cv2.LINE_AA,
        )

    def draw_toast(self, img, frame_w: int, frame_h: int):
        """Displays temporary notification message with animated banner."""
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
# MAIN APPLICATION CONTROLLER
# ---------------------------------------------------------------------------
class HandTrackerApp:
    """
    Main real-time Hand Tracking & Gesture Recognition pipeline.
    """

    def __init__(self, max_hands: int = 4, cam_index: int = 0):
        self.max_hands = max_hands
        self.cam_index = cam_index

        # Feature display toggles
        self.show_skeleton = True
        self.show_box = True
        self.show_fingers = True
        self.show_hud = True

        # Pipeline components
        self.gesture_detector = AdvancedGestureDetector()
        self.visualizer = FancyHandVisualizer()
        self.recognizer = None

        # Telemetry
        self.fps_buffer = deque(maxlen=20)
        self.last_time = time.perf_counter()

        # Capture output folder
        self.capture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "captures")
        os.makedirs(self.capture_dir, exist_ok=True)

    def initialize_detector(self):
        """Initializes the MediaPipe GestureRecognizer task model."""
        if not ensure_model_exists(MODEL_PATH):
            raise RuntimeError(f"Could not load or download {MODEL_PATH}")

        print(" [INIT] Initializing MediaPipe Vision Gesture Recognizer...")
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.max_hands,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        print(f" [INIT] Ready! Configured to track up to {self.max_hands} hands simultaneously.")

    def run(self):
        """Main camera acquisition and visualization loop."""
        self.initialize_detector()

        print(f" [CAMERA] Opening camera device at index {self.cam_index}...")
        cap = cv2.VideoCapture(self.cam_index)

        # Attempt to set 1280x720 resolution for crisp UI
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print(f" [WARNING] Could not open camera {self.cam_index}. Trying camera index 1...")
            cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            print(" [ERROR] No webcam could be opened! Please ensure your camera is connected and not in use by another app.")
            return

        window_name = "CYBERNETIC HAND MATRIX // AI VISION"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("=" * 65)
        print(" SYSTEM ONLINE - Real-time Hand & Gesture Recognition Running")
        print(" Press [Q] or [ESC] to quit.")
        print(" Press [C] to cycle neon color palettes.")
        print(" Press [SPACE] to take a high-res screenshot.")
        print("=" * 65)

        try:
            while True:
                success, frame = cap.read()
                if not success:
                    print(" [WARNING] Blank frame received. Retrying...")
                    time.sleep(0.02)
                    continue

                # Mirror frame horizontally for natural webcam user experience
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                # Convert BGR to RGB for MediaPipe Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Inference
                result = self.recognizer.recognize(mp_image)

                # Calculate smoothed FPS
                now = time.perf_counter()
                dt = now - self.last_time
                self.last_time = now
                if dt > 0:
                    self.fps_buffer.append(1.0 / dt)
                fps = float(np.mean(self.fps_buffer)) if self.fps_buffer else 30.0

                hand_count = len(result.hand_landmarks) if result and result.hand_landmarks else 0
                theme = self.visualizer.current_theme
                hands_palette = theme["hands"]

                # Process each detected hand
                if hand_count > 0:
                    for i in range(hand_count):
                        lms = result.hand_landmarks[i]
                        color_palette = hands_palette[i % len(hands_palette)]

                        # 1. Handedness
                        handedness_label = "Hand"
                        hand_score = 0.90
                        if result.handedness and i < len(result.handedness) and result.handedness[i]:
                            category = result.handedness[i][0]
                            handedness_label = category.category_name
                            hand_score = category.score

                        # 2. ML Gesture
                        ml_gesture_name = ""
                        ml_gesture_score = 0.0
                        if result.gestures and i < len(result.gestures) and result.gestures[i]:
                            top_gesture = result.gestures[i][0]
                            ml_gesture_name = top_gesture.category_name
                            ml_gesture_score = top_gesture.score

                        # 3. Analyze Gestures & Fingers
                        gesture_name, gesture_conf, ext_fingers, pinch_dist = (
                            self.gesture_detector.detect_gesture(
                                lms, ml_gesture_name, ml_gesture_score, w, h, hand_id=i
                            )
                        )

                        # 4. Render Skeleton
                        coords = []
                        if self.show_skeleton:
                            coords = self.visualizer.draw_skeleton(frame, lms, color_palette, w, h)
                            self.visualizer.draw_pinch_line(frame, coords, pinch_dist, color_palette["primary"])
                        else:
                            coords = [(int(lm.x * w), int(lm.y * h)) for lm in lms]

                        # 5. Render Hand HUD Card
                        if self.show_hud:
                            self.visualizer.draw_hand_hud_card(
                                frame,
                                coords,
                                hand_idx=i,
                                handedness=handedness_label,
                                hand_conf=hand_score,
                                gesture=gesture_name,
                                gesture_conf=gesture_conf,
                                extended_fingers=ext_fingers,
                                color_palette=color_palette,
                                show_box=self.show_box,
                                show_fingers=self.show_fingers,
                            )

                # Render Global HUD Overlays
                if self.show_hud:
                    self.visualizer.draw_top_hud(frame, fps, hand_count, self.max_hands, w, h)
                    self.visualizer.draw_bottom_hud(frame, w, h)
                    self.visualizer.draw_toast(frame, w, h)

                # Show frame
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
                elif key in (ord("b"), ord("B")):
                    self.show_box = not self.show_box
                    status = "ENABLED" if self.show_box else "DISABLED"
                    self.visualizer.show_toast(f"TARGET BOX: {status}")
                elif key in (ord("f"), ord("F")):
                    self.show_fingers = not self.show_fingers
                    status = "ENABLED" if self.show_fingers else "DISABLED"
                    self.visualizer.show_toast(f"FINGER BARS: {status}")
                elif key in (ord("h"), ord("H")):
                    self.show_hud = not self.show_hud
                elif key in (ord("c"), ord("C")):
                    new_theme_name = self.visualizer.cycle_theme()
                    self.visualizer.show_toast(f"THEME: {new_theme_name}")
                elif key == 32:  # SPACE bar
                    filename = f"hand_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    filepath = os.path.join(self.capture_dir, filename)
                    cv2.imwrite(filepath, frame)
                    self.visualizer.show_toast(f"SNAPSHOT SAVED: {filename}")
                    print(f" [SNAPSHOT] Saved high-res capture to: {filepath}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.recognizer:
                self.recognizer.close()
            print(" [EXIT] Hand Tracker clean shutdown complete.")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = HandTrackerApp(max_hands=4, cam_index=0)
    app.run()
