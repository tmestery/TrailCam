"""Motion-based ROI so the CNN sees the animal, not the whole scene."""

from __future__ import annotations

import cv2
import numpy as np


def _clamp_box(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int
) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def _square_pad_box(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int, pad_frac: float
) -> tuple[int, int, int, int]:
    """Expand bbox with padding and make it square (matches crop-style training)."""
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(bw, bh) * (1.0 + 2.0 * pad_frac)
    side = max(side, 0.10 * min(w, h))
    side = min(side, 0.55 * min(w, h))
    half = side / 2.0
    nx1 = int(round(cx - half))
    ny1 = int(round(cy - half))
    nx2 = int(round(cx + half))
    ny2 = int(round(cy + half))
    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if nx2 > w:
        nx1 -= nx2 - w
        nx2 = w
    if ny2 > h:
        ny1 -= ny2 - h
        ny2 = h
    return _clamp_box(nx1, ny1, nx2, ny2, w, h)


class MotionCropper:
    """Track a compact moving region and return a square animal-style crop.

    Full-frame flicker (scene cuts, exposure pops) is rejected so the CNN
    does not get a giant yard crop again.
    """

    def __init__(
        self,
        min_area_frac: float = 0.002,
        max_area_frac: float = 0.20,
        pad_frac: float = 0.40,
        hold_frames: int = 25,
        warmup_frames: int = 12,
    ) -> None:
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.pad_frac = pad_frac
        self.hold_frames = hold_frames
        self.warmup_frames = warmup_frames
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=40,
            detectShadows=False,
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._last_box: tuple[int, int, int, int] | None = None
        self._hold_left = 0
        self._frame_i = 0
        self._prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=40,
            detectShadows=False,
        )
        self._last_box = None
        self._hold_left = 0
        self._frame_i = 0
        self._prev_gray = None

    def _motion_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        fg = self._subtractor.apply(frame_bgr, learningRate=0.02)
        _, mog = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        if self._prev_gray is None:
            diff = np.zeros_like(gray)
        else:
            diff = cv2.absdiff(gray, self._prev_gray)
            _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        self._prev_gray = gray

        mask = cv2.bitwise_and(mog, diff)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=2)
        return mask

    def find_box(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """Return a square ROI around compact motion, or the held last box."""
        self._frame_i += 1
        h, w = frame_bgr.shape[:2]
        mask = self._motion_mask(frame_bgr)

        if self._frame_i <= self.warmup_frames:
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.min_area_frac * w * h
        max_area = self.max_area_frac * w * h

        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Reject strips that span most of the frame (global flicker).
            if bw > 0.75 * w or bh > 0.75 * h:
                continue
            box = _square_pad_box(x, y, x + bw, y + bh, w, h, self.pad_frac)
            bx1, by1, bx2, by2 = box
            boxed_area = (bx2 - bx1) * (by2 - by1)
            if boxed_area > 0.45 * w * h:
                continue
            # Prefer mid-sized, somewhat central animals.
            cx = 0.5 * (bx1 + bx2) / w
            cy = 0.5 * (by1 + by2) / h
            center_score = 1.0 - min(1.0, ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5)
            score = area * (0.4 + 0.6 * center_score)
            candidates.append((score, box))

        if candidates:
            candidates.sort(key=lambda t: t[0], reverse=True)
            box = candidates[0][1]
            self._last_box = box
            self._hold_left = self.hold_frames
            return box

        if self._last_box is not None and self._hold_left > 0:
            self._hold_left -= 1
            return self._last_box

        return None


def crop_bgr(frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return frame_bgr[y1:y2, x1:x2].copy()
