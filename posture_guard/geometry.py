"""Small pure-geometry helpers shared by feature extraction and the synthetic model."""

from __future__ import annotations

import math

import numpy as np

EPS = 1e-9


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, float) + np.asarray(b, float)) / 2.0


def angle_from_horizontal_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Signed angle of segment a->b relative to the horizontal axis, in degrees."""
    d = np.asarray(b, float) - np.asarray(a, float)
    return math.degrees(math.atan2(d[1], d[0]))


def angle_from_vertical_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Absolute angle of segment a->b away from vertical, in degrees."""
    d = np.asarray(b, float) - np.asarray(a, float)
    return abs(math.degrees(math.atan2(d[0], d[1] + EPS)))


def rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def median_abs_deviation(values: np.ndarray) -> float:
    """MAD scaled to be comparable with a standard deviation for normal data."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))
