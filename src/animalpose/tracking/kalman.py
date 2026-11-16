"""Constant-velocity Kalman filter on box centre and size (SORT/DeepSORT style)."""

from __future__ import annotations

import numpy as np


class BoxKalman:
    """State ``[cx, cy, w, h, vcx, vcy, vw, vh]``; noise scales with box size."""

    _F = np.eye(8)
    _F[:4, 4:] = np.eye(4)
    _H = np.eye(4, 8)

    def __init__(self, box: np.ndarray, pos_std: float = 1 / 20, vel_std: float = 1 / 160):
        self.pos_std, self.vel_std = pos_std, vel_std
        z = self._to_z(box)
        self.x = np.concatenate([z, np.zeros(4)])
        s = max(z[2], z[3])
        self.P = np.diag(np.square([2 * pos_std * s] * 4 + [10 * vel_std * s] * 4))

    @staticmethod
    def _to_z(box) -> np.ndarray:
        x0, y0, x1, y1 = box
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, 1.0), max(y1 - y0, 1.0)])

    @property
    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        w, h = max(w, 1.0), max(h, 1.0)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    @property
    def velocity(self) -> np.ndarray:
        return self.x[4:6]

    def predict(self) -> np.ndarray:
        s = max(self.x[2], self.x[3])
        Q = np.diag(np.square([self.pos_std * s] * 4 + [self.vel_std * s] * 4))
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + Q
        return self.box

    def innovation(self, box) -> tuple[np.ndarray, np.ndarray]:
        s = max(self.x[2], self.x[3])
        R = np.diag(np.square([self.pos_std * s] * 4))
        S = self._H @ self.P @ self._H.T + R
        return self._to_z(box) - self._H @ self.x, S

    def mahalanobis(self, box) -> float:
        y, S = self.innovation(box)
        return float(y[:2] @ np.linalg.solve(S[:2, :2], y[:2]))

    def update(self, box) -> None:
        y, S = self.innovation(box)
        Kg = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + Kg @ y
        self.P = (np.eye(8) - Kg @ self._H) @ self.P
