"""Pinhole camera with Brown-Conrady distortion (OpenCV convention)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Axis-angle (3,) -> rotation matrix (3, 3)."""
    rvec = np.asarray(rvec, dtype=np.float64)
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * kx @ kx


def rotation_to_rvec(r: np.ndarray) -> np.ndarray:
    """Rotation matrix (3, 3) -> axis-angle (3,)."""
    cos = np.clip((np.trace(r) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos)
    if theta < 1e-12:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        # Near 180 degrees: recover the axis from the symmetric part.
        b = (r + np.eye(3)) / 2
        axis = np.sqrt(np.clip(np.diag(b), 0, None))
        i = int(np.argmax(axis))
        axis = b[:, i] / np.sqrt(b[i, i])
        return axis * theta
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    return w * theta / (2 * np.sin(theta))


@dataclass
class Camera:
    """World -> camera: ``x_cam = R @ X + t``; image: ``K @ distort(x_cam / z)``."""

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    size: tuple[int, int] = (640, 480)  # (width, height)
    name: str = "cam"

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64)
        self.R = np.asarray(self.R, dtype=np.float64)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)
        d = np.zeros(5)
        dist = np.asarray(self.dist, dtype=np.float64).ravel()
        d[: len(dist)] = dist
        self.dist = d

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.hstack([self.R, self.t[:, None]])

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def to_camera(self, X: np.ndarray) -> np.ndarray:
        return X @ self.R.T + self.t

    def _distort(self, xn: np.ndarray) -> np.ndarray:
        k1, k2, p1, p2, k3 = self.dist
        x, y = xn[..., 0], xn[..., 1]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.stack([xd, yd], axis=-1)

    def project(self, X: np.ndarray, return_depth: bool = False):
        """Project world points ``(..., 3)`` to pixels ``(..., 2)``."""
        Xc = self.to_camera(np.asarray(X, dtype=np.float64))
        z = Xc[..., 2]
        xn = Xc[..., :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[..., None]
        xd = self._distort(xn)
        uv = xd @ self.K[:2, :2].T + self.K[:2, 2]
        return (uv, z) if return_depth else uv

    def undistort_points(self, uv: np.ndarray, iters: int = 10) -> np.ndarray:
        """Pixels ``(..., 2)`` -> normalized, undistorted image coordinates."""
        uv = np.asarray(uv, dtype=np.float64)
        xd = (uv - self.K[:2, 2]) @ np.linalg.inv(self.K[:2, :2]).T
        if not np.any(self.dist):
            return xd
        xn = xd.copy()
        for _ in range(iters):  # fixed-point inversion, as in cv2.undistortPoints
            k1, k2, p1, p2, k3 = self.dist
            x, y = xn[..., 0], xn[..., 1]
            r2 = x * x + y * y
            radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
            dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            xn = np.stack([(xd[..., 0] - dx) / radial, (xd[..., 1] - dy) / radial], axis=-1)
        return xn

    def in_image(self, uv: np.ndarray, margin: float = 0.0) -> np.ndarray:
        w, h = self.size
        return (uv[..., 0] >= -margin) & (uv[..., 0] < w + margin) & (uv[..., 1] >= -margin) & (uv[..., 1] < h + margin)

    @classmethod
    def look_at(cls, eye, target, up=(0, 0, 1), focal: float = 500.0, size=(640, 480), dist=None, name="cam"):
        """Camera at ``eye`` looking at ``target`` (world z-up, image y-down)."""
        eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
        fwd = target - eye
        fwd /= np.linalg.norm(fwd)
        if abs(np.dot(fwd, up / np.linalg.norm(up))) > 0.999:
            up = np.array([0.0, 1.0, 0.0])  # looking straight down: pick image "up" = +y
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd])
        t = -R @ eye
        w, h = size
        K = np.array([[focal, 0, (w - 1) / 2], [0, focal, (h - 1) / 2], [0, 0, 1]])
        return cls(K, R, t, np.zeros(5) if dist is None else dist, tuple(size), name)

    def to_dict(self) -> dict:
        return {"name": self.name, "K": self.K.tolist(), "R": self.R.tolist(), "t": self.t.tolist(),
                "dist": self.dist.tolist(), "size": list(self.size)}

    @classmethod
    def from_dict(cls, d: dict) -> Camera:
        return cls(np.array(d["K"]), np.array(d["R"]), np.array(d["t"]), np.array(d.get("dist", [0] * 5)),
                   tuple(d.get("size", (640, 480))), d.get("name", "cam"))


def essential_matrix(cam_a: Camera, cam_b: Camera) -> np.ndarray:
    """E such that ``x_b^T E x_a = 0`` for undistorted normalized coordinates."""
    R = cam_b.R @ cam_a.R.T
    t = cam_b.t - R @ cam_a.t
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return tx @ R
