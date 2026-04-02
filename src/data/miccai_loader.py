"""
MICCAI EndoVis Robotic Instrument Tracking Data Loader
=======================================================

This module loads per-frame instrument-tip trajectories from the MICCAI
EndoVis dataset.  The dataset contains stereo surgical video sequences stored
as ``video.mp4`` files (1280×1024, vertically stacked stereo pair) alongside
``info.yaml`` and ``calibration.yaml`` metadata files.

Because the dataset ships raw video without pre-computed annotations, this
loader extracts instrument trajectories automatically using OpenCV background
subtraction (MOG2).  Extracted trajectories are cached as ``.npy`` files
so subsequent loads are instant.

Dataset layout expected
-----------------------
::

    <train_root>/
        case_1/
            1/
                video.mp4
                info.yaml
                calibration.yaml
            2/ ...
        case_2/ ...

Stereo convention
-----------------
The ``video_stack: "vertical"`` field in ``info.yaml`` means the left-camera
frame occupies the **top** half of each video frame and the right-camera frame
occupies the **bottom** half.  Each single-view resolution is **1280 × 512**.

Usage
-----
    >>> from src.data import MICCAILoader
    >>> loader = MICCAILoader()
    >>> # Load from a sequence directory (auto-detects video.mp4)
    >>> trajectory = loader.load_trajectory("C:/Users/dprad/Downloads/train/train/case_1/1")
    >>> print(trajectory.shape)   # (N, 2)  — (x, y) pixel coords
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class MICCAILoader:
    """Extract per-frame instrument-tip trajectories from MICCAI EndoVis videos.

    Parameters
    ----------
    stereo_half : str
        Which stereo half to use: ``"top"`` (left camera, default) or
        ``"bottom"`` (right camera).
    min_contour_area : int
        Minimum contour area (pixels²) to be considered a valid instrument
        detection.  Smaller blobs are discarded as noise.
    use_cache : bool
        If ``True`` (default), save/load the extracted trajectory as a ``.npy``
        file next to the video to avoid re-processing on subsequent runs.
    smooth_sigma : float
        Standard deviation for Gaussian smoothing of the raw centroid trajectory.
        Set to ``0`` to disable.  Default is ``3.0``.
    """

    # Single-view dimensions for the vertically-stacked stereo video
    STEREO_FULL_H = 1024
    STEREO_HALF_H = 512
    STEREO_W = 1280

    def __init__(
        self,
        stereo_half: str = "top",
        min_contour_area: int = 150,
        use_cache: bool = True,
        smooth_sigma: float = 3.0,
    ) -> None:
        if stereo_half not in ("top", "bottom"):
            raise ValueError(f"stereo_half must be 'top' or 'bottom', got '{stereo_half}'")
        self.stereo_half = stereo_half
        self.min_contour_area = min_contour_area
        self.use_cache = use_cache
        self.smooth_sigma = smooth_sigma

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load_trajectory(self, sequence_dir: str | os.PathLike) -> np.ndarray:
        """Load the instrument-tip trajectory for a sequence directory.

        Looks for ``video.mp4`` inside *sequence_dir* and extracts the
        centroid trajectory via background subtraction.  Results are cached
        as ``trajectory_<half>.npy`` alongside the video.

        Parameters
        ----------
        sequence_dir : str or path-like
            Path to a sequence folder, e.g. ``".../case_1/1"``.

        Returns
        -------
        np.ndarray, shape (N, 2)
            ``(x, y)`` pixel coordinates of the detected instrument tip for
            each of the *N* video frames, in the single-view coordinate system
            (origin at top-left of the selected stereo half).

        Raises
        ------
        FileNotFoundError
            If no ``video.mp4`` is found in *sequence_dir*.
        RuntimeError
            If the video cannot be opened by OpenCV.
        """
        seq_dir = Path(sequence_dir)
        if not seq_dir.is_dir():
            raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

        video_path = seq_dir / "video.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(
                f"No video.mp4 found in {seq_dir}.  "
                "Expected MICCAI EndoVis format with video.mp4 per sequence."
            )

        # Check cache
        cache_path = seq_dir / f"trajectory_{self.stereo_half}.npy"
        if self.use_cache and cache_path.is_file():
            logger.info("Loading cached trajectory from: %s", cache_path)
            traj = np.load(str(cache_path))
            logger.info("Cached trajectory: %d frames", len(traj))
            return traj

        logger.info("Extracting trajectory from video: %s", video_path)
        traj = self._extract_trajectory_from_video(video_path)

        if self.use_cache:
            np.save(str(cache_path), traj)
            logger.info("Cached trajectory saved to: %s", cache_path)

        return traj

    def load_video_info(self, sequence_dir: str | os.PathLike) -> dict:
        """Read ``info.yaml`` from a sequence directory.

        Returns
        -------
        dict
            Parsed YAML contents.  Includes ``resolution.width``,
            ``resolution.height``, ``video_stack``, etc.
        """
        info_path = Path(sequence_dir) / "info.yaml"
        if not info_path.is_file():
            return {}
        with open(info_path, "r") as fh:
            return yaml.safe_load(fh) or {}

    def get_frame_size(self, sequence_dir: str | os.PathLike) -> Tuple[int, int]:
        """Return the ``(width, height)`` of a single stereo view.

        For vertically-stacked stereo video the height is half the full
        video height.  Falls back to the hardcoded defaults if ``info.yaml``
        is absent.

        Returns
        -------
        (width, height) : tuple of int
        """
        info = self.load_video_info(sequence_dir)
        res = info.get("resolution", {})
        w = int(res.get("width", self.STEREO_W))
        h = int(res.get("height", self.STEREO_FULL_H))
        stack = info.get("video_stack", "vertical")
        if stack == "vertical":
            h = h // 2
        return w, h

    # ------------------------------------------------------------------ #
    # Video processing                                                     #
    # ------------------------------------------------------------------ #

    def _extract_trajectory_from_video(self, video_path: Path) -> np.ndarray:
        """Run MOG2 background subtraction to extract instrument centroids.

        Algorithm
        ---------
        1. Warm up MOG2 on the first ~30 frames (not recorded).
        2. For each remaining frame:
           a. Crop the requested stereo half (top/bottom 512 rows).
           b. Convert to greyscale and apply the foreground mask.
           c. Morphologically clean the mask.
           d. Find the largest contour; record its centroid.
        3. Forward-fill any frames where no contour was detected.
        4. Optionally smooth with a Gaussian kernel.

        Returns
        -------
        np.ndarray, shape (N, 2)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Video: %d frames total", total_frames)

        # MOG2 background subtractor — history tuned for surgical video
        fgbg = cv2.createBackgroundSubtractorMOG2(
            history=50,
            varThreshold=25,
            detectShadows=False,
        )

        # Morphological kernel for noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Warm-up: feed ~30 frames to the background model without recording
        WARMUP = min(30, total_frames // 4)
        for _ in range(WARMUP):
            ret, frame = cap.read()
            if not ret:
                break
            half = self._crop_stereo_half(frame)
            grey = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
            fgbg.apply(grey)

        # Main extraction pass
        coords: list[Optional[Tuple[float, float]]] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            half = self._crop_stereo_half(frame)
            grey = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)

            # Apply background subtraction
            fg_mask = fgbg.apply(grey)

            # Morphological clean-up
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

            # Find contours and pick the largest
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            centroid: Optional[Tuple[float, float]] = None
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) >= self.min_contour_area:
                    M = cv2.moments(largest)
                    if M["m00"] > 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        centroid = (cx, cy)

            coords.append(centroid)

        cap.release()
        logger.info("Extracted %d frames (%d with valid detections)",
                    len(coords), sum(1 for c in coords if c is not None))

        if not coords:
            raise RuntimeError(f"No frames could be read from {video_path}")

        # Forward-fill missing detections
        traj = self._fill_missing(coords)

        # Optional Gaussian smoothing
        if self.smooth_sigma > 0:
            traj = self._smooth_trajectory(traj, self.smooth_sigma)

        return traj

    def _crop_stereo_half(self, frame: np.ndarray) -> np.ndarray:
        """Return the top or bottom half of a stereo video frame."""
        h = frame.shape[0]
        half_h = h // 2
        if self.stereo_half == "top":
            return frame[:half_h, :]
        else:
            return frame[half_h:, :]

    # ------------------------------------------------------------------ #
    # Trajectory utilities                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fill_missing(
        coords: list[Optional[Tuple[float, float]]]
    ) -> np.ndarray:
        """Forward-fill (then backward-fill) frames with no detection."""
        n = len(coords)
        traj = np.full((n, 2), np.nan, dtype=np.float64)
        for i, c in enumerate(coords):
            if c is not None:
                traj[i] = c

        # Forward fill
        last = None
        for i in range(n):
            if not np.isnan(traj[i, 0]):
                last = traj[i].copy()
            elif last is not None:
                traj[i] = last

        # Backward fill (handles leading NaNs)
        last = None
        for i in range(n - 1, -1, -1):
            if not np.isnan(traj[i, 0]):
                last = traj[i].copy()
            elif last is not None:
                traj[i] = last

        # If still NaN (entire sequence empty), fall back to frame centre
        if np.any(np.isnan(traj)):
            logger.warning("Could not detect instrument in any frame; using frame centre.")
            traj[np.isnan(traj[:, 0]), :] = [MICCAILoader.STEREO_W / 2,
                                              MICCAILoader.STEREO_HALF_H / 2]

        return traj

    @staticmethod
    def _smooth_trajectory(traj: np.ndarray, sigma: float) -> np.ndarray:
        """Apply per-axis Gaussian smoothing to reduce jitter."""
        from scipy.ndimage import gaussian_filter1d
        smoothed = np.stack(
            [gaussian_filter1d(traj[:, 0], sigma),
             gaussian_filter1d(traj[:, 1], sigma)],
            axis=1,
        )
        return smoothed.astype(np.float64)

    @staticmethod
    def compute_centroid_from_bbox(
        x1: float, y1: float, x2: float, y2: float
    ) -> Tuple[float, float]:
        """Compute the centroid of an axis-aligned bounding box.

        Examples
        --------
        >>> MICCAILoader.compute_centroid_from_bbox(100, 200, 200, 300)
        (150.0, 250.0)
        """
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


# --------------------------------------------------------------------------- #
# Quick test                                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python miccai_loader.py <sequence_dir>")
        print("Example: python miccai_loader.py C:/Users/dprad/Downloads/train/train/case_1/1")
        sys.exit(1)

    seq_dir = sys.argv[1]
    loader = MICCAILoader(stereo_half="top", smooth_sigma=3.0)

    info = loader.load_video_info(seq_dir)
    w, h = loader.get_frame_size(seq_dir)
    print(f"Sequence info: {info}")
    print(f"Single-view frame size: {w} x {h}")

    traj = loader.load_trajectory(seq_dir)
    print(f"Trajectory shape : {traj.shape}")
    print(f"X range: [{traj[:, 0].min():.1f}, {traj[:, 0].max():.1f}]")
    print(f"Y range: [{traj[:, 1].min():.1f}, {traj[:, 1].max():.1f}]")
    print(f"First 5 points   : {traj[:5]}")
