"""
MICCAI EndoVis Robotic Instrument Tracking Data Loader
=======================================================

This module provides utilities for loading and parsing annotation files from the
MICCAI EndoVis Robotic Instrument Tracking challenge datasets.

Dataset Description
-------------------
The MICCAI EndoVis Robotic Instrument Segmentation & Tracking datasets contain
laparoscopic/endoscopic video sequences with per-frame annotations of robotic
surgical instrument positions.  Depending on the challenge year (2015, 2017, 2018)
the annotations are distributed in several formats:

CSV centroid format (older releases)
    Columns: ``frame``, ``x``, ``y``
    One row per frame giving the (x, y) pixel coordinate of the instrument tip.

CSV bounding-box format
    Columns: ``frame``, ``xmin``, ``ymin``, ``xmax``, ``ymax`` (or similar names
    such as ``x1``, ``y1``, ``x2``, ``y2``).  The centroid is derived as the
    midpoint of the bounding box.

XML bounding-box format (PASCAL VOC style)
    Each frame has a companion ``.xml`` file with ``<bndbox>`` elements containing
    ``<xmin>``, ``<ymin>``, ``<xmax>``, ``<ymax>`` tags.

JSON format
    A single JSON file where each entry maps a frame index to a dict with keys
    ``x``, ``y`` (centroid) or ``xmin``/``ymin``/``xmax``/``ymax`` (bounding box).

Download
--------
    MICCAI 2017: https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/
    MICCAI 2018: https://endovissub2018-roboticscenesegmentation.grand-challenge.org/

Usage
-----
    >>> from src.data import MICCAILoader
    >>> loader = MICCAILoader()
    >>> trajectory = loader.load_trajectory("data/raw/seq_01")
    >>> print(trajectory.shape)  # (N, 2)
"""

from __future__ import annotations

import json
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MICCAILoader:
    """Load per-frame instrument-tip trajectories from MICCAI EndoVis datasets.

    The loader auto-detects the annotation format present in the given sequence
    directory and returns a unified ``(N, 2)`` numpy array of ``(x, y)`` pixel
    coordinates.

    Parameters
    ----------
    csv_filename : str, optional
        Name of the CSV annotation file to look for inside a sequence directory.
        Defaults to ``"annotations.csv"``.
    xml_subdir : str, optional
        Sub-directory (relative to the sequence dir) that may contain per-frame
        XML files.  Defaults to ``"xml"``.
    json_filename : str, optional
        Name of the JSON annotation file to look for.  Defaults to
        ``"annotations.json"``.
    """

    # ------------------------------------------------------------------ #
    # Common column-name aliases for bounding-box CSV files               #
    # ------------------------------------------------------------------ #
    _XMIN_ALIASES = ("xmin", "x1", "left", "bbox_x1")
    _YMIN_ALIASES = ("ymin", "y1", "top", "bbox_y1")
    _XMAX_ALIASES = ("xmax", "x2", "right", "bbox_x2")
    _YMAX_ALIASES = ("ymax", "y2", "bottom", "bbox_y2")
    _X_ALIASES = ("x", "cx", "centroid_x", "center_x")
    _Y_ALIASES = ("y", "cy", "centroid_y", "center_y")
    _FRAME_ALIASES = ("frame", "frame_id", "frame_idx", "index")

    def __init__(
        self,
        csv_filename: str = "annotations.csv",
        xml_subdir: str = "xml",
        json_filename: str = "annotations.json",
    ) -> None:
        self.csv_filename = csv_filename
        self.xml_subdir = xml_subdir
        self.json_filename = json_filename

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load_trajectory(self, sequence_dir: str | os.PathLike) -> np.ndarray:
        """Load the instrument-tip trajectory for an entire sequence.

        The method tries annotation sources in the following priority order:

        1. CSV file named ``self.csv_filename`` in *sequence_dir*.
        2. JSON file named ``self.json_filename`` in *sequence_dir*.
        3. Per-frame XML files located in ``<sequence_dir>/<self.xml_subdir>/``.

        Parameters
        ----------
        sequence_dir : str or path-like
            Path to the sequence directory (e.g. ``"data/raw/seq_01"``).

        Returns
        -------
        np.ndarray
            Array of shape ``(N, 2)`` with columns ``[x, y]`` — pixel coordinates
            of the instrument tip for each of the *N* frames, sorted by frame index.

        Raises
        ------
        FileNotFoundError
            If no recognised annotation file is found in *sequence_dir*.
        ValueError
            If the annotation file is found but cannot be parsed into a valid
            ``(N, 2)`` trajectory.
        """
        seq_dir = Path(sequence_dir)
        if not seq_dir.is_dir():
            raise FileNotFoundError(f"Sequence directory not found: {seq_dir}")

        # 1. Try CSV
        csv_path = seq_dir / self.csv_filename
        if csv_path.is_file():
            logger.info("Loading annotations from CSV: %s", csv_path)
            return self._load_csv(csv_path)

        # Also scan for any *.csv in the directory
        csv_files = sorted(seq_dir.glob("*.csv"))
        if csv_files:
            logger.info("Loading annotations from CSV: %s", csv_files[0])
            return self._load_csv(csv_files[0])

        # 2. Try JSON
        json_path = seq_dir / self.json_filename
        if json_path.is_file():
            logger.info("Loading annotations from JSON: %s", json_path)
            return self._load_json(json_path)

        json_files = sorted(seq_dir.glob("*.json"))
        if json_files:
            logger.info("Loading annotations from JSON: %s", json_files[0])
            return self._load_json(json_files[0])

        # 3. Try per-frame XML files
        xml_dir = seq_dir / self.xml_subdir
        if not xml_dir.is_dir():
            xml_dir = seq_dir  # Fall back to the sequence dir itself

        xml_files = sorted(xml_dir.glob("*.xml"))
        if xml_files:
            logger.info(
                "Loading annotations from %d XML files in: %s",
                len(xml_files),
                xml_dir,
            )
            return self._load_xml_dir(xml_files)

        raise FileNotFoundError(
            f"No recognised annotation files (CSV / JSON / XML) found in: {seq_dir}"
        )

    @staticmethod
    def compute_centroid_from_bbox(
        x1: float, y1: float, x2: float, y2: float
    ) -> Tuple[float, float]:
        """Compute the centroid of an axis-aligned bounding box.

        Parameters
        ----------
        x1, y1 : float
            Top-left corner of the bounding box (pixel coordinates).
        x2, y2 : float
            Bottom-right corner of the bounding box (pixel coordinates).

        Returns
        -------
        (cx, cy) : tuple of float
            Centroid coordinates.

        Examples
        --------
        >>> MICCAILoader.compute_centroid_from_bbox(100, 200, 200, 300)
        (150.0, 250.0)
        """
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return float(cx), float(cy)

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _load_csv(self, path: Path) -> np.ndarray:
        """Parse a CSV annotation file and return an (N, 2) trajectory array."""
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"Failed to read CSV file {path}: {exc}") from exc

        cols_lower = {c.lower().strip(): c for c in df.columns}

        # Sort by frame if a frame column exists
        frame_col = self._find_col(cols_lower, self._FRAME_ALIASES)
        if frame_col is not None:
            df = df.sort_values(by=frame_col).reset_index(drop=True)

        # Case 1: explicit centroid columns
        x_col = self._find_col(cols_lower, self._X_ALIASES)
        y_col = self._find_col(cols_lower, self._Y_ALIASES)
        if x_col is not None and y_col is not None:
            traj = df[[x_col, y_col]].to_numpy(dtype=np.float64)
            self._validate_trajectory(traj, path)
            return traj

        # Case 2: bounding-box columns → compute centroid
        xmin_col = self._find_col(cols_lower, self._XMIN_ALIASES)
        ymin_col = self._find_col(cols_lower, self._YMIN_ALIASES)
        xmax_col = self._find_col(cols_lower, self._XMAX_ALIASES)
        ymax_col = self._find_col(cols_lower, self._YMAX_ALIASES)

        if all(c is not None for c in (xmin_col, ymin_col, xmax_col, ymax_col)):
            cx = (df[xmin_col] + df[xmax_col]) / 2.0
            cy = (df[ymin_col] + df[ymax_col]) / 2.0
            traj = np.stack([cx.to_numpy(), cy.to_numpy()], axis=1).astype(np.float64)
            self._validate_trajectory(traj, path)
            return traj

        raise ValueError(
            f"CSV file {path} does not contain recognised centroid or bounding-box "
            f"columns.  Found columns: {list(df.columns)}"
        )

    def _load_json(self, path: Path) -> np.ndarray:
        """Parse a JSON annotation file and return an (N, 2) trajectory array.

        Supported formats
        -----------------
        List of dicts: ``[{"frame": 0, "x": 100, "y": 200}, ...]``
        Dict keyed by frame: ``{"0": {"x": 100, "y": 200}, ...}``
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            raise ValueError(f"Failed to read JSON file {path}: {exc}") from exc

        # Normalise to list of (frame_index, x, y) tuples
        records: List[Tuple[int, float, float]] = []

        if isinstance(data, list):
            for entry in data:
                frame = int(entry.get("frame", len(records)))
                x, y = self._extract_xy_from_dict(entry, path)
                records.append((frame, x, y))
        elif isinstance(data, dict):
            for key, entry in data.items():
                frame = int(key) if key.isdigit() else len(records)
                x, y = self._extract_xy_from_dict(entry, path)
                records.append((frame, x, y))
        else:
            raise ValueError(
                f"JSON file {path} has an unexpected top-level structure "
                f"(expected list or dict, got {type(data).__name__})"
            )

        records.sort(key=lambda r: r[0])
        traj = np.array([[x, y] for _, x, y in records], dtype=np.float64)
        self._validate_trajectory(traj, path)
        return traj

    def _extract_xy_from_dict(
        self, entry: dict, path: Path
    ) -> Tuple[float, float]:
        """Extract (x, y) centroid from a single annotation dict entry."""
        cols_lower = {k.lower().strip(): k for k in entry}

        # Direct centroid
        x_key = self._find_col(cols_lower, self._X_ALIASES)
        y_key = self._find_col(cols_lower, self._Y_ALIASES)
        if x_key is not None and y_key is not None:
            return float(entry[x_key]), float(entry[y_key])

        # Bounding box → centroid
        xmin_key = self._find_col(cols_lower, self._XMIN_ALIASES)
        ymin_key = self._find_col(cols_lower, self._YMIN_ALIASES)
        xmax_key = self._find_col(cols_lower, self._XMAX_ALIASES)
        ymax_key = self._find_col(cols_lower, self._YMAX_ALIASES)
        if all(k is not None for k in (xmin_key, ymin_key, xmax_key, ymax_key)):
            return self.compute_centroid_from_bbox(
                entry[xmin_key], entry[ymin_key], entry[xmax_key], entry[ymax_key]
            )

        raise ValueError(
            f"Cannot extract (x, y) from JSON entry in {path}: {entry!r}"
        )

    def _load_xml_dir(self, xml_files: List[Path]) -> np.ndarray:
        """Parse a sorted list of per-frame PASCAL VOC XML files."""
        coords: List[Tuple[float, float]] = []
        for xml_path in xml_files:
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                # Look for the first <bndbox> element
                bndbox = root.find(".//bndbox")
                if bndbox is not None:
                    x1 = float(bndbox.findtext("xmin", default="0"))
                    y1 = float(bndbox.findtext("ymin", default="0"))
                    x2 = float(bndbox.findtext("xmax", default="0"))
                    y2 = float(bndbox.findtext("ymax", default="0"))
                    cx, cy = self.compute_centroid_from_bbox(x1, y1, x2, y2)
                else:
                    # Fall back to direct <x>/<y> elements
                    x_el = root.find(".//x")
                    y_el = root.find(".//y")
                    if x_el is None or y_el is None:
                        logger.warning(
                            "No <bndbox> or <x>/<y> found in %s — skipping", xml_path
                        )
                        continue
                    cx, cy = float(x_el.text), float(y_el.text)

                coords.append((cx, cy))
            except ET.ParseError as exc:
                logger.warning("Skipping malformed XML file %s: %s", xml_path, exc)
                continue

        if not coords:
            raise ValueError(
                f"No valid bounding-box annotations could be extracted from the XML files."
            )

        traj = np.array(coords, dtype=np.float64)
        self._validate_trajectory(traj, xml_files[0].parent)
        return traj

    # ------------------------------------------------------------------ #
    # Utilities                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_col(
        cols_lower: dict, aliases: Tuple[str, ...]
    ) -> Optional[str]:
        """Return the original column name matching one of the given aliases."""
        for alias in aliases:
            if alias in cols_lower:
                return cols_lower[alias]
        return None

    @staticmethod
    def _validate_trajectory(traj: np.ndarray, source: object) -> None:
        """Raise ValueError if the trajectory array is degenerate."""
        if traj.ndim != 2 or traj.shape[1] != 2:
            raise ValueError(
                f"Trajectory from {source} has unexpected shape {traj.shape}; "
                f"expected (N, 2)."
            )
        if traj.shape[0] == 0:
            raise ValueError(f"Trajectory from {source} contains no frames.")
        if np.any(np.isnan(traj)):
            raise ValueError(
                f"Trajectory from {source} contains NaN values.  "
                "Check the annotation file for missing or corrupt entries."
            )
