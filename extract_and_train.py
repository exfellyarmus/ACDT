"""Pose feature extraction and baseline training using YOLOv8 Pose."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

# COCO keypoint order for YOLOv8 pose
KEYPOINT_NAMES: List[str] = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pose features and train a baseline classifier.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Root directory containing good/ and bad/ folders")
    parser.add_argument("--model", type=str, default="yolov8s-pose.pt", help="YOLOv8 pose model path")
    parser.add_argument("--conf_thr", type=float, default=0.3, help="Confidence threshold for detections/keypoints")
    parser.add_argument("--output", type=Path, default=Path("pose_features.csv"), help="Output CSV path")
    return parser.parse_args()


def angle_between(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Return angle in degrees between two vectors. NaN if zero length."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return math.nan
    cos_theta = float(np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def get_point(idx: int, keypoints: np.ndarray, confs: np.ndarray, conf_thr: float) -> Optional[np.ndarray]:
    if idx >= keypoints.shape[1]:
        return None
    if confs[0, idx] < conf_thr:
        return None
    return keypoints[0, idx]


def compute_features(
    keypoints: np.ndarray, kpt_confs: np.ndarray, conf_thr: float
) -> Optional[Dict[str, float]]:
    def p(name: str) -> Optional[np.ndarray]:
        return get_point(KEYPOINT_NAMES.index(name), keypoints, kpt_confs, conf_thr)

    l_sh, r_sh = p("left_shoulder"), p("right_shoulder")
    l_hip, r_hip = p("left_hip"), p("right_hip")

    if any(pt is None for pt in (l_sh, r_sh, l_hip, r_hip)):
        return None

    mid_shoulder = (l_sh + r_sh) / 2.0
    mid_hip = (l_hip + r_hip) / 2.0

    torso_vec = mid_shoulder - mid_hip
    torso_len = float(np.linalg.norm(torso_vec))
    if torso_len < 1e-6:
        return None

    vertical = np.array([0.0, -1.0])
    torso_angle = angle_between(torso_vec, vertical)

    nose = p("nose")
    l_ear, r_ear = p("left_ear"), p("right_ear")
    head: Optional[np.ndarray] = None
    if nose is not None:
        head = nose
    elif l_ear is not None and r_ear is not None:
        head = (l_ear + r_ear) / 2.0

    forward_head = math.nan
    if head is not None:
        forward_head = float((head[0] - mid_shoulder[0]) / torso_len)

    rounded_shoulder = float((mid_shoulder[0] - mid_hip[0]) / torso_len)

    l_knee, r_knee = p("left_knee"), p("right_knee")
    knee = l_knee if l_knee is not None else r_knee

    hip_knee_angle = math.nan
    if knee is not None:
        knee_vec = knee - mid_hip
        hip_knee_angle = angle_between(torso_vec, knee_vec)

    return {
        "torso_angle": float(torso_angle),
        "forward_head": float(forward_head),
        "rounded_shoulder": float(rounded_shoulder),
        "hip_knee_angle": float(hip_knee_angle),
    }


def iter_image_files(folder: Path) -> Sequence[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def extract_features(args: argparse.Namespace) -> pd.DataFrame:
    model = YOLO(args.model)
    rows: List[Dict[str, float]] = []
    processed = 0
    skipped = 0
    for label_name, label_id in ("good", 0), ("bad", 1):
        label_dir = args.data_dir / label_name
        if not label_dir.exists():
            print(f"[WARN] Missing directory: {label_dir}")
            continue
        for img_path in iter_image_files(label_dir):
            try:
                results = model(img_path, conf=args.conf_thr, verbose=False)
            except Exception as exc:  # minimal exception handling
                print(f"[ERROR] Failed on {img_path}: {exc}")
                skipped += 1
                continue
            if not results:
                skipped += 1
                continue
            res = results[0]
            if res.boxes is None or res.boxes.xyxy is None or res.keypoints is None:
                skipped += 1
                continue

            boxes = res.boxes.xyxy.cpu().numpy()
            if boxes.size == 0:
                skipped += 1
                continue
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_idx = int(np.argmax(areas))

            keypoints = res.keypoints.xy.cpu().numpy()[best_idx : best_idx + 1]
            kpt_confs = res.keypoints.conf.cpu().numpy()[best_idx : best_idx + 1]

            feats = compute_features(keypoints, kpt_confs, args.conf_thr)
            if feats is None:
                skipped += 1
                continue

            row = {
                "path": str(img_path.as_posix()),
                "label": label_id,
                **feats,
            }
            rows.append(row)
            processed += 1
            if processed % 10 == 0:
                print(f"Processed {processed} images...")
    print(f"Extraction done. Processed: {processed}, Skipped: {skipped}")
    return pd.DataFrame(rows)


def train_classifier(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    features = ["torso_angle", "forward_head", "rounded_shoulder", "hip_knee_angle"]
    df_clean = df.dropna(subset=features + ["label"])
    if df_clean.empty:
        print("No valid samples for training.")
        return

    X = df_clean[features]
    y = df_clean["label"]
    if y.nunique() < 2:
        print("Not enough classes for training.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("\nClassification report:\n", report)


def main() -> None:
    args = parse_args()
    df = extract_features(args)
    if df.empty:
        print("No features extracted. Exiting.")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved features to {args.output}")
    train_classifier(args.output)


if __name__ == "__main__":
    main()
