"""
guardian_mmaction.py  –  Caregiver AI · Fall Detection via PoseC3D + NTU60

Two-stage MMAction2 pipeline (replaces YOLO heuristics in guardian.py):
  Stage 1 – RTMPose-M (via MMPoseInferencer): person detection + 17-keypoint
             pose estimation per frame.
  Stage 2 – PoseC3D (SlowOnly-R50, NTU60-XSub): classifies a buffered
             keypoint sequence into one of 60 actions, including
             'falling' and 'staggering'.

Video loading, display, privacy mode, and fall-alert UI are unchanged.

Setup (one-time):
    conda run -n pytorch_env pip install mmengine mmpose mmdet
    conda run -n pytorch_env pip install "mmcv==2.1.0"
    git clone https://github.com/open-mmlab/mmaction2
    conda run -n pytorch_env pip install -e mmaction2 --no-deps

Usage:
    conda run -n pytorch_env python guardian_mmaction.py \\
        --video "datasets/fall_data/Home_01/Home_01/Videos/video (5).avi"
"""

import collections
import argparse
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch

from mmpose.apis import MMPoseInferencer
from mmaction.apis import inference_skeleton, init_recognizer

# ─────────────────────────────────────────────────────────────────────────────
# PoseC3D model – NTU60 (60 classes, 17 COCO keypoints, 93.6 % top-1)
# ─────────────────────────────────────────────────────────────────────────────
POSEC3D_CONFIG = (
    "mmaction2/configs/skeleton/posec3d/"
    "slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py"
)
POSEC3D_CHECKPOINT = (
    "https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/"
    "slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint/"
    "slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth"
)
LABEL_MAP_PATH = "mmaction2/tools/data/skeleton/label_map_ntu60.txt"

# ─────────────────────────────────────────────────────────────────────────────
# Sliding-window settings
# ─────────────────────────────────────────────────────────────────────────────
CLIP_LEN = 48         # frames per PoseC3D clip (matches training config)
INFERENCE_STRIDE = 16 # run action recognition every N frames

# ─────────────────────────────────────────────────────────────────────────────
# NTU60 fall-related labels (exact strings from label_map_ntu60.txt)
# ─────────────────────────────────────────────────────────────────────────────
FALL_LABELS = {"falling", "staggering"}
FALL_CONFIRM_SECS = 0.8
FALL_SCORE_THRESHOLD = 0.05  # trigger if any fall label exceeds this score

# COCO-17 skeleton connections for drawing
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6),                                    # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers – identical to guardian.py
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resize_frame(frame: np.ndarray, scale: float = 1.5) -> np.ndarray:
    w = int(frame.shape[1] * scale)
    h = int(frame.shape[0] * scale)
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_privacy(frame: np.ndarray, blur_ksize: int = 99) -> np.ndarray:
    return cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), 0)


# ─────────────────────────────────────────────────────────────────────────────
# Pose helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_pose_result(inferencer_output: dict) -> dict:
    """Convert MMPoseInferencer output to the dict format inference_skeleton needs.

    Returns {'keypoints': (N,17,2), 'keypoint_scores': (N,17)}
    Falls back to a silent zero-person placeholder if nobody was detected.
    """
    # predictions is [[person, person, ...]] — outer list = images, inner = persons
    persons = inferencer_output.get("predictions", [[]])[0]
    if not persons:
        return {
            "keypoints": np.zeros((1, 17, 2), dtype=np.float32),
            "keypoint_scores": np.zeros((1, 17), dtype=np.float32),
        }
    kpts = np.array([p["keypoints"] for p in persons], dtype=np.float32)    # (N,17,2)
    scores = np.array([p["keypoint_scores"] for p in persons], dtype=np.float32)  # (N,17)
    return {"keypoints": kpts, "keypoint_scores": scores}


def draw_skeleton(frame: np.ndarray, pose_result: dict,
                  kpt_thr: float = 0.3) -> np.ndarray:
    """Draw COCO-17 skeleton lines and joint dots on *frame* (in-place copy)."""
    out = frame.copy()
    kpts = pose_result["keypoints"]      # (N,17,2)
    scores = pose_result["keypoint_scores"]  # (N,17)

    for person_kpts, person_scores in zip(kpts, scores):
        # Joints
        for i, (x, y) in enumerate(person_kpts):
            if person_scores[i] >= kpt_thr:
                cv2.circle(out, (int(x), int(y)), 4, (0, 255, 0), -1)
        # Bones
        for a, b in COCO_SKELETON:
            if person_scores[a] >= kpt_thr and person_scores[b] >= kpt_thr:
                pt1 = (int(person_kpts[a][0]), int(person_kpts[a][1]))
                pt2 = (int(person_kpts[b][0]), int(person_kpts[b][1]))
                cv2.line(out, pt1, pt2, (0, 200, 255), 2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_fall_alert(frame: np.ndarray, time_elapsed: float) -> None:
    if time_elapsed > FALL_CONFIRM_SECS:
        cv2.putText(frame, "!!! EMERGENCY: FALL DETECTED !!!",
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)
    else:
        cv2.putText(frame, f"ANALYZING MOVEMENT... ({time_elapsed:.1f}s)",
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)


def draw_ui_overlay(frame: np.ndarray, latency_ms: int, privacy_mode: bool,
                    action_label: str, action_score: float,
                    warming_up: bool) -> None:
    h = frame.shape[0]
    hw = "GB10 Powered" if torch.cuda.is_available() else "MPS Powered"
    lat = "GB10 Latency" if torch.cuda.is_available() else "MPS Latency"

    cv2.putText(frame, f"{hw} | Privacy: {'ON' if privacy_mode else 'OFF'} | PoseC3D+NTU60",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"{lat}: {latency_ms}ms",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if warming_up:
        label_text = f"Action: buffering… ({len(action_label)} chars)"
        color = (150, 150, 150)
    else:
        label_text = f"Action: {action_label}  ({action_score:.2f})"
        color = (0, 255, 0) if action_label in FALL_LABELS else (200, 200, 255)

    cv2.putText(frame, label_text,
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, "Press 'p' for Privacy Toggle | 'q' to Quit",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionState:
    current_label: str = "—"
    current_score: float = 0.0
    fall_start_time: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_guardian(
    video_path: str,
    scale_factor: float = 1.5,
    device: Optional[str] = None,
    window_name: str = "Caregiver AI – PoseC3D+NTU60 | Dell Pro Max GB10",
) -> None:
    if device is None:
        device = get_device()

    # ── Load models ──────────────────────────────────────────────────────────
    # mmcv's NMS custom op has no MPS backend, so RTMDet (inside MMPoseInferencer)
    # must run on CPU. PoseC3D is pure PyTorch and runs fine on MPS/CUDA.
    pose_device = "cpu" if device == "mps" else device
    if pose_device != device:
        print(f"[Guardian] Note: RTMDet NMS has no MPS backend — pose on CPU, PoseC3D on {device}")
    print(f"[Guardian] Loading RTMPose-M on {pose_device} …")
    pose_inferencer = MMPoseInferencer("human", device=pose_device)

    print(f"[Guardian] Loading PoseC3D (NTU60) on {device} …")
    action_model = init_recognizer(POSEC3D_CONFIG, POSEC3D_CHECKPOINT, device=device)

    labels = [l.strip() for l in open(LABEL_MAP_PATH).readlines()]
    print(f"[Guardian] Ready. {len(labels)} NTU60 action classes loaded.")

    # ── Video capture ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    pose_buffer: collections.deque[dict] = collections.deque(maxlen=CLIP_LEN)
    action_state = ActionState()
    privacy_mode = True
    frame_idx = 0
    latency_ms = 0
    img_shape = None  # set on first frame

    print(f"[Guardian] Active. Privacy: {'ON' if privacy_mode else 'OFF'}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = resize_frame(frame, scale_factor)
        if img_shape is None:
            img_shape = (frame.shape[0], frame.shape[1])

        frame_idx += 1

        # ── Stage 1: Pose estimation (every frame) ────────────────────────────
        pose_out = next(pose_inferencer(frame, show=False, return_vis=False))
        pose_result = extract_pose_result(pose_out)
        pose_buffer.append(pose_result)

        # ── Stage 2: Action recognition (every INFERENCE_STRIDE frames) ───────
        warming_up = len(pose_buffer) < CLIP_LEN
        if not warming_up and frame_idx % INFERENCE_STRIDE == 0:
            t0 = time.perf_counter()
            result = inference_skeleton(action_model, list(pose_buffer), img_shape)
            latency_ms = int((time.perf_counter() - t0) * 1000)

            scores: list[float] = result.pred_score.tolist()
            top_idx = int(np.argmax(scores))
            action_state.current_label = labels[top_idx]
            action_state.current_score = scores[top_idx]

            # Fall detection: check each fall label's score directly, not just top-1.
            # NTU60 trained in lab conditions vs real-world footage means fall rarely
            # wins top-1, but its score is reliably elevated during actual falls.
            fall_score = max(scores[labels.index(lbl)] for lbl in FALL_LABELS)
            is_falling = fall_score >= FALL_SCORE_THRESHOLD
            top1 = labels[top_idx]
            print(f"[DEBUG] top1={top1}({action_state.current_score:.2f}) fall_score={fall_score:.2f} is_falling={is_falling}", flush=True)

            if is_falling:
                # Show the highest-scoring fall label in the UI
                fall_label = max(FALL_LABELS, key=lambda lbl: scores[labels.index(lbl)])
                action_state.current_label = fall_label
                action_state.current_score = fall_score
                if action_state.fall_start_time is None:
                    action_state.fall_start_time = time.time()
            else:
                action_state.fall_start_time = None

        # ── Display ───────────────────────────────────────────────────────────
        if privacy_mode:
            display_frame = apply_privacy(frame)
            # Draw skeleton on blurred frame so no raw footage is shown
            display_frame = draw_skeleton(display_frame, pose_result)
        else:
            display_frame = draw_skeleton(frame.copy(), pose_result)

        if action_state.fall_start_time is not None:
            draw_fall_alert(display_frame, time.time() - action_state.fall_start_time)

        draw_ui_overlay(display_frame, latency_ms, privacy_mode,
                        action_state.current_label, action_state.current_score,
                        warming_up)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            privacy_mode = not privacy_mode

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guardian AI – PoseC3D + NTU60 Fall Detection")
    parser.add_argument(
        "--video", type=str,
        default="datasets/fall_data/Home_01/Home_01/Videos/video (5).avi",
        help="Path to video file")
    parser.add_argument("--scale", type=float, default=1.5,
                        help="Display scale factor")
    parser.add_argument("--device", type=str, default=get_device(),
                        help="Device: cuda / mps / cpu (auto-detected if omitted)")
    args = parser.parse_args()

    run_guardian(
        video_path=args.video,
        scale_factor=args.scale,
        device=args.device,
    )


if __name__ == "__main__":
    main()
