import cv2
import torch
from ultralytics import YOLO
import time
import math
from dataclasses import dataclass
import argparse


def get_device() -> str:
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resize_frame(frame, scale: float = 1.5):
    """Scale frame for display."""
    w = int(frame.shape[1] * scale)
    h = int(frame.shape[0] * scale)
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_privacy(frame, blur_ksize: int = 99):
    """Blur frame for privacy mode."""
    return cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), 0)


def get_hip_velocity(kpts, prev_hip_coords):
    """
    Compute velocity from hip keypoints (11, 12). Returns (curr_hip_xy, velocity).
    curr_hip_xy is (x, y) or None if tracking is lost.
    """
    left_hip, right_hip = kpts[11], kpts[12]
    conf_thresh = 0.5
    if left_hip[2].item() <= conf_thresh or right_hip[2].item() <= conf_thresh:
        return None, 0.0

    curr_x = (left_hip[0].item() + right_hip[0].item()) / 2
    curr_y = (left_hip[1].item() + right_hip[1].item()) / 2
    curr = (curr_x, curr_y)

    velocity = 0.0
    if prev_hip_coords is not None:
        dx = curr_x - prev_hip_coords[0]
        dy = curr_y - prev_hip_coords[1]
        velocity = math.sqrt(dx**2 + dy**2)

    return curr, velocity


def get_aspect_ratio(box_xyxy):
    """Width / height of bounding box."""
    w = box_xyxy[2] - box_xyxy[0]
    h = box_xyxy[3] - box_xyxy[1]
    return w / h if h > 0 else 0


def is_fall_candidate(curr_y, frame_height, aspect_ratio, velocity, velocity_threshold, low_zone_ratio=0.65):
    """True if person is in low zone and (horizontal pose or high velocity)."""
    if curr_y is None:
        return False
    is_low = curr_y > frame_height * low_zone_ratio
    is_horizontal = aspect_ratio > 1.2
    is_fast = velocity > velocity_threshold
    return is_low and (is_horizontal or is_fast)


def draw_fall_alert(display_frame, time_elapsed, confirmed_threshold=0.8):
    """Draw 'ANALYZING...' or 'FALL DETECTED' overlay."""
    if time_elapsed > confirmed_threshold:
        cv2.putText(
            display_frame, "!!! EMERGENCY: FALL DETECTED !!!",
            (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5
        )
    else:
        cv2.putText(
            display_frame, f"ANALYZING MOVEMENT... ({time_elapsed:.1f}s)",
            (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3
        )


def draw_ui_overlay(display_frame, latency_ms, privacy_mode, show_debug, velocity, aspect_ratio):
    """Draw status and debug text on frame."""
    h = display_frame.shape[0]
    cv2.putText(display_frame, f"GB10 Powered | Privacy: {'ON' if privacy_mode else 'OFF'}" if torch.cuda.is_available() else f"MPS Powered | Privacy: {'ON' if privacy_mode else 'OFF'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, f"GB10 Latency: {latency_ms}ms" if torch.cuda.is_available() else f"MPS Latency: {latency_ms}ms",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    if show_debug and not privacy_mode:
        cv2.putText(display_frame, f"Vector Vel: {int(velocity)} | AR: {aspect_ratio:.1f}",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(display_frame, "Press 'p' for Privacy Toggle | 'q' to Quit",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


@dataclass
class FallState:
    """Tracks fall detection state across frames."""
    fall_start_time: float | None = None
    prev_hip_coords: tuple[float, float] | None = None
    velocity_threshold: float = 20.0


def run_guardian(
    video_path: str,
    scale_factor: float = 1.5,
    velocity_threshold: float = 20,
    window_name: str = "Caregiver AI - Dell Pro Max GB10",
):
    """Main loop: load model, run inference, handle display and keyboard."""
    device = get_device()
    model = YOLO("yolov8m-pose.pt")
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    fall_state = FallState(velocity_threshold=velocity_threshold)
    privacy_mode = True

    print(f"Guardian AI Active on GB10. Privacy Mode: {'ON' if privacy_mode else 'OFF'}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = resize_frame(frame, scale_factor)
        frame_height = frame.shape[0]

        t0 = time.perf_counter()
        results = model(frame, verbose=False, conf=0.5, device=device)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        display_frame = apply_privacy(frame) if privacy_mode else frame.copy()
        velocity, aspect_ratio = 0.0, 0.0

        for r in results:
            if r.keypoints is None or len(r.keypoints.data) == 0:
                continue
            display_frame = r.plot(img=display_frame, kpt_line=True, labels=False)
            kpts = r.keypoints.data[0]

            curr_hip, velocity = get_hip_velocity(kpts, fall_state.prev_hip_coords)
            fall_state.prev_hip_coords = curr_hip

            aspect_ratio = get_aspect_ratio(r.boxes.xyxy[0])
            curr_y = curr_hip[1] if curr_hip else None

            if is_fall_candidate(curr_y, frame_height, aspect_ratio, velocity, velocity_threshold):
                if fall_state.fall_start_time is None:
                    fall_state.fall_start_time = time.time()
                draw_fall_alert(display_frame, time.time() - fall_state.fall_start_time)
            else:
                if curr_y is None or curr_y <= frame_height * 0.65:
                    fall_state.fall_start_time = None
            break

        draw_ui_overlay(display_frame, latency_ms, privacy_mode, True, velocity, aspect_ratio)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            privacy_mode = not privacy_mode

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Guardian AI Fall Detection")
    parser.add_argument("--video", type=str, default="datasets/fall_data/Home_01/Home_01/Videos/video (5).avi", help="Path to video file")
    parser.add_argument("--scale", type=float, default=1.5, help="Scale factor for video display")
    parser.add_argument("--threshold", type=int, default=20, help="Velocity threshold for fall detection")
    args = parser.parse_args()
    run_guardian(
        video_path=args.video,
        scale_factor=args.scale,
        velocity_threshold=args.threshold,
    )

if __name__ == "__main__":
    main()
