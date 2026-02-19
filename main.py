import cv2
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO('yolov8m-pose.pt').to('cuda')

VIDEO_PATH = "datasets/fall_data/Home_01/Home_01/Videos/video (1).avi"

# Global state
privacy_mode     = False
is_fall_detected = False
alerts_enabled   = True


# ------------------------------------------------------------------
# Preprocess: annotate every frame once at startup and cache them.
# This avoids re-running YOLO on the freeze loop, keeping CPU/GPU
# load low while the "frozen" last-2-seconds plays on repeat.
# ------------------------------------------------------------------
def preprocess_video(path: str, freeze_seconds: float = 2.0):
    """
    Returns:
        frames        – list of all annotated JPEG bytes, one per frame
        freeze_start  – index of the frame where the freeze loop begins
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    freeze_frames = max(1, int(fps * freeze_seconds))
    freeze_start  = max(0, total - freeze_frames)

    print(f"[preprocess] {total} frames @ {fps:.1f} fps — "
          f"freeze loop starts at frame {freeze_start} "
          f"(last {freeze_seconds}s = {freeze_frames} frames)")

    frames = []
    idx    = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, verbose=False, device=0)

        # We annotate with the *raw* frame here (no privacy blur).
        # Privacy blur is applied at stream time so toggling it live still works.
        annotated = frame.copy()
        for r in results:
            annotated = r.plot(img=frame.copy(), conf=False, boxes=False)

        _, buf = cv2.imencode('.jpg', annotated)
        frames.append(buf.tobytes())
        idx += 1

    cap.release()
    print(f"[preprocess] Done. Cached {len(frames)} annotated frames.")
    return frames, freeze_start


# Run once at startup — takes a few seconds depending on video length
print("[startup] Preprocessing video…")
ALL_FRAMES, FREEZE_START = preprocess_video(VIDEO_PATH, freeze_seconds=2.0)
print("[startup] Ready.")


# ------------------------------------------------------------------
# Frame generator — plays through once, then loops the freeze window
# ------------------------------------------------------------------
def generate_frames():
    global is_fall_detected, privacy_mode

    # Load the original video once more just for fall detection per-frame
    # (we need the raw frame to run the aspect-ratio check)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Read all raw frames into memory for fall detection sync
    raw_frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        raw_frames.append(f)
    cap.release()

    total = len(raw_frames)

    def fall_check(frame):
        """Run YOLO on a single frame and return whether a fall is detected."""
        results = model(frame, verbose=False, device=0)
        for r in results:
            if r.boxes and len(r.boxes):
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    w, h = float(x2 - x1), float(y2 - y1)
                    if h > 0 and (w / h) > 1.2:
                        return True
        return False

    def apply_privacy(frame_bytes: bytes, raw_frame) -> bytes:
        """
        If privacy mode is on, blur the raw frame and re-annotate.
        Returns JPEG bytes.
        """
        if not privacy_mode:
            return frame_bytes
        blurred = cv2.GaussianBlur(raw_frame, (51, 51), 0)
        results = model(blurred, verbose=False, device=0)
        display = blurred.copy()
        for r in results:
            display = r.plot(img=blurred.copy(), conf=False, boxes=False)
        _, buf = cv2.imencode('.jpg', display)
        return buf.tobytes()

    # ── Play through the full video once ──────────────────────────
    for i in range(total):
        raw   = raw_frames[i]
        annotated_bytes = apply_privacy(ALL_FRAMES[i], raw)
        is_fall_detected = fall_check(raw)

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + annotated_bytes
            + b'\r\n'
        )

    # ── Video ended — freeze-loop the last 2 seconds forever ──────
    # is_fall_detected stays at whatever the last real frame returned,
    # so the iOS app will keep showing the alert as long as this loops.
    print("[stream] Video ended — entering freeze loop.")

    freeze_indices = list(range(FREEZE_START, total))  # e.g. frames 230–250

    loop_idx = 0
    while True:
        i   = freeze_indices[loop_idx % len(freeze_indices)]
        raw = raw_frames[i]
        annotated_bytes = apply_privacy(ALL_FRAMES[i], raw)

        # Re-run fall detection on each freeze frame so privacy toggle
        # still re-annotates correctly and the fall state stays accurate
        is_fall_detected = fall_check(raw)

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + annotated_bytes
            + b'\r\n'
        )

        loop_idx += 1


@app.post("/toggle_privacy")
@app.get("/toggle_privacy")
async def toggle_privacy(value: bool = Query(default=None)):
    global privacy_mode
    privacy_mode = value if value is not None else not privacy_mode
    return {"privacy_mode": privacy_mode}


@app.post("/toggle_alerts")
@app.get("/toggle_alerts")
async def toggle_alerts(value: bool = Query(default=None)):
    global alerts_enabled
    alerts_enabled = value if value is not None else not alerts_enabled
    return {"alerts_enabled": alerts_enabled}


@app.get("/status")
async def get_status():
    return {
        "is_fall":       is_fall_detected,
        "privacy_mode":  privacy_mode,
        "alerts_enabled": alerts_enabled,
        "connected":     True,
    }


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
