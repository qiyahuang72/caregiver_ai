import cv2
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI()

# Allow requests from the iOS app (WKWebView / URLSession)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO('yolov8m-pose.pt').to('cuda')

# Global state
privacy_mode = False
is_fall_detected = False
alerts_enabled = True


# ------------------------------------------------------------------
# Privacy toggle — now accepts an explicit ?value=true/false
# so the iOS app and server never get out of sync.
# Falls back to flip-mode if no value is provided.
# ------------------------------------------------------------------
@app.post("/toggle_privacy")
@app.get("/toggle_privacy")          # keep GET so curl testing still works
async def toggle_privacy(value: bool = Query(default=None)):
    global privacy_mode
    if value is not None:
        privacy_mode = value
    else:
        privacy_mode = not privacy_mode
    return {"privacy_mode": privacy_mode}


@app.post("/toggle_alerts")
@app.get("/toggle_alerts")
async def toggle_alerts(value: bool = Query(default=None)):
    global alerts_enabled
    if value is not None:
        alerts_enabled = value
    else:
        alerts_enabled = not alerts_enabled
    return {"alerts_enabled": alerts_enabled}


# ------------------------------------------------------------------
# Status — now returns ALL state the iOS app needs in one call.
# This fixes the "always offline" bug caused by decode failure.
# ------------------------------------------------------------------
@app.get("/status")
async def get_status():
    return {
        "is_fall": is_fall_detected,
        "privacy_mode": privacy_mode,
        "alerts_enabled": alerts_enabled,
        "connected": True           # If this endpoint responds, we're connected
    }


# ------------------------------------------------------------------
# Frame generator
# ------------------------------------------------------------------
def generate_frames():
    global is_fall_detected, privacy_mode

    cap = cv2.VideoCapture(
        "datasets/fall_data/Home_01/Home_01/Videos/video (1).avi"
    )

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model(frame, verbose=False, device=0)

        if privacy_mode:
            display_frame = cv2.GaussianBlur(frame, (51, 51), 0)
        else:
            display_frame = frame.copy()

        annotated_frame = display_frame  # fallback if no detections

        for r in results:
            annotated_frame = r.plot(img=display_frame, conf=False, boxes=False)

            # Fall detection: bounding box wider than tall → person is horizontal
            if r.boxes and len(r.boxes):
                fell = False
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    w, h = float(x2 - x1), float(y2 - y1)
                    if h > 0 and (w / h) > 1.2:   # ratio threshold, tweak as needed
                        fell = True
                        break
                is_fall_detected = fell
            else:
                is_fall_detected = False

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + buffer.tobytes()
            + b'\r\n'
        )


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
