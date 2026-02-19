import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

app = FastAPI()
model = YOLO('yolov8m-pose.pt').to('cuda')

# Global state
privacy_mode = False
is_fall_detected = False

@app.get("/toggle_privacy")
async def toggle_privacy():
    global privacy_mode
    privacy_mode = not privacy_mode
    return {"privacy_mode": privacy_mode}

def generate_frames():
    global is_fall_detected, privacy_mode
    cap = cv2.VideoCapture("datasets/fall_data/Home_01/Home_01/Videos/video (1).avi")

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model(frame, verbose=False, device=0)
        
        # Determine the base frame
        if privacy_mode:
            # Create a heavily blurred background
            display_frame = cv2.GaussianBlur(frame, (51, 51), 0)
        else:
            display_frame = frame.copy()

        # Draw skeleton on top of the (blurred or clear) frame
        for r in results:
            # This uses YOLO's internal plotter but on our specific display_frame
            annotated_frame = r.plot(img=display_frame, conf=False, boxes=False)
            
            # Simple Fall Detection Logic
            if r.boxes:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    w, h = x2 - x1, y2 - y1
                    if w > h: # Horizontal person = likely fall
                        is_fall_detected = True
                    else:
                        is_fall_detected = False

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
async def get_status():
    return {"is_fall": is_fall_detected}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
