import cv2
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

app = FastAPI()

# Load your YOLO model on the GB10
model = YOLO('yolov8m-pose.pt')
model.to('cuda')

# Global variable to store the latest fall status
is_fall_detected = False

def generate_frames():
    global is_fall_detected
    # Use one of your video paths
    video_path = "datasets/fall_data/Home_01/Home_01/Videos/video (1).avi"
    cap = cv2.VideoCapture(video_path)

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop for demo
            continue

        # 1. Run AI Inference
        results = model(frame, verbose=False, device=0)
        
        # 2. Simple Fall Logic (Placeholder for your full logic)
        # If any person's bounding box is wider than it is tall
        for r in results:
            if r.boxes:
                box = r.boxes.xywh[0]
                w, h = box[2], box[3]
                if w > h * 1.2: # Simple ratio check for the demo
                    is_fall_detected = True
                else:
                    is_fall_detected = False

        # 3. Draw on the frame for the phone to see
        frame = results[0].plot() 
        
        # 4. Encode as JPEG for the stream
        _, buffer = cv2.imencode('.jpg', frame)
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
    # 0.0.0.0 makes it accessible to your Mac/iPhone on the same Wi-Fi
    uvicorn.run(app, host="0.0.0.0", port=8000)