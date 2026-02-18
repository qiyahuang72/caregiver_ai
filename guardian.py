import cv2
import torch
from ultralytics import YOLO
import time

# 1. Initialize Model on GB10
model = YOLO('yolov8m-pose.pt') 
model.to('cuda')

# 2. State Variables
privacy_mode = True 
fall_start_time = None

video_path = "fall_test.mp4" 
cap = cv2.VideoCapture(video_path)

print(f"Guardian AI Active on GB10. Privacy Mode: {'ON' if privacy_mode else 'OFF'}")

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # --- LATENCY TRACKING ---
    start_time = time.time()
    
    # Inference (Optimized for Blackwell)
    results = model(frame, verbose=False, conf=0.5, device=0)
    
    end_time = time.time()
    latency_ms = int((end_time - start_time) * 1000)
    # ------------------------------------

    if privacy_mode:
        display_frame = frame.copy() * 0 
    else:
        display_frame = frame

    frame_height = frame.shape[0]

    for r in results:
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            display_frame = r.plot(img=display_frame, kpt_line=True, labels=False)

            kpts = r.keypoints.data[0] 
            nose_y = kpts[0][1].item()
            
            if nose_y > (frame_height * 0.75):
                if fall_start_time is None:
                    fall_start_time = time.time()
                
                if time.time() - fall_start_time > 2:
                    cv2.putText(display_frame, "!!! EMERGENCY: FALL DETECTED !!!", (50, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            else:
                fall_start_time = None

    # --- UI UPDATES ---
    # Show the GB10 Latency at the top
    cv2.putText(display_frame, f"GB10 Latency: {latency_ms}ms", (10, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    cv2.putText(display_frame, f"GB10 Powered | Privacy: {'ON' if privacy_mode else 'OFF'}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(display_frame, "Press 'p' for Privacy Toggle | 'q' to Quit", 
                (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Aegis - Dell Pro Max GB10", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('p'): privacy_mode = not privacy_mode

cap.release()
cv2.destroyAllWindows()