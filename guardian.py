import cv2
import torch
from ultralytics import YOLO
import time
import math

# 1. Initialize Model on GB10
model = YOLO('yolov8m-pose.pt') 
model.to('cuda')

# 2. Global State Variables (Must be OUTSIDE the loop)
privacy_mode = True 
fall_start_time = None
prev_hip_coords = None  # Stores (x, y)
fall_velocity_threshold = 20  # Total pixel distance moved per frame
scale_factor = 1.5  # Increase this to 2.0 for even larger video

video_path = "datasets/fall_data/Home_01/Home_01/Videos/video (5).avi"
cap = cv2.VideoCapture(video_path)

print(f"Guardian AI Active on GB10. Privacy Mode: {'ON' if privacy_mode else 'OFF'}")

while cap.isOpened():
    success, frame = cap.read()
    if not success: 
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # 1. RESIZE THE FRAME (Make it larger)
    width = int(frame.shape[1] * scale_factor)
    height = int(frame.shape[0] * scale_factor)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    # --- LATENCY TRACKING ---
    start_time = time.time()
    results = model(frame, verbose=False, conf=0.5, device=0)
    end_time = time.time()
    latency_ms = int((end_time - start_time) * 1000)

    # --- BACKGROUND PROCESSING ---
    if privacy_mode:
        display_frame = cv2.GaussianBlur(frame, (99, 99), 0)
    else:
        display_frame = frame

    frame_height = frame.shape[0]

    for r in results:
        if r.keypoints is not None and len(r.keypoints.data) > 0:
            # Draw skeleton on top of blur/frame
            display_frame = r.plot(img=display_frame, kpt_line=True, labels=False)
            
            kpts = r.keypoints.data[0]
            
            # 1. HIP-BASED VECTOR VELOCITY
            # Using average of Hips (points 11 and 12) for stability
            left_hip = kpts[11]
            right_hip = kpts[12]
            
            if left_hip[2].item() > 0.5 and right_hip[2].item() > 0.5:
                curr_x = (left_hip[0].item() + right_hip[0].item()) / 2
                curr_y = (left_hip[1].item() + right_hip[1].item()) / 2
                
                velocity = 0
                if prev_hip_coords is not None:
                    dx = curr_x - prev_hip_coords[0]
                    dy = curr_y - prev_hip_coords[1]
                    # Pythagorean Theorem: Captures forward, vertical, or diagonal movement
                    velocity = math.sqrt(dx**2 + dy**2)
                
                prev_hip_coords = (curr_x, curr_y)
            else:
                velocity = 0
                curr_y = None # Lost tracking

            # 2. ASPECT RATIO (Box Flip)
            box = r.boxes.xyxy[0]
            w, h = (box[2] - box[0]), (box[3] - box[1])
            aspect_ratio = w / h

            # 3. DECISION ENGINE
            # Logic: If person is horizontal OR moving with high "impact" velocity while in the low zone
            is_low = curr_y > (frame_height * 0.65) if curr_y else False
            is_horizontal = aspect_ratio > 1.2
            is_fast = velocity > fall_velocity_threshold

            if (is_horizontal or is_fast) and is_low:
                if fall_start_time is None:
                    fall_start_time = time.time()
                
                # Visual feedback BEFORE the 0.8s threshold
                time_elapsed = time.time() - fall_start_time

                if time_elapsed > 0.8: # Confirmed fall
                    cv2.putText(display_frame, "!!! EMERGENCY: FALL DETECTED !!!", (50, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)
                else:
                    # PENDING ALERT (Yellow - shows you it's working instantly)
                    cv2.putText(display_frame, f"ANALYZING MOVEMENT... ({time_elapsed:.1f}s)", (50, 150), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    
            else:
                if not is_low: # Reset if they are back in the upper zones
                    fall_start_time = None

            # --- DEBUG OVERLAY ---
            if not privacy_mode:
                cv2.putText(display_frame, f"Vector Vel: {int(velocity)} | AR: {aspect_ratio:.1f}", 
                            (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # --- UI UPDATES ---
    cv2.putText(display_frame, f"GB10 Latency: {latency_ms}ms", (10, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(display_frame, f"GB10 Powered | Privacy: {'ON' if privacy_mode else 'OFF'}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, "Press 'p' for Privacy Toggle | 'q' to Quit", 
                (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Caregiver AI - Dell Pro Max GB10", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('p'): privacy_mode = not privacy_mode

cap.release()
cv2.destroyAllWindows()