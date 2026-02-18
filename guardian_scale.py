import cv2
import torch
from ultralytics import YOLO
import time
import math

# 1. Setup Models and Paths
model = YOLO('yolov8m-pose.pt')
model.to('cuda')

# Define the scenarios and one specific video from each
# Adjust these paths to match your exact filenames
video_sources = [
    "datasets/fall_data/Home_01/Home_01/Videos/video (2).avi",
    "datasets/fall_data/Home_02/Home_02/Videos/video (32).avi",
    "datasets/fall_data/Coffee_room_01/Coffee_room_01/Videos/video (2).avi",
    "datasets/fall_data/Coffee_room_02/Coffee_room_02/Videos/video (50).avi"
]

caps = [cv2.VideoCapture(src) for src in video_sources]
names = ["Home 01", "Home 02", "Coffee Room 01", "Coffee Room 02"]

# 2. State Tracking for multiple streams
# We use lists to keep track of previous positions for each camera independently
prev_hip_coords = [None] * len(caps)
fall_start_times = [None] * len(caps)
fall_velocity_threshold = 20

print("Community Node Active. Monitoring 4 locations on GB10...")

while True:
    frames = []
    for cap in caps:
        success, frame = cap.read()
        if not success:
            # If one video ends, restart it to keep the grid full
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, frame = cap.read()
        
        # Resize each to a standard size for the grid (e.g., 640x480)
        frame = cv2.resize(frame, (640, 480))
        frames.append(frame)

    # 3. BATCH INFERENCE (The GB10 Way)
    # Processing all 4 frames in ONE GPU pass
    start_inference = time.time()
    results = model(frames, verbose=False, conf=0.5, device=0)
    latency_ms = int((time.time() - start_inference) * 1000)

    processed_frames = []

    # 4. Process individual results for each stream
    for i, r in enumerate(results):
        display_frame = frames[i].copy()
        h_frame, w_frame = display_frame.shape[:2]
        
        # Draw Label
        cv2.rectangle(display_frame, (0, 0), (200, 40), (0, 0, 0), -1)
        cv2.putText(display_frame, names[i], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if r.keypoints is not None and len(r.keypoints.data) > 0:
            kpts = r.keypoints.data[0]
            
            # Hip Velocity Logic
            left_hip, right_hip = kpts[11], kpts[12]
            if left_hip[2] > 0.5 and right_hip[2] > 0.5:
                curr_x = (left_hip[0].item() + right_hip[0].item()) / 2
                curr_y = (left_hip[1].item() + right_hip[1].item()) / 2
                
                velocity = 0
                if prev_hip_coords[i] is not None:
                    dx = curr_x - prev_hip_coords[i][0]
                    dy = curr_y - prev_hip_coords[i][1]
                    velocity = math.sqrt(dx**2 + dy**2)
                
                prev_hip_coords[i] = (curr_x, curr_y)
                
                # Aspect Ratio
                box = r.boxes.xyxy[0]
                w, h = (box[2] - box[0]), (box[3] - box[1])
                aspect_ratio = w / h

                # Decision Engine
                is_low = curr_y > (h_frame * 0.65)
                if (aspect_ratio > 1.2 or velocity > fall_velocity_threshold) and is_low:
                    if fall_start_times[i] is None: fall_start_times[i] = time.time()
                    
                    # Alert logic
                    if time.time() - fall_start_times[i] > 0.8:
                        cv2.rectangle(display_frame, (0,0), (w_frame, h_frame), (0,0,255), 8) # Red Border
                        cv2.putText(display_frame, "FALL DETECTED", (w_frame//4, h_frame//2), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 4)
                else:
                    fall_start_times[i] = None

        processed_frames.append(display_frame)

    # 5. CREATE THE GRID
    # Combine 4 frames: [ [1, 2], [3, 4] ]
    top_row = cv2.hconcat([processed_frames[0], processed_frames[1]])
    bottom_row = cv2.hconcat([processed_frames[2], processed_frames[3]])
    grid = cv2.vconcat([top_row, bottom_row])

    # Show Overall System Performance
    cv2.putText(grid, f"GB10 Batch Latency: {latency_ms}ms", (10, grid.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Aegis Community Dashboard - Dell Pro Max", grid)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps: cap.release()
cv2.destroyAllWindows()