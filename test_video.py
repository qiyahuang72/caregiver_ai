import cv2

video_path = "datasets/fall_data/Home_01/Home_01/Videos/video (5).avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video. Check path or codecs.")
else:
    print("Success! Video loaded.")