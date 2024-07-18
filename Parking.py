import cv2
import numpy as np

# Load the video for parking lot surveillance
video_path = r'C:\Users\Zablon\Downloads\3848792-uhd_3840_2160_30fps.mp4'  # Using a raw string for Windows path
video = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not video.isOpened():
    print(f"Error: Could not open video '{video_path}'")
    exit()

# Define the background subtraction method
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = video.read()
    
    if not ret or frame is None:
        break
    
    # Apply background subtraction
    fgmask = fgbg.apply(frame)
    
    # Find contours of the detected objects
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Set minimum area threshold for detection
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Parking Lot Occupancy Detection', frame)
    
    # Exit if 'ESC' key is pressed
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()

