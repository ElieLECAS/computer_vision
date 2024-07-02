import os
from ultralytics import YOLO

# Define paths relative to the root directory of the project
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

model_path = os.path.join(root_dir, 'test_model', 'fall_detection_flo.pt')
save_path = os.path.join(root_dir, 'test_runs', 'runs', 'detect', 'predict')

# Ensure the save path exists
os.makedirs(save_path, exist_ok=True)

# Initialize the YOLO model with the specified path
model = YOLO(model_path)

# Check if the webcam is accessible before running the model
import cv2
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise ConnectionError("Failed to open webcam. Please ensure the webcam is connected and accessible.")

cap.release()

# Run the model on the webcam feed with streaming mode
results = model(source=0, show=True, conf=0.65, save=True, project=save_path, stream=True)

# Process and display results in a loop
for r in results:
    print(r)  # Display result
    # If you want to access specific attributes, you can do so, e.g., r.boxes, r.masks, etc.
