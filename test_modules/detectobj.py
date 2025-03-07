import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 pretrained model
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" for better accuracy

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Frame not captured.")
        break

    # Run YOLO object detection
    results = model(frame)

    # Draw bounding boxes & labels
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
            label = model.names[int(box.cls[0])]  # Get class name
            conf = box.conf[0]  # Confidence score

            # Highlight detected objects
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
