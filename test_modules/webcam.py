import cv2
import time
import os

# directory to store images
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

# open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

time.sleep(0.5)
start_time = time.time()
image_count = 0

while time.time() - start_time <= 1.5:
    ret, frame = cap.read()
    
    if ret:
        filename = os.path.join(save_dir, f"frame_{image_count:02d}.jpg")
        cv2.imwrite(filename, frame)  # save
        print(f"Saved: {filename}")
        image_count += 1
    else:
        print("Warning: Frame not captured.")
    time.sleep(0.1)

# release resources
cap.release()
print(f"\n{image_count} images saved in '{save_dir}'")