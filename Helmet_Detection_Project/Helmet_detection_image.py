import cv2
import math
import cvzone
from ultralytics import YOLO

# --- Load the trained YOLO model ---
model = YOLO("Weights/best.pt")

# --- Define class names ---
classNames = ['With Helmet', 'Without Helmet']

# --- Load the image (correct path) ---
image_path = "Media/Mediatest_image.jpg"
img = cv2.imread(image_path)

# ✅ Check if the image loaded successfully
if img is None:
    raise FileNotFoundError(f"❌ Image not found at: {image_path}")

# --- Run YOLO detection ---
results = model(img, stream=True)

# --- Draw bounding boxes and labels ---
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(img, (x1, y1, w, h))
        conf = round(float(box.conf[0]), 2)
        cls = int(box.cls[0])
        cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

# --- Resize image to fit screen ---
screen_res = (1280, 720)  # HD size (change if needed)
scale_width = screen_res[0] / img.shape[1]
scale_height = screen_res[1] / img.shape[0]
scale = min(scale_width, scale_height)
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)
resized_img = cv2.resize(img, (window_width, window_height))

# --- Show nicely scaled image ---
cv2.namedWindow("Helmet Detection", cv2.WINDOW_NORMAL)
cv2.imshow("Helmet Detection", resized_img)

print("✅ Press 'Q' or 'Esc' to close.")
while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q') or key == 27:
        break

cv2.destroyAllWindows()
