import cv2
import math
import cvzone
from ultralytics import YOLO
from tqdm import tqdm
import time

# --- Load video and model ---
video_path = "Media/Sample.mp4"   # <-- change if your video name/path is different
cap = cv2.VideoCapture(video_path)
model = YOLO("Weights/best.pt")   # or use "yolov8n.pt" for faster results
classNames = ['With Helmet', 'Without Helmet']

# --- Check video ---
if not cap.isOpened():
    print("❌ Error: Cannot open video file.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"🎥 Total Frames: {total_frames} | FPS: {fps}")

# --- Progress bar setup ---
pbar = tqdm(total=total_frames, desc="Processing Video", ncols=80)

while True:
    success, img = cap.read()
    if not success or img is None:
        print("✅ End of video reached.")
        break

    # --- Resize for speed (optional) ---
    # comment this line if you want original resolution
    img = cv2.resize(img, (1280, 720))  # adjust to (640, 360) if too slow

    # --- YOLO detection ---
    results = model(img, stream=True)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # --- Time overlay ---
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    elapsed_time = current_frame / fps
    cv2.putText(img, f"Time: {elapsed_time:.1f}s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # --- Fullscreen window ---
    cv2.namedWindow("Helmet Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Helmet Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Helmet Detection", img)

    # --- Exit key ---
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q') or key == 27:  # q or Esc
        print("🛑 Stopped manually.")
        break

    pbar.update(1)

cap.release()
cv2.destroyAllWindows()
pbar.close()
print("✅ Done. Window closed successfully.")
