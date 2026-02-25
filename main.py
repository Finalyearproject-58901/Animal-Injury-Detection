import cv2
import numpy as np
import datetime
from ultralytics import YOLO

# ===============================
# CONFIGURATION
# ===============================

RTSP_URL = "rtsp://admin:admin123@10.45.0.201:554/avstream/channel=<1>/stream=<0-mainstream;1-substream>.sdp"

CONF_THRESHOLD = 0.5
INJURY_THRESHOLD = 0.75
SKIP_FRAMES = 5  # Process every 5th frame

# COCO Animal Classes
ANIMAL_CLASSES = [15, 16, 17, 18, 19, 20, 21, 22, 23]

# ===============================
# LOAD YOLO MODEL
# ===============================
print("[INFO] Loading YOLO model...")
model = YOLO("yolov8n.pt")

# ===============================
# INJURY DETECTION (Placeholder)
# Replace with trained model later
# ===============================
def detect_injury(cropped_img):
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    variance = np.var(gray)

    injury_probability = min(variance / 5000, 1.0)
    return injury_probability

# ===============================
# CAMERA SETUP (From Your Code)
# ===============================

cap = cv2.VideoCapture(RTSP_URL)

# Reduce delay
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Reduce resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Reduce FPS
cap.set(cv2.CAP_PROP_FPS, 15)

# Display window settings
desired_width = 800
desired_height = 600
cv2.namedWindow('Animal Injury Detection System', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Animal Injury Detection System', desired_width, desired_height)

print("[INFO] Animal Injury Detection System Started...")

# ===============================
# MAIN LOOP
# ===============================

frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    frame_counter += 1

    # Skip frames for performance
    if frame_counter % SKIP_FRAMES != 0:
        cv2.imshow('Animal Injury Detection System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # YOLO Detection
    results = model(frame, verbose=False)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if confidence > CONF_THRESHOLD:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cropped = frame[y1:y2, x1:x2]

                if cropped.size == 0:
                    continue

                injury_prob = detect_injury(cropped)

                label = "Healthy"
                color = (0, 255, 0)

                if injury_prob > INJURY_THRESHOLD:
                    label = "⚠ Injured"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label} ({confidence:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

    cv2.imshow('Animal Injury Detection System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
print("[INFO] System stopped.")