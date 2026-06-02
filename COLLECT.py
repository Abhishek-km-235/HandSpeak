"""
collect_data.py  —  Hybrid Data Collector
Saves both 128x128 cropped hand images AND landmark CSVs simultaneously.
Usage: Change `label` variable, run once per gesture.
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

# ─── Config ───────────────────────────────────────────────────────────────────
label        = "CUSTOM_10"   # Change for each gesture
IMG_SIZE     = 128
OFFSET       = 60
WRIST_EXTRA  = 60
MAX_SAMPLES  = 500    # Stop after this many captures
CAPTURE_DELAY = 0.5     # Seconds between captures
CSV_FILE     = "landmarks.csv"
DATASET_DIR  = "dataset"
# ──────────────────────────────────────────────────────────────────────────────

mpHands     = mp.solutions.hands
mpDraw      = mp.solutions.drawing_utils
hands       = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)

folder = os.path.join(DATASET_DIR, label)
os.makedirs(folder, exist_ok=True)

# Write CSV header only if file doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["label"]
            + [f"x{i}" for i in range(21)]
            + [f"y{i}" for i in range(21)]
        )

count            = len([f for f in os.listdir(folder) if f.endswith(".jpg")])
last_capture_time = 0

print(f"[INFO] Collecting '{label}' — target: {MAX_SAMPLES} samples")
print("[INFO] Press ESC to quit early")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    imgRGB  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            xList, yList = [], []
            xs, ys = [], []   # ✅ Separate lists → correct CSV column order

            for lm in handLms.landmark:
                px, py = int(lm.x * w), int(lm.y * h)
                xList.append(px)
                yList.append(py)
                xs.append(lm.x)
                ys.append(lm.y)

            # Bounding box with padding
            x_min = max(0,  min(xList) - OFFSET)
            y_min = max(0,  min(yList) - OFFSET)
            x_max = min(w,  max(xList) + OFFSET)
            y_max = min(h,  max(yList) + OFFSET + WRIST_EXTRA)

            imgCrop = frame[y_min:y_max, x_min:x_max]

            if imgCrop.size != 0:
                # ── Place crop on white square canvas ──
                imgWhite   = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
                hC, wC, _  = imgCrop.shape
                aspectRatio = hC / wC

                if aspectRatio > 1:
                    k      = IMG_SIZE / hC
                    wCal   = int(k * wC)
                    resize = cv2.resize(imgCrop, (wCal, IMG_SIZE), interpolation=cv2.INTER_AREA)
                    gap    = (IMG_SIZE - wCal) // 2
                    imgWhite[:, gap:gap + wCal] = resize
                else:
                    k      = IMG_SIZE / wC
                    hCal   = int(k * hC)
                    resize = cv2.resize(imgCrop, (IMG_SIZE, hCal), interpolation=cv2.INTER_AREA)
                    gap    = (IMG_SIZE - hCal) // 2
                    imgWhite[gap:gap + hCal, :] = resize

                cv2.imshow("Hand Crop", imgWhite)

                # ── Auto-save at interval ──
                now = time.time()
                if now - last_capture_time >= CAPTURE_DELAY:
                    # Save image
                    img_path = os.path.join(folder, f"{count}.jpg")
                    cv2.imwrite(img_path, imgWhite)

                    # Save landmarks (all x's then all y's → matches header)
                    with open(CSV_FILE, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([label] + xs + ys)

                    print(f"  Saved [{count+1}/{MAX_SAMPLES}] → {img_path}")
                    count += 1
                    last_capture_time = now

                    if count >= MAX_SAMPLES:
                        print(f"[DONE] Reached {MAX_SAMPLES} samples for '{label}'")
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # HUD
    progress = int((count / MAX_SAMPLES) * 200)
    cv2.rectangle(frame, (10, 60), (210, 80), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, 60), (10 + progress, 80), (0, 255, 100), -1)
    cv2.putText(frame, f"Label: {label}  [{count}/{MAX_SAMPLES}]",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()