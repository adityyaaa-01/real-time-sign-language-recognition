# capture_dataset.py
import cv2, os, time
import numpy as np
import mediapipe as mp

CLASS_NAME = "Z"  # <-- change to your class each run: A,B,C,... or any gesture name
SAVE_DIR = os.path.join("dataset", CLASS_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = 64
BURST_COUNT = 20
PADDING = 30  # extra pixels around the detected hand box

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

def crop_hand(frame, hand_landmarks):
    h, w, _ = frame.shape
    xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
    ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
    x1, x2 = max(min(xs)-PADDING, 0), min(max(xs)+PADDING, w-1)
    y1, y2 = max(min(ys)-PADDING, 0), min(max(ys)+PADDING, h-1)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return None
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    return roi

cap = cv2.VideoCapture(0)
counter = len(os.listdir(SAVE_DIR))

print(f"[INFO] Capturing class: {CLASS_NAME}")
print("[S] save single  [B] burst x20  [Q] quit")

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6) as hands:
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Class: {CLASS_NAME} | Saved: {counter}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and res.multi_hand_landmarks:
            roi = crop_hand(frame, res.multi_hand_landmarks[0])
            if roi is not None:
                cv2.imwrite(os.path.join(SAVE_DIR, f"{counter:05d}.jpg"), roi)
                counter += 1

        elif key == ord('b') and res.multi_hand_landmarks:
            for _ in range(BURST_COUNT):
                ok2, frame2 = cap.read()
                if not ok2: break
                frame2 = cv2.flip(frame2, 1)
                rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                res2 = hands.process(rgb2)
                if res2.multi_hand_landmarks:
                    roi = crop_hand(frame2, res2.multi_hand_landmarks[0])
                    if roi is not None:
                        cv2.imwrite(os.path.join(SAVE_DIR, f"{counter:05d}.jpg"), roi)
                        counter += 1
                cv2.putText(frame2, "BURST...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.imshow("Capture", frame2)
                cv2.waitKey(1)
                time.sleep(0.03)

        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
