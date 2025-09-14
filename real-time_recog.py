import cv2, json, numpy as np, mediapipe as mp, tensorflow as tf, collections, pyttsx3, time, os

MODEL_PATH = "sign_language_model.h5"
LABELS_PATH = "labels.json"
IMG_SIZE = 64
PADDING = 30
CONF_THRESH = 0.60
SMOOTH_N = 8 

with open(LABELS_PATH) as f:
    idx2label = json.load(f)  
labels = [idx2label[str(i)] for i in range(len(idx2label))]

model = tf.keras.models.load_model(MODEL_PATH)
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

engine = pyttsx3.init()
def say(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except: pass

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
pred_queue = collections.deque(maxlen=SMOOTH_N)
current_text = ""
last_spoken = ""
last_pred = None
last_change_time = time.time()

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6) as hands:
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        pred_label = "-"
        conf = 0.0

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            roi = crop_hand(frame, lm)
            if roi is not None:
                x = roi.astype("float32")/255.0
                x = np.expand_dims(x, axis=0)
                probs = model.predict(x, verbose=0)[0]
                j = int(np.argmax(probs))
                conf = float(probs[j])
                pred_label = labels[j] if conf >= CONF_THRESH else "-"

        pred_queue.append(pred_label)
        stable = max(set(pred_queue), key=pred_queue.count)

        if stable != "-" and stable != last_pred and (time.time()-last_change_time) > 0.6:
            current_text += stable
            last_pred = stable
            last_change_time = time.time()

        cv2.rectangle(frame, (0,0), (frame.shape[1], 90), (0,0,0), -1)
        cv2.putText(frame, f"Pred: {pred_label} ({conf:.2f}) | Stable: {stable}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"Text: {current_text}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("Real-Time Sign Language Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):       
            current_text += " "
        elif key == ord('d'):     
            current_text = current_text[:-1]
        elif key == ord('c'):     
            current_text = ""
        elif key == ord('t'):    
            if current_text.strip() and current_text != last_spoken:
                say(current_text)
                last_spoken = current_text
        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
