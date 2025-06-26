import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

model = load_model("sketch_model.keras")

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

print(f"✅ Model loaded with {len(class_names)} classes.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = None, None
PINCH_THRESHOLD = 40


def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    roi = cv2.resize(thresh, (64, 64))
    roi = roi.astype("float32") / 255.0
    roi = roi.reshape(1, 64, 64, 1)
    return roi


def predict_canvas(canvas_img):
    roi = preprocess(canvas_img)
    prediction = model.predict(roi)[0]
    index = np.argmax(prediction)
    conf = prediction[index]
    if conf > 0.6:
        return class_names[index]
    return "Uncertain"

print("🖐️ Draw in air using pinch gesture. Press 's' to predict, 'c' to clear, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.ones_like(frame) * 255

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            ix = int(hand.landmark[8].x * w)
            iy = int(hand.landmark[8].y * h)
            tx = int(hand.landmark[4].x * w)
            ty = int(hand.landmark[4].y * h)
            dist = np.hypot(ix - tx, iy - ty)

            if dist > PINCH_THRESHOLD:
                if prev_x is not None:
                    cv2.line(canvas, (prev_x, prev_y), (ix, iy), (0, 0, 0), 8)
                prev_x, prev_y = ix, iy
                cv2.circle(frame, (ix, iy), 10, (0, 255, 0), -1)
            else:
                prev_x, prev_y = None, None
    else:
        prev_x, prev_y = None, None

    combined = cv2.addWeighted(frame, 0.5, canvas.astype(np.uint8), 0.5, 0)
    cv2.imshow("✍️ Air Canvas", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        label = predict_canvas(canvas)
        print(f"🧠 Prediction: {label}")
        cv2.putText(combined, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow("✍️ Air Canvas", combined)
        cv2.waitKey(1000)
    elif key == ord("c"):
        canvas = np.ones_like(frame) * 255
        print("🧹 Canvas cleared.")
    elif key == ord("q"):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
