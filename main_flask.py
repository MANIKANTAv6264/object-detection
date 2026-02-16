from flask import Flask, render_template, Response
import cv2
import numpy as np
import pyttsx3
from collections import defaultdict

app = Flask(__name__)

# ===================== GLOBAL FRAME =====================
latest_frame = None

# ===================== LOAD YOLO =====================
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

with open("coco.names") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# ===================== TEXT TO SPEECH =====================
def speak_object_counts(results):
    if not results:
        return

    engine = pyttsx3.init()

    phrases = []
    for obj, count in results.items():
        if count == 1:
            phrases.append(f"1 {obj} detected")
        else:
            phrases.append(f"{count} {obj}s detected")

    # 🔊 Speak EVERYTHING in ONE sentence
    final_sentence = " and ".join(phrases)

    engine.say(final_sentence)
    engine.runAndWait()


# ===================== LIVE CAMERA STREAM =====================
def generate_frames():
    global latest_frame

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera not accessible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        latest_frame = frame.copy()

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

    cap.release()

# ===================== DETECTION =====================
def detect_objects(frame):
    height, width, _ = frame.shape

    boxes, confidences, class_ids = [], [], []

    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True
    )
    net.setInput(blob)
    outs = net.forward(output_layers)

    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.25:
                center_x = int(det[0] * width)
                center_y = int(det[1] * height)
                w = int(det[2] * width)
                h = int(det[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    counts = defaultdict(int)
    if len(indexes) > 0:
        for i in indexes.flatten():
            label = classes[class_ids[i]]
            counts[label] += 1

    # 🔊 Speak ONLY detected objects ONCE
    speak_object_counts(counts)

    return dict(counts)

# ===================== ROUTES =====================
@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/monitor")
def monitor():
    return render_template("monitor.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/detect")
def detect():
    if latest_frame is None:
        return "No frame captured"

    results = detect_objects(latest_frame)

    # send flag to UI to show speaker icon
    return render_template("detect.html", results=results, speaking=True)

# ===================== RUN =====================
if __name__ == "__main__":
    app.run(debug=True)
