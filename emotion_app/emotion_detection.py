from flask import Flask, request, jsonify, Response, stream_with_context, render_template
import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToPILImage, Grayscale, ToTensor, Resize
from .emotions_utils import EmotionNet
from werkzeug.utils import secure_filename

import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Create upload folder if it does not exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
emotion_dict = {0: "Angry", 1: "Fearful", 2: "Happy", 3: "Neutral", 4: "Sad", 5: "Surprised"}

# Initialize the Caffe model for face detection
net = cv2.dnn.readNetFromCaffe('emotion_app/model/deploy.prototxt.txt', 'emotion_app/model/res10_300x300_ssd_iter_140000_fp16.caffemodel')

# Initialize the emotion recognition model
model = EmotionNet(num_of_channels=1, num_of_classes=len(emotion_dict))
model_weights = torch.load('E:/FaceRecognition/Emotion-Detection/EmotionDetection/emotion_app/output/model.pth', map_location=device)
model.load_state_dict(model_weights)
model.to(device)
model.eval()

# Define image transformations
data_transform = transforms.Compose([
    ToPILImage(),
    Grayscale(num_output_channels=1),
    Resize((48, 48)),
    ToTensor()
])

def detect_emotion(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300))
    net.setInput(blob)
    detections = net.forward()

    emotion_result = {"emotion": "Unknown", "probability": 0.0}
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            face = frame[start_y:end_y, start_x:end_x]
            face = data_transform(face)
            face = face.unsqueeze(0)
            face = face.to(device)
            predictions = model(face)
            prob = torch.nn.functional.softmax(predictions, dim=1)
            top_p, top_class = prob.topk(1, dim=1)
            top_p, top_class = top_p.item(), top_class.item()

            emotion_prob = [p.item() for p in prob[0]]
            emotion_value = emotion_dict.values()
            for (i, (emotion, prob)) in enumerate(zip(emotion_value, emotion_prob)):
                if top_class == i:
                    emotion_result = {"emotion": emotion, "probability": top_p * 100}

    return emotion_result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Read the image
        image = cv2.imread(file_path)
        emotion_result = detect_emotion(image)

        return jsonify(emotion_result)

@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(generate_frames()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    vs = cv2.VideoCapture(0)
    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            break
        emotion_result = detect_emotion(frame)
        output = frame.copy()
        face_emotion = emotion_result["emotion"]
        face_text = f"{face_emotion}: {emotion_result['probability']:.2f}%"
        cv2.putText(output, face_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', output)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
