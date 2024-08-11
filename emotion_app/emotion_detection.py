import cv2
import torch
import numpy as np
from django.http import StreamingHttpResponse
from django.shortcuts import render
from torchvision import transforms
from torchvision.transforms import ToPILImage, Grayscale, ToTensor, Resize
from .emotions_utils import EmotionNet

# Load the models and other necessary components
device = "cuda" if torch.cuda.is_available() else "cpu"
emotion_dict = {0: "Angry", 1: "Fearful", 2: "Happy", 3: "Neutral", 4: "Sad", 5: "Surprised"}

# Initialize the Caffe model for face detection
net = cv2.dnn.readNetFromCaffe('emotion_app/model/deploy.prototxt.txt', 'emotion_app/model/res10_300x300_ssd_iter_140000_fp16.caffemodel')

# Initialize the emotion recognition model
model = EmotionNet(num_of_channels=1, num_of_classes=len(emotion_dict))
model_weights = torch.load('emotion_app/output/model.pth', map_location=device)
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

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (start_x, start_y, end_x, end_y) = box.astype("int")
            face = frame[start_y:end_y, start_x:end_x]

            face = data_transform(face)
            face = face.unsqueeze(0).to(device)

            predictions = model(face)
            prob = torch.nn.functional.softmax(predictions, dim=1)
            top_p, top_class = prob.topk(1, dim=1)
            emotion_label = emotion_dict[top_class.item()]
            emotion_prob = top_p.item() * 100

            # Draw the box around the face
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion_label}: {emotion_prob:.2f}%", (start_x, start_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def generate_frames():
    video_capture = cv2.VideoCapture(0)
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            frame = detect_emotion(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    return render(request, 'index.html')
