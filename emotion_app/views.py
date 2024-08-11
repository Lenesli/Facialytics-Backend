from django.shortcuts import render
#function based views (FBVs) : take a web request and return a web response 
# Create your views here.
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from .emotions_utils.emotionNet import EmotionNet
from django.conf import settings
import os
from django.http import StreamingHttpResponse

from .emotion_detection import detect_emotion, generate_frames

# Load the pre-trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
emotion_dict = {0: "Angry", 1: "Fearful", 2: "Happy", 3: "Neutral", 4: "Sad", 5: "Surprised"}
model = EmotionNet(num_of_channels=1, num_of_classes=len(emotion_dict))

model_weights_path = os.path.join(settings.BASE_DIR, 'emotion_app/output/model.pth')
model_weights = torch.load(model_weights_path, map_location=device)
model.load_state_dict(model_weights)
model.to(device)
model.eval()

# Define preprocessing transformation
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

def emotion_detection_view(request):
    return render(request, 'emotion.html')

@csrf_exempt
def predict_emotion(request):
    if request.method == 'POST':
        try:
            # Load the image file
            image_file = request.FILES['image']
            image = Image.open(image_file)
            image = np.array(image)

            # Convert the image to grayscale and apply transformations
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = data_transform(image)
            image = image.unsqueeze(0)
            image = image.to(device)

            # Predict emotion
            with torch.no_grad():
                output = model(image)
                prob = torch.nn.functional.softmax(output, dim=1)
                top_p, top_class = prob.topk(1, dim=1)
                top_p, top_class = top_p.item(), top_class.item()

            # Prepare the response
            emotion = emotion_dict[top_class]
            return JsonResponse({'emotion': emotion, 'confidence': top_p})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)

def video_feed(request):
    return StreamingHttpResponse(generate_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')