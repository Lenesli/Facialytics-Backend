import json
from channels.generic.websocket import WebsocketConsumer
from .emotion_detection import detect_emotion  # Adjust import as necessary

class EmotionConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def receive(self, text_data):
        frame_data = json.loads(text_data)
        emotion = detect_emotion(frame_data['frame'])
        self.send(text_data=json.dumps({
            'emotion': emotion
        }))
