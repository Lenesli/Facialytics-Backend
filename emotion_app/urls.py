from django.urls import path
from .views import predict_emotion, emotion_detection_view,video_feed

urlpatterns = [
    path('', emotion_detection_view, name='emotion_home'),
    path('video_feed/', video_feed, name='video_feed'),
]

#urlpatterns = [
  #  path('emotion/', views.emotion_detection_view, name='emotion_detection'),
 #   path('predict-emotion/', views.predict_emotion, name='predict_emotion'),
#]