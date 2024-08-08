from django.urls import path
from . import views

urlpatterns = [
    path('emotion/', views.emotion_detection_view, name='emotion_detection'),
    path('predict-emotion/', views.predict_emotion, name='predict_emotion'),
]
