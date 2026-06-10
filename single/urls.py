from django.urls import path
from . import views

urlpatterns = [
    path('', views.single, name='single'),
    path('analyze_sentiment/', views.analyze_sentiment_view, name='analyze_sentiment'),
    path('store_feedback/', views.store_feedback_view, name='store_feedback_view'),
]
