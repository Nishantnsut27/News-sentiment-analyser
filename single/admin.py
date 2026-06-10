from django.contrib import admin
from .models import SentimentFeedback

@admin.register(SentimentFeedback)
class SentimentFeedbackAdmin(admin.ModelAdmin):
    list_display = ('headline', 'predicted_label', 'perceived_label', 'timestamp')

