from django.db import models

class SentimentFeedback(models.Model):
    NEUTRAL = 0
    POSITIVE = 1
    NEGATIVE = 2

    SENTIMENT_CHOICES = [
        (NEUTRAL, 'Neutral'),
        (POSITIVE, 'Positive'),
        (NEGATIVE, 'Negative'),
    ]

    headline = models.TextField()
    predicted_label = models.IntegerField(choices=SENTIMENT_CHOICES)
    perceived_label = models.IntegerField(choices=SENTIMENT_CHOICES, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.headline[:50]

