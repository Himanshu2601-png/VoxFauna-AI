from django.db import models


class PredictionHistory(models.Model):

    filename = models.CharField(max_length=255)

    prediction = models.CharField(max_length=50)

    confidence = models.FloatField()

    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.filename