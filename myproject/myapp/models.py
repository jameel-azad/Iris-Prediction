from django.db import models

# Create your models here.
class IrisPrediction(models.Model):
    sepal_length = models.FloatField()
    sepal_width = models.FloatField()
    petal_length = models.FloatField()
    petal_width = models.FloatField()
    prediction = models.CharField(max_length=20)