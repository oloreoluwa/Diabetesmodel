from django.db import models

class UserInfo(models.Model):
    GENDER_CHOICES = [
        ('male', 'Male'),
        ('female', 'Female'),
        ('other', 'Other'),
    ]

    YES_NO_CHOICES = [
        ('yes', 'Yes'),
        ('no', 'No'),
    ]

    SMOKING_CHOICES = [
        ('no info', 'No Info'),
        ('current', 'Current Smoker'),
        ('ever', 'A Few'),
        ('former', 'Former Smoker'),
        ('never', 'Never Smoked'),
        ('not current', 'Not Currently Smoking'),
    ]

    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    email = models.EmailField(unique=True)

    gender = models.CharField(max_length=10, choices=GENDER_CHOICES, null=True, blank=True)
    age = models.IntegerField(null=True, blank=True)
    hypertension = models.CharField(max_length=3, choices=YES_NO_CHOICES, null=True, blank=True)
    heart_disease = models.CharField(max_length=3, choices=YES_NO_CHOICES, null=True, blank=True)
    smoking_history = models.CharField(max_length=20, choices=SMOKING_CHOICES, null=True, blank=True)

    height = models.FloatField(null=True, blank=True)
    weight = models.FloatField(null=True, blank=True)
    bmi = models.FloatField(null=True, blank=True)

    hba1c_level = models.FloatField(null=True, blank=True)
    blood_glucose_level = models.FloatField(null=True, blank=True)
    prediction_result = models.CharField(max_length=255, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.first_name} {self.last_name} ({self.email})"


class Prediction(models.Model):
    user = models.ForeignKey(UserInfo, on_delete=models.CASCADE, related_name='predictions')
    gender = models.CharField(max_length=10)
    age = models.PositiveIntegerField()
    hypertension = models.CharField(max_length=3)
    heart_disease = models.CharField(max_length=3)
    smoking_history = models.CharField(max_length=20)
    bmi = models.FloatField()
    hba1c_level = models.FloatField()
    blood_glucose_level = models.FloatField()
    result = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.email} - {self.result} on {self.created_at.strftime('%Y-%m-%d %H:%M')}"
