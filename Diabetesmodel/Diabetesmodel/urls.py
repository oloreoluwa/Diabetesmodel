"""
URL configuration for Diabetesmodel project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'), 
    path('submit-info/', views.submit_info, name='submit_info'),
    path('info/', views.info_form, name='info'),
    path("predict/", views.predict_view, name='predict'),
    path("predict/result", views.result, name='result'),
    path('learn/normal/', views.learn_normal, name='learn_normal'),
    path('learn/pre-diabetic/', views.learn_prediabetes, name='learn_prediabetic'),
    path('learn/diabetic/', views.learn_diabetes, name='learn_diabetic'),
]
