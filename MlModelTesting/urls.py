from django.urls import path
from .views import *


urlpatterns = [
    path('', Predict.as_view()),
]