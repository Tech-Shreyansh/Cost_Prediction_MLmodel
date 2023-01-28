from django.shortcuts import render
import pandas
from joblib import load
import numpy as np
from rest_framework import generics
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView

model = load('./Savedmodels/ml-models.joblib')

def predicted_price(quantity, volume, distance,flow):
    fields = ([[quantity, volume, distance]])
    field= np.array(fields).reshape((1,-1))
    freight = model.predict(field)
    if flow == 1:
        return freight[0]
    else:
        return 0.9*freight[0]

class Predict(APIView):
    def get(self, request):
        quantity = request.data.get('quantity')
        volume = request.data.get('volume')
        distance = request.data.get('distance')
        flow = request.data.get('flow')
        p = predicted_price(quantity,volume,distance,flow)
        return Response(p)