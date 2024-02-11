from django.shortcuts import render, redirect
from .forms import IrisForm
from .models import IrisPrediction
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from joblib import load

# create your views here
def iris_predict(request):
    if request.method == 'POST':
        form = IrisForm(request.POST)
        if form.is_valid():
            sepal_length = form.cleaned_data['sepal_length']
            sepal_width = form.cleaned_data['sepal_width']
            petal_length = form.cleaned_data['petal_length']
            petal_width = form.cleaned_data['petal_width']
            
            model = load('myapp/iris_model.joblib')    

            prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])       
            
            # Save the prediction
            IrisPrediction.objects.create(
                sepal_length=sepal_length,
                sepal_width=sepal_width,
                petal_length=petal_length,
                petal_width=petal_width,
                prediction=prediction
            )
            
            # Redirect to a new URL to display the prediction
            return render(request, 'myapp/prediction_result.html', {'prediction': prediction})
    else:
        form = IrisForm()
    return render(request, 'myapp/iris_form.html', {'form': form})
