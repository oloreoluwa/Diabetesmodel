from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from django.urls import reverse

def home(request):
    return render(request, "home.html")

# Predict page function
def predict(request):
    result = request.session.pop('result2', None)  # Remove after getting
    return render(request, "predict.html", {"result2": result})

def result(request):
    if request.method == "POST":
        data = pd.read_csv(r"C:\Users\oniyi\Desktop\system 1\data\diabetes_prediction_dataset.csv")

        encoder = LabelEncoder()
        if "gender" in data.columns:
            data["gender"] = encoder.fit_transform(data["gender"])
        if "smoking_history" in data.columns:
            data["smoking_history"] = encoder.fit_transform(data["smoking_history"])

        X = data.drop("diabetes", axis=1)
        Y = data["diabetes"]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, Y_train)

        # Get input values from POST
        val1 = float(request.POST["a1"])
        val2 = float(request.POST["a2"])
        val3 = float(request.POST["a3"])
        val4 = float(request.POST["a4"])
        val5 = float(request.POST["a5"])
        val6 = float(request.POST["a6"])
        val7 = float(request.POST["a7"])
        val8 = float(request.POST["a8"])

        # Make prediction
        input_data = pd.DataFrame([[val1, val2, val3, val4, val5, val6, val7, val8]],
                                  columns=X.columns)
        pred = model.predict(input_data)
        if pred == [1]:
            result1 = "⚠️ You have diabetes or high chances of having diabetes."
        else:
            result1 = "✅ You do not have diabetes or a low risk of having diabetes."

        request.session['result2'] = result1

        # Redirect to predict page
        return redirect('predict')

    return redirect('predict')