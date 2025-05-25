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
import os
from django.conf import settings
from .models import UserInfo, Prediction
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def home(request):
    return render(request, "home.html")

def learn_normal(request):
    return render(request, 'normal.html')

def learn_prediabetes(request):
    return render(request, 'prediabetic.html')

def learn_diabetes(request):
    return render(request, 'diabetic.html')



def submit_info(request):
    if request.method == 'POST':
        try:
            first = request.POST.get('first_name')
            last = request.POST.get('last_name')
            email = request.POST.get('email')

            # Save to session
            request.session['user_email'] = email

            # Check if user already exists
            exists = UserInfo.objects.filter(
                first_name=first,
                last_name=last,
                email=email
            ).exists()

            if exists:
                return render(request, 'info.html', {"message": "You have already submitted this information."})

            if first and last and email:
                UserInfo.objects.create(first_name=first, last_name=last, email=email)
                print("Saved to database")
                return render(request, 'thanks.html')

            return render(request, 'info.html', {"message": "Please fill out all fields."})
        except Exception as e:
            print("Error:", str(e))
            return render(request, 'info.html', {"message": "Something went wrong."})

    return render(request, 'info.html')


def predict_view(request):
    result = request.session.pop('result2', None)
    return render(request, "predict.html", {"result2": result})


def info_form(request):
    return render(request, 'info.html')


def result(request):
    if request.method == "POST":
        try:
            # Load and preprocess dataset
            csv_path = os.path.join(settings.BASE_DIR, 'data', 'diabetes_prediction_dataset.csv')
            data = pd.read_csv(csv_path)

            # Define encoding maps
            gender_map = {"female": 0, "male": 1, "other": 2}
            hypertension_map = {"no": 0, "yes": 1}
            heart_map = {"no": 0, "yes": 1}
            smoking_map = {
                "no info": 0,
                "current": 1,
                "ever": 2,
                "former": 3,
                "never": 4,
                "not current": 5
            }

            # Apply mappings for model training
            data["gender"] = data["gender"].map(gender_map)
            data["hypertension"] = data["hypertension"].map(hypertension_map)
            data["heart_disease"] = data["heart_disease"].map(heart_map)
            data["smoking_history"] = data["smoking_history"].map(smoking_map)

            # Clean data
            data = data.dropna(subset=["diabetes"])
            data = data.dropna(axis=1, how="all")

            # Define features/target
            X = data.drop("diabetes", axis=1)
            y = data["diabetes"]
            used_columns = X.columns.tolist()

            # Impute missing
            imputer = SimpleImputer(strategy="mean")
            X_imputed = imputer.fit_transform(X)

            # Train model
            X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
            model = LogisticRegression(max_iter=2000)
            model.fit(X_train, y_train)

            # --- Get form inputs ---
            gender = request.POST["a1"]                 # String: 'male', 'female'
            age = int(request.POST["a2"])               # Integer
            hypertension = request.POST["a3"]           # String: 'yes' or 'no'
            heart_disease = request.POST["a4"]          # String: 'yes' or 'no'
            smoking_history = request.POST["a5"]        # String: 'not current', etc.
            height = float(request.POST["height"])      # Float
            weight = float(request.POST["weight"])      # Float
            hba1c = float(request.POST["a7"])           # Float
            glucose = float(request.POST["a8"])         # Float

            bmi = weight / (height ** 2)

            # --- Encode for prediction only ---
            gender_val = gender_map.get(gender, 2)
            hypertension_val = hypertension_map.get(hypertension, 0)
            heart_val = heart_map.get(heart_disease, 0)
            smoking_val = smoking_map.get(smoking_history, 5)

            input_dict = {
                "gender": gender_val,
                "age": age,
                "hypertension": hypertension_val,
                "heart_disease": heart_val,
                "smoking_history": smoking_val,
                "bmi": bmi,
                "HbA1c_level": hba1c,
                "blood_glucose_level": glucose
            }

            input_df = pd.DataFrame([input_dict])[used_columns]
            input_imputed = imputer.transform(input_df)
            pred = model.predict(input_imputed)[0]

            # Interpret result
            if glucose > 130:
                result1 = "⚠️ Likely Diabetic"
            elif 100 < glucose <= 130:
                result1 = "⚠️ Likely Pre-Diabetic"
            else:
                result1 = "✅ Normal"

            request.session["result2"] = result1

            # Save prediction with original string values
            email = request.session.get("user_email")
            user = UserInfo.objects.filter(email=email).first()

            if user:
                Prediction.objects.create(
                    user=user,
                    gender=gender,
                    age=age,
                    hypertension=hypertension,
                    heart_disease=heart_disease,
                    smoking_history=smoking_history,
                    bmi=bmi,
                    hba1c_level=hba1c,
                    blood_glucose_level=glucose,
                    result=result1
                )

                user.gender = gender
                user.age = age
                user.hypertension = hypertension
                user.heart_disease = heart_disease
                user.smoking_history = smoking_history
                user.height = height
                user.weight = weight
                user.bmi = bmi
                user.hba1c_level = hba1c
                user.blood_glucose_level = glucose
                user.prediction_result = result1
                user.save()

        except Exception as e:
            print("Prediction error:", str(e))
            return redirect("predict")

        return redirect("predict")

    return redirect("predict")
