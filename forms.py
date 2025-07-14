from django import forms
import joblib

# Load feature names
feature_cols = joblib.load("feature_columns.pkl")

class FlightDelayForm(forms.Form):
    for feature in feature_cols:
        locals()[feature] = forms.FloatField(label=feature, required=True)
