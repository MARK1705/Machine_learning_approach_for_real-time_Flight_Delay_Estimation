import pandas as pd
import joblib
import requests
from geopy.distance import great_circle
from django.shortcuts import render
from django.http import HttpResponse


# **Load OpenFlights Airport Data**
url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
columns = ["Airport ID", "Name", "City", "Country", "IATA", "ICAO", "Latitude", "Longitude", "Altitude", "Timezone", "DST", "Tz database time zone", "Type", "Source"]
airport_data = pd.read_csv(url, names=columns, index_col=False)

# Convert IATA codes to uppercase for consistency
airport_data["IATA"] = airport_data["IATA"].astype(str).str.upper()

# **Function to Get Airport Coordinates**
def get_airport_coordinates(iata_code):
    """Fetch latitude & longitude of an airport using its IATA code."""
    result = airport_data[airport_data["IATA"] == iata_code.upper()]
    
    if not result.empty:
        lat, lon = float(result.iloc[0]["Latitude"]), float(result.iloc[0]["Longitude"])
        return lat, lon
    
    return None

# **Function to Calculate Airline Distance**
def calculate_airline_distance(iata1, iata2):
    """Calculate the great-circle distance between two airports given their IATA codes."""
    coords1 = get_airport_coordinates(iata1)
    coords2 = get_airport_coordinates(iata2)
    
    if not coords1 or not coords2:
        return None  # Error in fetching coordinates
    
    distance_km = great_circle(coords1, coords2).kilometers
    distance_miles = distance_km * 0.621371  # Convert to miles
    return distance_miles

# **Function to Get Weather Conditions**
def get_weather(city):
    """Fetch real-time weather condition of a city using WeatherAPI."""
    WEATHER_API_KEY = "75204f7ca988479583d201759251407"
    WEATHER_URL = "http://api.weatherapi.com/v1/current.json"
     
     
    try:
        response = requests.get(WEATHER_URL, params={"key": WEATHER_API_KEY, "q": city}, timeout=3)
        data = response.json()
        if "current" in data:
            return data["current"]["condition"]["text"]
    except Exception:
        return "Unknown"
    return "Unknown"

# **Load Trained Models**
import joblib
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Create an uninitialized regressor
xgb_model = xgb.XGBRegressor()
# Load the trained model from JSON
xgb_model.load_model("xgb_delay_model.json")
feature_cols = joblib.load("feature_columns.pkl")

# **Django View for Prediction**
def predict_flight_delay(request):
    if request.method == "POST":
        origin_code = request.POST.get("origin_code", "").strip().upper()
        dest_code = request.POST.get("dest_code", "").strip().upper()
        dep_time = int(request.POST.get("dep_time", 0))
        arr_time = int(request.POST.get("arr_time", 0))
        day_of_week = int(request.POST.get("day_of_week", 0))
        month = int(request.POST.get("month", 0))

        # **Calculate Distance**
        distance = calculate_airline_distance(origin_code, dest_code)

        if distance is None:
            return render(request, "result.html", {"error": "Error: Could not calculate distance."})

        # **Get Weather Conditions**
        origin_weather = get_weather(origin_code)
        dest_weather = get_weather(dest_code)

        # **Convert Weather Conditions to Categorical Codes**
        weather_categories = {'Clear': 0, 'Partly Cloudy': 1, 'Cloudy': 2, 'Rain': 3, 'Thunderstorm': 4, 'Snow': 5, 'Unknown': -1}
        origin_weather_code = weather_categories.get(origin_weather, -1)
        dest_weather_code = weather_categories.get(dest_weather, -1)

        # **Prepare Input Data**
        input_data = pd.DataFrame([[dep_time, arr_time, distance, day_of_week, month, origin_weather_code, dest_weather_code]], columns=feature_cols)

        # **Make Predictions**
        
        delay_pred_xgb = xgb_model.predict(input_data)[0]

        # **Prepare Output**
        result_context = {
            "distance": f"{distance:.2f} miles",
            "origin_weather": origin_weather,
            "dest_weather": dest_weather,
            "delay_message": "The Flight is likely to be On Time" if delay_pred_xgb < 0 else f"The Flight is likely to be Delayed ⏳ Estimated Delay: {delay_pred_xgb:.2f} minutes"
        }

        return render(request, "result.html", result_context)

    return render(request, "forms.html")


def favicon(request):
    return HttpResponse(status=204) 