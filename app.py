from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Load model and dataset
model = load_model("my_model.keras")
df = pd.read_csv("Dataset_Example_20days.csv")

# Preprocessing (fit encoders and scaler)
station_encoder = LabelEncoder()
day_encoder = LabelEncoder()
holiday_encoder = LabelEncoder()

df['Station'] = station_encoder.fit_transform(df['Station'])
df['Day of the Week'] = day_encoder.fit_transform(df['Day of the Week'])
df['Holiday/Festival'] = holiday_encoder.fit_transform(df['Holiday/Festival'])

scaler = StandardScaler()
df[['Max Capacity', 'Hour']] = scaler.fit_transform(df[['Max Capacity', 'Hour']])

@app.route("/")
def hello():
    return "Metro Crowd Prediction API is Live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Extract and transform
        station = station_encoder.transform([data["station"]])[0]
        day = day_encoder.transform([data["day"]])[0]
        holiday = holiday_encoder.transform([data["holiday"]])[0]
        hour = float(data["hour"])
        max_capacity = float(data.get("max_capacity", 1800))

        # Create DataFrame
        input_df = pd.DataFrame({
            "Hour": [hour],
            "Station": [station],
            "Max Capacity": [max_capacity],
            "Day of the Week": [day],
            "Holiday/Festival": [holiday]
        })

        # Scale
        input_df[['Max Capacity', 'Hour']] = scaler.transform(input_df[['Max Capacity', 'Hour']])
        input_array = input_df.values.reshape((1, 1, input_df.shape[1]))

        prediction = model.predict(input_array)
        class_idx = int(np.argmax(prediction))
        labels = ["Very Low", "Low", "Medium", "High", "Very High"]

        return jsonify({
            "label": labels[class_idx],
            "confidence": float(np.max(prediction))
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
