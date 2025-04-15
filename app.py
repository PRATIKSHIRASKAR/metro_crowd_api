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

# Labels for output
labels = ["Very Low", "Low", "Medium", "High", "Very High"]

@app.route("/")
def hello():
    return "Metro Crowd Prediction API is Live!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # 🔍 Check required fields
        required_keys = ["station", "day", "holiday", "hour", "max_capacity"]
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing required field: {key}"}), 400

        # ✅ Extract input values
        station = data["station"]
        day = data["day"]
        holiday = data["holiday"]
        hour = float(data["hour"])
        max_capacity = float(data["max_capacity"])

        # 🔐 Validate categorical values
        if station not in station_encoder.classes_:
            return jsonify({"error": f"Invalid station '{station}'. Must be one of: {list(station_encoder.classes_)}"}), 400

        if day not in day_encoder.classes_:
            return jsonify({"error": f"Invalid day '{day}'. Must be one of: {list(day_encoder.classes_)}"}), 400

        if holiday not in holiday_encoder.classes_:
            return jsonify({"error": f"Invalid holiday value '{holiday}'. Must be one of: {list(holiday_encoder.classes_)}"}), 400

        # 🔄 Transform input
        station = station_encoder.transform([station])[0]
        day = day_encoder.transform([day])[0]
        holiday = holiday_encoder.transform([holiday])[0]

        # 🧪 Create input DataFrame
        input_df = pd.DataFrame({
            "Hour": [hour],
            "Station": [station],
            "Max Capacity": [max_capacity],
            "Day of the Week": [day],
            "Holiday/Festival": [holiday]
        })

        # 🔧 Scale numeric features
        input_df[['Max Capacity', 'Hour']] = scaler.transform(input_df[['Max Capacity', 'Hour']])
        input_array = input_df.values.reshape((1, 1, input_df.shape[1]))

        # 🔮 Predict
        prediction = model.predict(input_array)
        class_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return jsonify({
            "label": labels[class_idx],
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
