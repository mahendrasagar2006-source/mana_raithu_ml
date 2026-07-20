from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# ==========================
# Load Model and Artifacts
# ==========================
model = joblib.load("models/fertilizer_model.pkl")
encoder = joblib.load("models/fertilizer_encoder.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# Dropdown values
soil_types = [
    "Black",
    "Clayey",
    "Loamy",
    "Red",
    "Sandy"
]

crop_types = [
    "Barley",
    "Cotton",
    "Ground Nuts",
    "Maize",
    "Millets",
    "Oil seeds",
    "Paddy",
    "Pulses",
    "Sugarcane",
    "Tobacco",
    "Wheat"
]


@app.route("/")
def home():
    return render_template(
        "index.html",
        soil_types=soil_types,
        crop_types=crop_types,
        error=""
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read form values
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        moisture = float(request.form["moisture"])
        nitrogen = float(request.form["nitrogen"])
        potassium = float(request.form["potassium"])
        phosphorous = float(request.form["phosphorous"])

        soil = request.form["soil_type"]
        crop = request.form["crop_type"]

        # Create dictionary with all features = 0
        input_data = {}

        for col in feature_columns:
            input_data[col] = 0

        # Numerical values
        input_data["Temparature"] = temperature
        input_data["Humidity"] = humidity
        input_data["Moisture"] = moisture
        input_data["Nitrogen"] = nitrogen
        input_data["Potassium"] = potassium
        input_data["Phosphorous"] = phosphorous

        # One-hot encoded soil
        soil_col = f"Soil Type_{soil}"
        if soil_col in input_data:
            input_data[soil_col] = 1

        # One-hot encoded crop
        crop_col = f"Crop Type_{crop}"
        if crop_col in input_data:
            input_data[crop_col] = 1

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure column order matches training
        input_df = input_df[feature_columns]

        # Predict
        prediction = model.predict(input_df)

        fertilizer = encoder.inverse_transform(prediction)[0]

        return render_template(
    "result.html",
    prediction=fertilizer,
    probabilities=[],
    soil_type=soil,
    crop_type=crop
)

    except Exception as e:
        return render_template(
            "index.html",
            soil_types=soil_types,
            crop_types=crop_types,
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)