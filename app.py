from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# ==========================
# Load Model and Artifacts
# ==========================
model = joblib.load("models/crop_model.pkl")
crop_encoder = joblib.load("models/crop_encoder.pkl")
ferN_encoder = joblib.load("models/fert_n_encoder.pkl")
ferP_encoder = joblib.load("models/fert_p_encoder.pkl")
ferK_encoder = joblib.load("models/fert_k_encoder.pkl")
feature_encoders = joblib.load("models/feature_encoders.pkl")
input_features = joblib.load("models/input_features.pkl")

# Dropdown values (sorted to match encoder)
states = sorted(feature_encoders["State"]["classes_"])
soil_types = sorted(feature_encoders["Soil_Type"]["classes_"])
seasons = sorted(feature_encoders["Season"]["classes_"])
water_availabilities = sorted(feature_encoders["Water_Availability"]["classes_"])
previous_crops = sorted(feature_encoders["Previous_Crop"]["classes_"])


@app.route("/")
def home():
    return render_template(
        "index.html",
        states=states,
        soil_types=soil_types,
        seasons=seasons,
        water_availabilities=water_availabilities,
        previous_crops=previous_crops,
        error=""
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read form values
        state = request.form["state"]
        soil_type = request.form["soil_type"]
        season = request.form["season"]
        water_availability = request.form["water_availability"]
        previous_crop = request.form["previous_crop"]
        land_area = float(request.form["land_area"])
        temperature = float(request.form["temperature"])
        rainfall = float(request.form["rainfall"])

        # Encode categorical features using saved encoders
        input_data = {}

        for col in ["State", "Soil_Type", "Season",
                    "Water_Availability", "Previous_Crop"]:
            form_val = {
                "State": state,
                "Soil_Type": soil_type,
                "Season": season,
                "Water_Availability": water_availability,
                "Previous_Crop": previous_crop
            }[col]
            le = feature_encoders[col]["encoder"]
            input_data[col] = le.transform([form_val])[0]

        # Numerical features
        input_data["Land_Area_Acres"] = land_area
        input_data["Temperature_C"] = temperature
        input_data["Rainfall_mm"] = rainfall

        # Convert to DataFrame in correct feature order
        input_df = pd.DataFrame([input_data])[input_features]

        # Predict
        predictions = model.predict(input_df)

        # Decode predictions
        recommended_crop = crop_encoder.inverse_transform([predictions[0][0]])[0]
        fert_n = ferN_encoder.inverse_transform([predictions[0][1]])[0]
        fert_p = ferP_encoder.inverse_transform([predictions[0][2]])[0]
        fert_k = ferK_encoder.inverse_transform([predictions[0][3]])[0]

        # Get prediction probabilities for recommended crop
        crop_probs = model.estimators_[0].predict_proba(input_df)[0]
        top_indices = np.argsort(crop_probs)[::-1][:5]
        probabilities = [
            (crop_encoder.inverse_transform([i])[0], crop_probs[i])
            for i in top_indices
        ]

        return render_template(
            "result.html",
            prediction=recommended_crop,
            fert_n=fert_n,
            fert_p=fert_p,
            fert_k=fert_k,
            probabilities=probabilities,
            state=state,
            soil_type=soil_type,
            season=season
        )

    except Exception as e:
        return render_template(
            "index.html",
            states=states,
            soil_types=soil_types,
            seasons=seasons,
            water_availabilities=water_availabilities,
            previous_crops=previous_crops,
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)
