import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# ==========================
# Load Model and Artifacts
# ==========================
# Use absolute paths so Render can find the files regardless of working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# The full model (crop_model.pkl) is ~1.8 GB and doesn't fit Vercel's
# function size limits. Use the small model by default; override with the
# MODEL_FILE env var if you want a different one.
MODEL_FILE = os.environ.get("MODEL_FILE", "crop_model_50_12.pkl")
model = joblib.load(os.path.join(MODELS_DIR, "small", MODEL_FILE))
crop_encoder = joblib.load(os.path.join(MODELS_DIR, "crop_encoder.pkl"))
ferN_encoder = joblib.load(os.path.join(MODELS_DIR, "fert_n_encoder.pkl"))
ferP_encoder = joblib.load(os.path.join(MODELS_DIR, "fert_p_encoder.pkl"))
ferK_encoder = joblib.load(os.path.join(MODELS_DIR, "fert_k_encoder.pkl"))
feature_encoders = joblib.load(os.path.join(MODELS_DIR, "feature_encoders.pkl"))
input_features = joblib.load(os.path.join(MODELS_DIR, "input_features.pkl"))

# Dropdown values (sorted to match encoder)
states = sorted(feature_encoders["State"]["classes_"])
districts = sorted(feature_encoders["District"]["classes_"])
soil_types = sorted(feature_encoders["Soil_Type"]["classes_"])
seasons = sorted(feature_encoders["Season"]["classes_"])
water_availabilities = sorted(feature_encoders["Water_Availability"]["classes_"])
previous_crops = sorted(feature_encoders["Previous_Crop"]["classes_"])

# State -> district mapping so the district dropdown only shows
# districts of the state the farmer selected.
data = pd.read_csv(os.path.join(BASE_DIR, "dataset", "data_core.csv"))
district_state = data.drop_duplicates("District").set_index("District")["State"].to_dict()
states_districts = {s: sorted(d for d in districts if district_state.get(d) == s) for s in states}


# ==========================
# Routes
# ==========================

@app.route("/")
def home():
    return render_template(
        "index.html",
        states=states,
        districts=districts,
        states_districts=states_districts,
        soil_types=soil_types,
        seasons=seasons,
        water_availabilities=water_availabilities,
        previous_crops=previous_crops,
        error="",
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read form values
        state = request.form["state"]
        district = request.form["district"]
        soil_type = request.form["soil_type"]
        season = request.form["season"]
        water_availability = request.form["water_availability"]
        previous_crop = request.form["previous_crop"]
        land_area = float(request.form["land_area"])
        temperature = float(request.form["temperature"])
        rainfall = float(request.form["rainfall"])

        # Encode categorical features using saved encoders
        input_data = {}

        for col in ["State", "District", "Soil_Type", "Season",
                    "Water_Availability", "Previous_Crop"]:
            form_val = {
                "State": state,
                "District": district,
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
            district=district,
            soil_type=soil_type,
            season=season,
            water_availability=water_availability,
            previous_crop=previous_crop,
            land_area=land_area,
            temperature=temperature,
            rainfall=rainfall,
        )

    except Exception as e:
        return render_template(
            "index.html",
            states=states,
            districts=districts,
            soil_types=soil_types,
            seasons=seasons,
            water_availabilities=water_availabilities,
            previous_crops=previous_crops,
            error=str(e),
        )


if __name__ == "__main__":
    # This runs when you test locally.
    # Render uses "gunicorn app:app" in the Start Command.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
    