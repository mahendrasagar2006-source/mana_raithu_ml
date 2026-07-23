import pandas as pd
import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report

# ============================
# Load Dataset
# ============================

data = pd.read_csv("dataset/data_core.csv")

print("Dataset Loaded Successfully!")
print("Shape:", data.shape)
print("\nFirst 5 Rows:")
print(data.head())

# ============================
# Handle Missing Values
# ============================

data["Previous_Crop"] = data["Previous_Crop"].fillna("None")

data.dropna(inplace=True)

print("\nMissing Values:")
print(data.isnull().sum())

# ============================
# Display Categories
# ============================

print("\nUnique States:")
print(sorted(data["State"].unique()))

print("\nUnique Soil Types:")
print(sorted(data["Soil_Type"].unique()))

print("\nUnique Seasons:")
print(sorted(data["Season"].unique()))

print("\nUnique Water Availability:")
print(sorted(data["Water_Availability"].unique()))

print("\nUnique Previous Crops:")
print(sorted(data["Previous_Crop"].unique()))

print("\nUnique Recommended Crops:")
print(sorted(data["Recommended_Crop"].unique()))

print("\nUnique Fertilizer N:")
print(sorted(data["Fertilizer_N"].unique()))

print("\nUnique Fertilizer P:")
print(sorted(data["Fertilizer_P"].unique()))

print("\nUnique Fertilizer K:")
print(sorted(data["Fertilizer_K"].unique()))

# ============================
# Drop District (high cardinality: 67 unique / 580 rows)
# ============================

data.drop("District", axis=1, inplace=True)

# ============================
# Define Features and Targets
# ============================

input_features = [
    "State",
    "Soil_Type",
    "Season",
    "Water_Availability",
    "Previous_Crop",
    "Land_Area_Acres",
    "Temperature_C",
    "Rainfall_mm"
]

# Features
X = data[input_features].copy()

# Targets
y_crop = data["Recommended_Crop"]
y_fert_n = data["Fertilizer_N"]
y_fert_p = data["Fertilizer_P"]
y_fert_k = data["Fertilizer_K"]

# Combine targets for multi-output
targets = pd.DataFrame({
    "Recommended_Crop": y_crop,
    "Fertilizer_N": y_fert_n,
    "Fertilizer_P": y_fert_p,
    "Fertilizer_K": y_fert_k
})

# ============================
# Encode Categorical Features
# ============================

categorical_cols = ["State", "Soil_Type", "Season",
                   "Water_Availability", "Previous_Crop"]

ordinal_maps = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    ordinal_maps[col] = {
        "classes_": list(le.classes_),
        "encoder": le
    }

print("\nEncoded Feature Columns:")
print(X.columns.tolist())
print("\nFeature Shape:", X.shape)

# ============================
# Encode Targets
# ============================

crop_encoder = LabelEncoder()
ferN_encoder = LabelEncoder()
ferP_encoder = LabelEncoder()
ferK_encoder = LabelEncoder()

y_encoded = pd.DataFrame({
    "Recommended_Crop": crop_encoder.fit_transform(y_crop),
    "Fertilizer_N": ferN_encoder.fit_transform(y_fert_n),
    "Fertilizer_P": ferP_encoder.fit_transform(y_fert_p),
    "Fertilizer_K": ferK_encoder.fit_transform(y_fert_k)
})

print("\nTarget Encoded Shape:", y_encoded.shape)

# ============================
# Train Test Split
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded["Recommended_Crop"]
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ============================
# Multi-Output Random Forest
# ============================

base_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model = MultiOutputClassifier(base_model)
model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ============================
# Evaluation
# ============================

y_pred = model.predict(X_test)

print("\n--- Recommended Crop ---")
print("Accuracy: {:.2f}%".format(
    accuracy_score(y_test["Recommended_Crop"], y_pred[:, 0]) * 100
))
print(classification_report(
    y_test["Recommended_Crop"],
    y_pred[:, 0],
    target_names=crop_encoder.classes_
))

print("--- Fertilizer N ---")
print("Accuracy: {:.2f}%".format(
    accuracy_score(y_test["Fertilizer_N"], y_pred[:, 1]) * 100
))

print("--- Fertilizer P ---")
print("Accuracy: {:.2f}%".format(
    accuracy_score(y_test["Fertilizer_P"], y_pred[:, 2]) * 100
))

print("--- Fertilizer K ---")
print("Accuracy: {:.2f}%".format(
    accuracy_score(y_test["Fertilizer_K"], y_pred[:, 3]) * 100
))

# ============================
# Save Model and Artifacts
# ============================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/crop_model.pkl")
joblib.dump(crop_encoder, "models/crop_encoder.pkl")
joblib.dump(ferN_encoder, "models/fert_n_encoder.pkl")
joblib.dump(ferP_encoder, "models/fert_p_encoder.pkl")
joblib.dump(ferK_encoder, "models/fert_k_encoder.pkl")
joblib.dump(ordinal_maps, "models/feature_encoders.pkl")
joblib.dump(input_features, "models/input_features.pkl")

print("\n===================================")
print("Model Saved Successfully!")
print("===================================")
print("Files Saved:")
print("  crop_model.pkl")
print("  crop_encoder.pkl")
print("  fert_n_encoder.pkl")
print("  fert_p_encoder.pkl")
print("  fert_k_encoder.pkl")
print("  feature_encoders.pkl")
print("  input_features.pkl")
