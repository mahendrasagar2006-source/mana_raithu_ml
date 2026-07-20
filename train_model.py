import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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
# Remove Missing Values
# ============================

data.dropna(inplace=True)

print("\nMissing Values:")
print(data.isnull().sum())

# ============================
# Display Categories
# ============================

print("\nUnique Soil Types:")
print(data["Soil Type"].unique())

print("\nUnique Crop Types:")
print(data["Crop Type"].unique())

print("\nUnique Fertilizers:")
print(data["Fertilizer Name"].unique())

# ============================
# One-Hot Encoding
# ============================

categorical_columns = [
    "Soil Type",
    "Crop Type"
]

data = pd.get_dummies(
    data,
    columns=categorical_columns,
    drop_first=False
)

# ============================
# Encode Target
# ============================

encoder = LabelEncoder()

data["Fertilizer Name"] = encoder.fit_transform(
    data["Fertilizer Name"]
)

print("\nProcessed Dataset Shape:", data.shape)

# ============================
# Features and Labels
# ============================

X = data.drop("Fertilizer Name", axis=1)

y = data["Fertilizer Name"]

print("\nInput Features Shape:", X.shape)
print("Output Labels Shape:", y.shape)

print("\nFeature Columns:")
print(X.columns.tolist())

# ============================
# Train Test Split
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ============================
# Random Forest Model
# ============================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ============================
# Prediction
# ============================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)

# ============================
# Save Model
# ============================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/fertilizer_model.pkl"
)

joblib.dump(
    encoder,
    "models/fertilizer_encoder.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "models/feature_columns.pkl"
)

print("\n===================================")
print("Model Saved Successfully!")
print("===================================")

print("Files Saved:")

print("✔ fertilizer_model.pkl")
print("✔ fertilizer_encoder.pkl")
print("✔ feature_columns.pkl")