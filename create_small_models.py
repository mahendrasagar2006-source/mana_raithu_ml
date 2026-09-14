import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

DATASET = "dataset/data_core.csv"
MODELS_DIR = "models"
SMALL_DIR = "models/small"

os.makedirs(SMALL_DIR, exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(DATASET)

feature_encoders = joblib.load(
    os.path.join(MODELS_DIR, "feature_encoders.pkl")
)

input_features = joblib.load(
    os.path.join(MODELS_DIR, "input_features.pkl")
)

crop_encoder = joblib.load(
    os.path.join(MODELS_DIR, "crop_encoder.pkl")
)

n_encoder = joblib.load(
    os.path.join(MODELS_DIR, "fert_n_encoder.pkl")
)

p_encoder = joblib.load(
    os.path.join(MODELS_DIR, "fert_p_encoder.pkl")
)

k_encoder = joblib.load(
    os.path.join(MODELS_DIR, "fert_k_encoder.pkl")
)


# --------------------------------------------
# Prepare features
# --------------------------------------------

X = df[input_features].copy()

for column in feature_encoders:

    if column in X.columns:

        encoder = feature_encoders[column]["encoder"]

        # Convert missing categorical values to "None"
        X[column] = X[column].fillna("None").astype(str)

        X[column] = encoder.transform(X[column])


# --------------------------------------------
# Prepare targets
# --------------------------------------------

y = df[
    [
        "Recommended_Crop",
        "Fertilizer_N",
        "Fertilizer_P",
        "Fertilizer_K",
    ]
].copy()


# Remove rows with missing target values
valid_rows = (
    y["Recommended_Crop"].notna()
    & y["Fertilizer_N"].notna()
    & y["Fertilizer_P"].notna()
    & y["Fertilizer_K"].notna()
)

X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)


y["Recommended_Crop"] = crop_encoder.transform(
    y["Recommended_Crop"].astype(str)
)

y["Fertilizer_N"] = n_encoder.transform(
    y["Fertilizer_N"].astype(str)
)

y["Fertilizer_P"] = p_encoder.transform(
    y["Fertilizer_P"].astype(str)
)

y["Fertilizer_K"] = k_encoder.transform(
    y["Fertilizer_K"].astype(str)
)


# --------------------------------------------
# Train/test split
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------
# Models
# --------------------------------------------

configs = [
    (50, 10),
    (50, 12),
    (100, 10),
    (100, 15),
]


print()
print("Starting model training...")
print()


for n_estimators, max_depth in configs:

    print("=" * 60)

    print(
        f"Training {n_estimators} trees / "
        f"max_depth={max_depth}"
    )

    base_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model = MultiOutputClassifier(base_model)

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    filename = (
        f"crop_model_{n_estimators}_{max_depth}.pkl"
    )

    filepath = os.path.join(
        SMALL_DIR,
        filename
    )

    joblib.dump(model, filepath)

    size_mb = os.path.getsize(filepath) / (
        1024 * 1024
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Size     : {size_mb:.2f} MB")
    print(f"Saved    : {filepath}")
    print()


print("=" * 60)
print("ALL MODELS TRAINED SUCCESSFULLY")
print("=" * 60)