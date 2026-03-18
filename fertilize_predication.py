import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Fertilizer_Prediction.csv")

# Encode categorical data
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

df["Soil Type"] = le_soil.fit_transform(df["Soil Type"])
df["Crop Type"] = le_crop.fit_transform(df["Crop Type"])
df["Fertilizer Name"] = le_fert.fit_transform(df["Fertilizer Name"])

# Features & target
X = df.drop("Fertilizer Name", axis=1)
y = df["Fertilizer Name"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
print("Accuracy:", model.score(X_test, y_test))

# Save
pickle.dump(model, open("fert_model.pkl", "wb"))
pickle.dump(le_soil, open("soil_encoder.pkl", "wb"))
pickle.dump(le_crop, open("crop_encoder.pkl", "wb"))
pickle.dump(le_fert, open("fert_encoder.pkl", "wb"))