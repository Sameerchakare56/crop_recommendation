import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ==============================
# 1. LOAD DATA
# ==============================
df = pd.read_csv("Crop_recommendation.csv")

print("Dataset Loaded Successfully ✅")
print(df.head())

# ==============================
# 2. PREPARE DATA
# ==============================
X = df.drop("label", axis=1)
y = df["label"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ==============================
# 3. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ==============================
# 4. TRAIN RANDOM FOREST (PRIMARY)
# ==============================
print("\nTraining RandomForest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_acc = rf_model.score(X_test, y_test)
print("RandomForest Accuracy:", rf_acc)

# ==============================
# 5. TRAIN XGBOOST (BACKUP)
# ==============================
print("\nTraining XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=100,
    eval_metric='mlogloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)

xgb_acc = xgb_model.score(X_test, y_test)
print("XGBoost Accuracy:", xgb_acc)

# ==============================
# 6. SELECT BEST MODEL
# ==============================
if rf_acc >= xgb_acc:
    best_model = rf_model
    best_name = "RandomForest"
else:
    best_model = xgb_model
    best_name = "XGBoost"

print("\nBest Model Selected:", best_name)

# ==============================
# 7. SAVE MODELS
# ==============================
pickle.dump(best_model, open("best_model.pkl", "wb"))
pickle.dump(rf_model, open("rf_model.pkl", "wb"))
pickle.dump(xgb_model, open("xgb_model.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("\nModels Saved Successfully ✅")