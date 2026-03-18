from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ==============================
# LOAD MODELS
# ==============================
best_model = pickle.load(open("best_model.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

fert_model = pickle.load(open("fert_model.pkl", "rb"))
le_soil = pickle.load(open("soil_encoder.pkl", "rb"))
le_crop = pickle.load(open("crop_encoder.pkl", "rb"))
le_fert = pickle.load(open("fert_encoder.pkl", "rb"))

print("All models loaded ✅")

# ==============================
# HOME
# ==============================
@app.route("/")
def home():
    return jsonify({"message": "Smart Agriculture API 🚀"})


# ==============================
# 🌱 CROP PREDICTION
# ==============================
@app.route("/predict_crop", methods=["POST"])
def predict_crop():
    try:
        data = request.json

        features = [[
            float(data["N"]),
            float(data["P"]),
            float(data["K"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"])
        ]]

        probs = best_model.predict_proba(features)[0]
        top_indices = np.argsort(probs)[-3:][::-1]

        top_crops = le.inverse_transform(top_indices)
        top_probs = probs[top_indices]

        results = []
        for crop, prob in zip(top_crops, top_probs):
            results.append({
                "crop": crop,
                "confidence": round(float(prob) * 100, 2)
            })

        return jsonify({
            "top_3_crops": results,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# 🧪 FERTILIZER PREDICTION
# ==============================
@app.route("/predict_fertilizer", methods=["POST"])
def predict_fertilizer():
    try:
        data = request.json

        # Validate
        required = ["temperature", "humidity", "moisture", "soil_type", "crop_type", "N", "P", "K"]
        for f in required:
            if f not in data:
                return jsonify({"error": f"Missing {f}"}), 400

        # Encode
        soil = le_soil.transform([data["soil_type"]])[0]
        crop = le_crop.transform([data["crop_type"]])[0]

        features = [[
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["moisture"]),
            soil,
            crop,
            float(data["N"]),
            float(data["K"]),
            float(data["P"])
        ]]

        pred = fert_model.predict(features)
        fertilizer = le_fert.inverse_transform(pred)[0]

        return jsonify({
            "recommended_fertilizer": fertilizer,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# 🔥 COMBINED (OPTIONAL)
# ==============================
@app.route("/predict_all", methods=["POST"])
def predict_all():
    try:
        data = request.json

        # Step 1: Crop prediction
        crop_features = [[
            float(data["N"]),
            float(data["P"]),
            float(data["K"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"])
        ]]

        probs = best_model.predict_proba(crop_features)[0]
        top_indices = np.argsort(probs)[-1:][::-1]
        best_crop = le.inverse_transform(top_indices)[0]

        # Step 2: Fertilizer
        if best_crop not in le_crop.classes_:
            return jsonify({
                "best_crop": best_crop,
                "fertilizer": "Not available"
            })

        soil = le_soil.transform([data["soil_type"]])[0]
        crop = le_crop.transform([best_crop])[0]

        fert_features = [[
            float(data["temperature"]),
            float(data["humidity"]),
            float(data.get("moisture", 40)),
            soil,
            crop,
            float(data["N"]),
            float(data["K"]),
            float(data["P"])
        ]]

        fert_pred = fert_model.predict(fert_features)
        fertilizer = le_fert.inverse_transform(fert_pred)[0]

        return jsonify({
            "best_crop": best_crop,
            "recommended_fertilizer": fertilizer
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    app.run(debug=True)