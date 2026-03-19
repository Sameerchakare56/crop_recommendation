from flask import Flask, request, jsonify
import pickle
import numpy as np
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ==============================
# 🔑 API KEY (ONLY ONE PLACE)
# ==============================
API_KEY = "YOUR_REAL_API_KEY_HERE"

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

        required = ["temperature", "humidity", "moisture", "soil_type", "crop_type", "N", "P", "K"]
        for f in required:
            if f not in data:
                return jsonify({"error": f"Missing {f}"}), 400

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

        probs = fert_model.predict_proba(features)[0]
        top_indices = np.argsort(probs)[-3:][::-1]

        top_ferts = le_fert.inverse_transform(top_indices)
        top_probs = probs[top_indices]

        results = []
        for fert, prob in zip(top_ferts, top_probs):
            results.append({
                "fertilizer": fert,
                "confidence": round(float(prob) * 100, 2)
            })

        return jsonify({
            "top_3_fertilizers": results,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# 💰 MANDI PRICE FUNCTION
# ==============================
def get_mandi_price(crop, state):
    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": 5,
        "filters[commodity]": crop,
        "filters[state]": state
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "records" in data and len(data["records"]) > 0:
        record = data["records"][0]

        return {
            "crop": record.get("commodity"),
            "state": record.get("state"),
            "market": record.get("market"),
            "min_price": record.get("min_price"),
            "max_price": record.get("max_price"),
            "modal_price": record.get("modal_price"),
            "date": record.get("arrival_date")
        }
    else:
        return {"message": "No data found"}


# ==============================
# 💰 MANDI API ROUTE
# ==============================
@app.route("/mandi_price", methods=["GET"])
def mandi_price():
    try:
        crop = request.args.get("crop")
        state = request.args.get("state")

        if not crop or not state:
            return jsonify({"error": "crop and state required"}), 400

        result = get_mandi_price(crop, state)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# 🚀 RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(debug=True)