from flask import Flask, request, jsonify
import pickle
import numpy as np
import requests
from flask_cors import CORS
import time
app = Flask(__name__)
CORS(app)

# ==============================
# 🔑 API KEY (ONLY ONE PLACE)
# ==============================
API_KEY = "579b464db66ec23bdd0000016020faa1862045c56647000b8d1696f2"

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
# import requests
# import time

def get_mandi_price(crop, state, district=None):
    url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

    all_records = []

    # 🔁 Controlled pagination (safe)
    for offset in range(0, 300, 100):   # 0,100,200
        try:
            params = {
                "api-key": API_KEY,
                "format": "json",
                "limit": 100,
                "offset": offset
            }

            response = requests.get(url, params=params, timeout=5)
            data = response.json()

            records = data.get("records", [])
            if not records:
                break

            all_records.extend(records)

            time.sleep(1)  # 🔥 avoid API blocking

        except Exception as e:
            print("API Error:", e)
            break

    # 🔍 FILTER DATA
    filtered = []

    for record in all_records:
        commodity = str(record.get("commodity", "")).lower()
        state_name = str(record.get("state", "")).lower()
        district_name = str(record.get("district", "")).lower()

        if crop.lower() in commodity and state.lower() in state_name:
            if district and district.lower() not in district_name:
                continue

            try:
                min_p = float(record.get("min_price", 0))
                max_p = float(record.get("max_price", 0))
                modal_p = float(record.get("modal_price", 0))
            except:
                continue

            filtered.append({
                "market": record.get("market"),
                "district": record.get("district"),
                "min_price": min_p,
                "max_price": max_p,
                "modal_price": modal_p,
                "date": record.get("arrival_date")
            })

    if not filtered:
        return {"message": "No data found"}

    # 🔥 FIND BEST & WORST
    min_record = min(filtered, key=lambda x: x["min_price"])
    max_record = max(filtered, key=lambda x: x["max_price"])

    return {
        "crop": crop,
        "state": state,
        "district": district,
        "total_records": len(filtered),

        "lowest_price_mandi": {
            "market": min_record["market"],
            "price": min_record["min_price"]
        },

        "highest_price_mandi": {
            "market": max_record["market"],
            "price": max_record["max_price"]
        },

        "data": filtered[:10]   # preview (top 10)
    }

# ==============================
# 💰 MANDI API ROUTE
# ==============================
@app.route("/mandi_price", methods=["GET"])
def mandi_price():
    try:
        crop = request.args.get("crop")
        state = request.args.get("state")
        district = request.args.get("district")  # FIXED

        if not crop or not state:
            return jsonify({"error": "crop and state required"}), 400

        result = get_mandi_price(crop, state, district)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# 🚀 RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(debug=True)