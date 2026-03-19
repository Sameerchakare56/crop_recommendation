import requests
import json

BASE_URL = "http://127.0.0.1:5000"

# ==============================
# 🌱 TEST CROP API
# ==============================
def test_crop():
    url = f"{BASE_URL}/predict_crop"

    data = {
        "N": 90,
        "P": 40,
        "K": 40,
        "temperature": 25,
        "humidity": 80,
        "ph": 6.5,
        "rainfall": 200
    }

    print("\n🌱 Testing Crop API...")
    response = requests.post(url, json=data)

    print("Status:", response.status_code)
    print(json.dumps(response.json(), indent=4))


# ==============================
# 🧪 TEST FERTILIZER API
# ==============================
def test_fertilizer():
    url = f"{BASE_URL}/predict_fertilizer"

    data = {
        "temperature": 30,
        "humidity": 60,
        "moisture": 40,
        "soil_type": "Black",
        "crop_type": "Cotton",
        "N": 0,
        "P": 0,
        "K": 0
    }

    print("\n🧪 Testing Fertilizer API...")
    response = requests.post(url, json=data)

    print("Status:", response.status_code)
    print(json.dumps(response.json(), indent=4))


# ==============================
# 🔥 TEST COMBINED API
# ==============================
def test_all():
    url = f"{BASE_URL}/predict_all"

    data = {
        "N": 100,
        "P": 50,
        "K": 40,
        "temperature": 30,
        "humidity": 65,
        "ph": 7.2,
        "rainfall": 120,
        "soil_type": "Black",
        "moisture": 40
    }

    print("\n🔥 Testing Combined API...")
    response = requests.post(url, json=data)

    print("Status:", response.status_code)
    print(json.dumps(response.json(), indent=4))


# ==============================
# RUN ALL TESTS
# ==============================
if __name__ == "__main__":
    test_crop()
    test_fertilizer()
    test_all()