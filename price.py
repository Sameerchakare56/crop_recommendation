import requests
import json

# Base URL
url = "http://127.0.0.1:5000/mandi_price"

# Parameters
params = {
    "crop": "Cotton",
    "state": "Madhya Pradesh"
}

try:
    # Send GET request
    response = requests.get(url, params=params)

    print("Status Code:", response.status_code)

    data = response.json()

    print("\n💰 Mandi Price Response:")
    print(json.dumps(data, indent=4))

    # Access values
    if response.status_code == 200:
        print("\n📊 Details:")
        print("Crop:", data.get("crop"))
        print("Market:", data.get("market"))
        print("Min Price:", data.get("min_price"))
        print("Max Price:", data.get("max_price"))
        print("Modal Price:", data.get("modal_price"))

except Exception as e:
    print("❌ Error:", e)