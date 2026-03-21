import requests
import json

url = "http://127.0.0.1:5000/mandi_price"

params = {
    "crop": "Wheat",
    "state": "Madhya Pradesh",
    "district": "Indore AMPC"
}

try:
    response = requests.get(url, params=params)

    print("Status Code:", response.status_code)

    data = response.json()

    print("\nMandi Price Response:")
    print(json.dumps(data, indent=4))

    # ✅ Safe access
    if response.status_code == 200 and "message" not in data:
        print("\nDetails:")
        print("Crop:", data.get("crop"))
        print("Market:", data.get("market"))
        print("District:", data.get("district"))
        print("Min Price:", data.get("min_price"))
        print("Max Price:", data.get("max_price"))
        print("Modal Price:", data.get("modal_price"))
        print("Date:", data.get("date"))
    else:
        print("\nMessage:", data.get("message"))

except Exception as e:
    print("Error:", e)