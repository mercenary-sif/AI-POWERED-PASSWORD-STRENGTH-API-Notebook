import requests
import json

# API endpoint
url = "http://127.0.0.1:8000/analyze"

# Password to test
payload = {
    "password": "P@si1&é12!"
}

# Send POST request
response = requests.post(url, json=payload)

# Print status
print("Status Code:", response.status_code)

# Print response
if response.status_code == 200:
    data = response.json()
    print(json.dumps(data, indent=4))
else:
    print("Error:", response.text)