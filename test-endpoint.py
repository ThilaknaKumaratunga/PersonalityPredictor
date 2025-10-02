import requests

url = "https://ovran2zv40.execute-api.us-east-1.amazonaws.com/prod/predict"
headers = {"Content-Type": "application/json"}
data = {
    "Time_spent_Alone": 5,
    "Stage_fear": "Yes",
    "Social_event_attendance": 0.2,
    "Going_outside": 1.0,
    "Drained_after_socializing": "No",
    "Friends_circle_size": 10.3,
    "Post_frequency": 10
}

response = requests.post(url, json=data, headers=headers)
print("Status Code:", response.status_code)
print("Response:", response.text)