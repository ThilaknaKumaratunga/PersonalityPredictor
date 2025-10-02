import boto3
import json

# Create SageMaker runtime client
runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")

endpoint_name = "sagemaker-xgboost-2025-10-02-15-18-08-614"  # Replace with your endpoint

input_data = {
    'Time_spent_Alone': 5,
    'Stage_fear': "Yes",
    'Social_event_attendance': 0.2,
    'Going_outside': 1.0,
    'Drained_after_socializing': "No",
    'Friends_circle_size': 0.3,
    'Post_frequency': 10
}

# Convert to JSON string
payload = json.dumps(input_data)

# Call the endpoint
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=payload
)

# Decode response
result = json.loads(response['Body'].read())
print("Prediction:", result["prediction"])
