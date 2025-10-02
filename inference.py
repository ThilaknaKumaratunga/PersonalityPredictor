import xgboost as xgb
import numpy as np
import json

def model_fn(model_dir):
    """
    Loads the XGBoost-native model from SageMaker container path.
    """
    booster = xgb.Booster()
    booster.load_model(f"{model_dir}/personality_model.model")
    return booster

def input_fn(request_body, request_content_type):
    return json.loads(request_body)

def predict_fn(input_data, model):
    stage_fear_mapping = {"Yes": 1, "No": 0}
    drained_mapping = {"Yes": 1, "No": 0}

    stage_fear_encoded = stage_fear_mapping.get(input_data["Stage_fear"], 0)
    drained_encoded = drained_mapping.get(input_data["Drained_after_socializing"], 0)

    data = np.array([[input_data["Time_spent_Alone"],
                      stage_fear_encoded,
                      input_data["Social_event_attendance"],
                      input_data["Going_outside"],
                      input_data["Friends_circle_size"],
                      input_data["Post_frequency"]]])

    dmatrix = xgb.DMatrix(data)
    pred = model.predict(dmatrix)
    print("Raw prediction:", pred)
    result = "Extrovert" if pred[0] > 0.5 else "Introvert"
    return result

def output_fn(prediction, content_type):
    return json.dumps({"prediction": prediction})
