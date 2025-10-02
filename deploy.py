from sagemaker.xgboost.model import XGBoostModel
from sagemaker.serverless import ServerlessInferenceConfig

role = "arn:aws:iam::730335511841:role/sagemaker-role-personality-predictor"

model = XGBoostModel(
    model_data="s3://classification-task-thilakna-kumaratunga/personality_model.tar.gz",
    role=role,
    entry_point="inference.py",
    framework_version="1.7-1",   
)

predictor = model.deploy(
    serverless_inference_config=ServerlessInferenceConfig(
        memory_size_in_mb=1024,
        max_concurrency=5
    )
)

