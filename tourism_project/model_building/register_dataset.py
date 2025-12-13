
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login
import os

# Define dataset repository name
DATASET_REPO = "SharleyK/TourismPackagePrediction"

# # Login
# login(token=os.environ.get('HF_TOKEN'))
# api = HfApi(login)
# # api = HfApi(token=os.getenv("HF_TOKEN"))


# Load the raw dataset
df = pd.read_csv('tourism_project/data/tourism.csv')
print(f"Dataset loaded: {df.shape}")

# Create Hugging Face dataset
dataset = Dataset.from_pandas(df)

# Create dataset repository
try:
    api.create_repo(
        repo_id='SharleyK/TourismPackagePrediction',
        repo_type='dataset',
        exist_ok=True
    )
    print("✅ Dataset repository created")
except Exception as e:
    print(f"Repository exists or error: {e}")

# Push dataset to hub
dataset.push_to_hub('SharleyK/TourismPackagePrediction')
print(f"✅ Dataset uploaded to {DATASET_REPO}")
