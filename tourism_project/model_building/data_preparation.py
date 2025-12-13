

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login
import os
from huggingface_hub import HfApi, login

# Login to Hugging Face
# login(token=os.environ.get('HF_TOKEN'))
api = HfApi(token=os.getenv("HF_TOKEN"))

DATASET_REPO = "SharleyK/TourismPackagePrediction"

# Load dataset from Hugging Face
print("Loading dataset from Hugging Face...")
dataset = load_dataset('""" + DATASET_REPO + """', split='train')
df = dataset.to_pandas()
print(f"✅ Dataset loaded: {df.shape}")

# Display basic information
print("\\nDataset Info:")
print(df.info())
print("\\nFirst few rows:")
print(df.head())
print("\\nMissing values:")
print(df.isnull().sum())
print("\\nTarget variable distribution:")
print(df['ProdTaken'].value_counts())

# Data Cleaning
print("\\n" + "="*80)
print("DATA CLEANING")
print("="*80)

# Remove unnecessary columns
cols_to_drop = ['Unnamed: 0', 'CustomerID'] if 'Unnamed: 0' in df.columns else ['CustomerID']
df = df.drop(columns=cols_to_drop, errors='ignore')
print(f"✅ Removed unnecessary columns: {cols_to_drop}")

# Handle missing values
print("\\nHandling missing values...")
# Fill numerical columns with median
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Fill categorical columns with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

print(f"✅ Missing values handled")
print("\\nMissing values after cleaning:")
print(df.isnull().sum().sum())

# Encode categorical variables
print("\\nEncoding categorical variables...")
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  ✓ Encoded: {col}")

# Save label encoders
import joblib
os.makedirs('tourism_project/model_building/encoders', exist_ok=True)
for col, le in label_encoders.items():
    joblib.dump(le, f'tourism_project/model_building/encoders/{col}_encoder.pkl')
print("✅ Label encoders saved")

# Split features and target
X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']

print(f"\\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution:\\n{y.value_counts()}")

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\\nTraining set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Create train and test dataframes
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Save locally
train_df.to_csv('tourism_project/data/train.csv', index=False)
test_df.to_csv('tourism_project/data/test.csv', index=False)
print("✅ Train and test sets saved locally")

# Upload to Hugging Face
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

dataset_dict.push_to_hub('""" + DATASET_REPO + """')
print(f"✅ Train and test sets uploaded to {DATASET_REPO}")

print("\\n✅ DATA PREPARATION COMPLETED SUCCESSFULLY!")


# # Save data preparation script
# with open('tourism_project/model_building/data_preparation.py', 'w') as f:
#     f.write(data_prep_script)

print("✅ Data preparation script created: tourism_project/model_building/data_preparation.py")

# # Execute data preparation
# # Uncomment the line below after data registration is complete
# !python tourism_project/model_building/data_preparation.py
