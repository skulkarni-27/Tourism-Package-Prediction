from huggingface_hub import HfApi, login
import os

# Login to Hugging Face
login(token=os.environ.get('HF_TOKEN'))
api = HfApi()

# Create space
SPACE_NAME = 'SharleyK/tourism-package-app'

try:
    api.create_repo(
        repo_id=SPACE_NAME,
        repo_type='space',
        space_sdk='streamlit',
        exist_ok=True
    )
    print(f"✅ Space created: {SPACE_NAME}")
except Exception as e:
    print(f"Space exists or error: {e}")

# Upload files
files_to_upload = [
    ('tourism_project/deployment/app.py', 'app.py'),
    ('tourism_project/deployment/requirements.txt', 'requirements.txt'),
    ('tourism_project/deployment/Dockerfile', 'Dockerfile')
]

for local_path, repo_path in files_to_upload:
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=SPACE_NAME,
            repo_type='space'
        )
        print(f"✅ Uploaded: {repo_path}")
    except Exception as e:
        print(f"Error uploading {repo_path}: {e}")

print(f"\n✅ Deployment complete!")
print(f"View your app at: https://huggingface.co/spaces/{SPACE_NAME}")
