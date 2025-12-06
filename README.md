# Tourism-Package-Prediction
🚀 Wellness Tourism Package Purchase Prediction – MLOps Pipeline
by Visit With Us – Data-Driven Tourism Innovation
📌 Business Context

"Visit with Us" is a leading travel company aiming to transform customer engagement through intelligent targeting strategies. When introducing new travel packages like Wellness Tourism, manually identifying potential customers is slow, inconsistent, and error-prone.

To overcome this challenge, we implement a fully automated MLOps pipeline to:

✔ Integrate customer data
✔ Predict potential buyers before outreach
✔ Enable scalable automation using CI/CD workflows
✔ Continuously monitor and improve the model

This solution empowers marketing teams with actionable insights — boosting conversions, reducing campaign costs, and improving overall customer satisfaction.

🎯 Project Objective

As an MLOps Engineer, the mission is to:

🔹 Build a machine learning model predicting if a customer will purchase the Wellness Tourism Package
🔹 Automate the entire workflow using GitHub Actions pipelines
🔹 Deploy the model and make predictions available via Hugging Face Spaces
🔹 Ensure repeatability, scalability & continuous improvements

Key Pipeline Components:

Data ingestion & preprocessing

Model development, hyperparameter tuning & evaluation

Experiment logging & model versioning

Automated deployment using CI/CD

Docker-based containerization

Hosting frontend on Hugging Face Spaces

📊 Dataset Description

The dataset includes customer demographics and sales interaction details to determine purchase likelihood.

Target Variable
Feature	Description
ProdTaken	Purchased package? (0 = No, 1 = Yes)
Customer Attributes

Age, Gender, MaritalStatus, MonthlyIncome, CityTier, Occupation, Passport, OwnCar, PreferredPropertyStar, NumberOfTrips, etc.

Interaction Attributes

PitchSatisfactionScore, DurationOfPitch, ProductPitched, NumberOfFollowups, etc.

📌 Dataset is registered & accessed directly from HuggingFace Datasets.

🧩 Tech Stack
Layer	Tools & Frameworks
Version Control	GitHub
Workflow Automation	GitHub Actions
Model Building	Python, Scikit-Learn / XGBoost
Deployment	Docker + Hugging Face Spaces
Experiment Tracking & Registry	Hugging Face Model Hub
UI for predictions	Streamlit
🔄 MLOps Pipeline Workflow
flowchart LR
A[Data from Hugging Face Dataset] --> B[Data Cleaning & Preprocessing]
B --> C[Train-Test Split]
C --> D[Model Training & Hyperparameter Tuning]
D --> E[Model Evaluation & Logging]
E --> F[Register Best Model on HF Hub]
F --> G[Containerize Deployment]
G --> H[Deploy to Hugging Face Spaces]
H --> I[Continuous Monitoring & Automated Updates via GitHub Actions]

🧪 Model Development

✔ Multiple ML Models tested
✔ Best-performing model selected based on evaluation metrics
✔ Hyperparameters logged
✔ Model pushed to Hugging Face Model Hub

🐳 Deployment

Dockerfile created for containerization

Model and dependencies loaded from Hugging Face Model Hub

Streamlit frontend for real-time predictions

Hosted on Hugging Face Spaces

⚙️ GitHub Actions CI/CD Pipeline

Pipeline executes automatically on push to main:

Stage	Automated Task
Data Step	Load → Preprocess → Split → Upload datasets
Modeling Step	Train → Evaluate → Register best model
Deployment Step	Build docker image → Deploy Space
Monitoring Step	Rerun pipeline on updated code

Workflow file:
📌 .github/workflows/pipeline.yml

📂 Repository Structure
├── data/                    # Data loading & HF registration scripts
├── models/                  # Model-related scripts + metadata
├── app/                     # Streamlit frontend files
├── Dockerfile               # Container environment
├── requirements.txt         # Deployment dependencies
├── pipeline.yml             # GitHub Actions workflow
├── src/                     # All Python source code modules
│   ├── data_preprocess.py
│   ├── model_train.py
│   ├── evaluate.py
│   └── deploy.py
└── README.md                # Project documentation

🚀 Live Deployment & Resources
Resource	Link
📌 GitHub Repository	Add link here
🤖 Hugging Face Model	Add link here
🌐 Hugging Face Spaces App	Add link here
📘 Submission Notebook (HTML)	Add link here

Replace placeholders once deployment is complete.

📈 Evaluation Metrics

✔ Accuracy
✔ ROC-AUC
✔ Precision / Recall
✔ Confusion Matrix

A brief report summarizing feature importance and business interpretation is included inside the notebook.

🏆 Submission Requirement Checklist
Task	Status (✓/✗)
Data registered on HF dataset	
Train/Test upload back to HF	
Best model registered on HF Model Hub	
Hugging Face Space deployed	
GitHub Actions automation pipeline	
Notebook completed with insights	
✨ Future Enhancements

Real-time data refresh & monitoring

Model drift detection & auto-retraining

Enhanced feature engineering using NLP & behavioral analytics

Scalable cloud deployment

🙌 Acknowledgements

This project is completed as part of an AI/ML MLOps learning initiative.
Thanks to Visit With Us for the business dataset and challenge case.
