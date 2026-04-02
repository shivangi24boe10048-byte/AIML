# AIML
# Medical Diagnostic Assistant: Breast Cancer Classifier

## Overview
This Bring Your Own Project (BYOP) is a machine learning tool designed to classify breast cancer tumors as either malignant or benign. It uses digitized image features of fine needle aspirates (FNA) of breast masses to make predictions. This project demonstrates the application of supervised learning algorithms to assist in critical real-world healthcare decisions.

## The Problem
Early and accurate detection of breast cancer is vital for patient survival. However, human fatigue and the sheer volume of medical imaging can lead to diagnostic errors. This machine learning model acts as a secondary verification tool for medical professionals to cross-check their diagnoses.

## Technologies & Libraries Used
* **Python**: Core programming language.
* **Scikit-Learn**: Used for the dataset, model training (Random Forest), and evaluation metrics.
* **Pandas**: Used for data structuring and manipulation.

## Setup and Installation
To run this project locally, follow these steps:

1. **Clone the repository:**
   `git clone <your-github-repo-link>`
2. **Navigate to the project directory:**
   `cd BYOP_BreastCancer_ML`
3. **Install the required dependencies:**
   `pip install -r requirements.txt`
4. **Run the script:**
   `python main.py`

## How It Works
1. **Data Loading:** The script imports the standard Wisconsin Breast Cancer dataset directly from `sklearn.datasets`.
2. **Data Splitting:** It splits the dataset into an 80% training set and a 20% testing set.
3. **Model Training:** It trains a `RandomForestClassifier` on the training data.
4. **Evaluation:** It tests the model against the unseen 20% testing set and outputs the Accuracy, a full Classification Report (Precision, Recall, F1-Score), and a Confusion Matrix.
