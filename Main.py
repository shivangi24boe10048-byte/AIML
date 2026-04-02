# main.py
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("Loading Breast Cancer dataset...")
    # 1. Load the built-in dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target # 0 = malignant, 1 = benign

    # 2. Split the data into training (80%) and testing (20%) sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Initialize and train the Random Forest Classifier
    print("Training the Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # 4. Make predictions on the test set
    print("Making predictions...")
    y_pred = rf_model.predict(X_test)

    # 5. Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("\n" + "="*40)
    print("MODEL EVALUATION RESULTS")
    print("="*40)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Malignant (0)', 'Benign (1)']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
