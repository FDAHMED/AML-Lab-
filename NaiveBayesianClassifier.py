import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

def run_naive_bayes(csv_file):
    # 1. Load the dataset
    try:
        data = pd.read_csv(csv_file)
        print(f"--- Dataset Loaded: {csv_file} ---")
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return

    # 2. Preprocessing
    # Assuming the last column is the target/label
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target

    # Handle categorical data (if any) by converting to dummy variables
    X = pd.get_dummies(X)

    # 3. Split the data into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Initialize and Train the Naïve Bayes Classifier
    # We use GaussianNB assuming features follow a normal distribution
    model = GaussianNB()
    model.fit(X_train, y_train)

    # 5. Make Predictions
    y_pred = model.predict(X_test)

    # 6. Compute and Display Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # Example: Displaying a few test cases
    comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print("\nSample Test Results (First 5):")
    print(comparison.head())

# To run this, replace 'your_data.csv' with your actual filename
run_naive_bayes('sample_data.csv')