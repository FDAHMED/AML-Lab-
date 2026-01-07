
#Task 5: Bayesian network to demonstrate the diagnosis of heart patients.


# 1. Install pgmpy Uncomment the line below if pgmpy is not installed
#!pip install pgmpy

import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 2. Load the UCI Cleveland Heart Disease Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']
df = pd.read_csv(url, names=names)

# 3. Data Cleaning & Preprocessing
df = df.replace('?', np.nan).dropna()
df = df.apply(pd.to_numeric)

# Binarize Target: 0 = No Disease, 1 = Disease
df['heartdisease'] = df['heartdisease'].apply(lambda x: 1 if x > 0 else 0)

# 4. DISCRETIZATION (Crucial for Discrete Bayesian Networks)
# We convert continuous ranges into discrete categories (0, 1, 2)
df['age'] = pd.cut(df['age'], bins=[0, 45, 60, 100], labels=[0, 1, 2]) # Young, Mid, Old
df['trestbps'] = pd.cut(df['trestbps'], bins=[0, 120, 140, 300], labels=[0, 1, 2]) # Normal, High, Very High
df['chol'] = pd.cut(df['chol'], bins=[0, 200, 240, 600], labels=[0, 1, 2]) # Desirable, Borderline, High

# 5. Define Network Structure (DAG)
model = BayesianNetwork([
    ('age', 'trestbps'),
    ('age', 'fbs'),
    ('sex', 'heartdisease'),
    ('trestbps', 'heartdisease'),
    ('cp', 'heartdisease'),
    ('exang', 'heartdisease'),
    ('heartdisease', 'chol')
])

# 6. Parameter Learning (Generating CPTs)
print("Learning Conditional Probability Tables (CPTs)...")
model.fit(df, estimator=MaximumLikelihoodEstimator)

# 7. Diagnosis Engine
infer = VariableElimination(model)

# --- Sample Test Diagnosis ---
print("\n--- Diagnostic Test ---")
# Testing a patient: Old (age=2), High BP (trestbps=1), Asymptomatic CP (cp=4)
result = infer.query(variables=['heartdisease'], 
                     evidence={'age': 2, 'trestbps': 1, 'cp': 4})
print(result)

# 8. Calculate Metrics
# Predicting for the first 100 rows to calculate accuracy
y_true = df['heartdisease'][:100].values
y_pred = []

for _, row in df[:100].iterrows():
    p = infer.map_query(variables=['heartdisease'], 
                        evidence={'age': row['age'], 'cp': row['cp'], 'sex': row['sex'], 'exang': row['exang']},
                        show_progress=False)
    y_pred.append(p['heartdisease'])

print("\n--- Performance Metrics ---")
print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall:    {recall_score(y_true, y_pred):.4f}")