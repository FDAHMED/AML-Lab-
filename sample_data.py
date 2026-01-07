#Task 3: Create a sample dataset and save it as a CSV file.

import pandas as pd

# Creating a sample dataset
data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female',
               'Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Age': [19, 35, 26, 27, 19, 27, 27, 32, 25, 35, 
            26, 26, 20, 32, 18, 29, 47, 45, 46, 48],
    'EstimatedSalary': [19000, 20000, 43000, 57000, 76000, 58000, 84000, 150000, 33000, 65000,
                        80000, 52000, 86000, 18000, 82000, 80000, 25000, 26000, 28000, 29000],
    'Purchased': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
                  0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('sample_data.csv', index=False)
print("File 'sample_data.csv' has been created successfully!")