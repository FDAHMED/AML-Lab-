#Task 8 : Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set. Print both correct and wrong predictions. Java/Python ML library classes can be used for this problem.
#Change the k value[s] and Test data percentage to get both the correct and wrong predictions.

#In the k-Nearest Neighbours algorithm k is just a number that tells the algorithm how many nearby points or neighbors to look at when it makes a decision.

#(Learn about iris data set here : [https://www.kaggle.com/datasets/vikrishnan/iris-dataset/data])

#Python implementation

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load the dataset
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names

# 2. Split into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Initialize and train the k-NN model (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 4. Make predictions
predictions = knn.predict(X_test)

# 5. Output Results
print(f"{'Actual':<15} | {'Predicted':<15} | {'Status'}")
print("-" * 45)

correct_count = 0
wrong_count = 0

for actual, predicted in zip(y_test, predictions):
    actual_name = class_names[actual]
    predicted_name = class_names[predicted]
    
    if actual == predicted:
        status = "✅ Correct"
        correct_count += 1
    else:
        status = "❌ WRONG"
        wrong_count += 1
        
    print(f"{actual_name:<15} | {predicted_name:<15} | {status}")

# Final Summary
accuracy = accuracy_score(y_test, predictions)
print("-" * 45)
print(f"Total Correct: {correct_count}")
print(f"Total Wrong: {wrong_count}")
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
