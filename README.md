

***********************************************************************************************
ALL AdvanceML Lab:
******************
1. Write a program to demonstrate the working of the decision tree based ID3 algorithm. Use an appropriate dataset for building the decision tree and apply this knowledge to classify a new sample.
2. Build an Artificial Neural Network by implementing the Back propagation algorithm and test the same using appropriate data sets.
3. Write a program to implement the naïve Bayesian classifier for a sample training data set stored as a .CSV file. Compute the accuracy of the classifier, considering few test data sets.
4. Assuming a set of documents that need to be classified, use the naïve Bayesian Classifier model to perform this task. Built-in Java classes/API can be used to write the program. Calculate the accuracy, precision, and recall for your data set.
5. Write a program to construct a Bayesian network considering medical data. Use this model to demonstrate the diagnosis of heart patients using standard Heart Disease Data Set. You can use Java/Python ML library classes/API.
6. Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points. Select appropriate data set for your experiment and draw graphs.
7. Apply EM algorithm to cluster a set of data stored in a CSV file. Use the same data set for clustering using k Means algorithm. Compare the results of these two algorithms and comment on the quality of clustering. You can add Java/Python ML library classes/API in the program.
8. Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set. Print both correct and wrong predictions. Java/Python ML library classes can be used for this problem.
***********************************************************************************************

**Write a program to demonstrate the working of the decision tree based ID3 algorithm. Use an appropriate dataset for building the decision tree and apply this knowledge to classify a new sample.

That text can be formatted into a clear, structured list using Markdown.

Implementing the ID3 algorithm in Python provides a hands-on understanding of how it works. Below is a step-by-step guide to creating a decision tree using the ID3 algorithm.

Step 1: Import Necessary Libraries

Start by importing the required libraries for data handling and visualization (e.g., pandas, NumPy).

Step 2: Define Functions for Entropy and Information Gain

These core functions are necessary to select the best splitting attribute at each node.

Step 3: Build the ID3 Algorithm

This involves recursively defining the tree structure, using the Information Gain calculation to choose the root/split node.

Step 4: Apply the Algorithm to a Dataset

Execute the built algorithm on your training data to generate the final decision tree model.

Step 5: Visualize the Decision Tree

For better understanding, visualize the decision tree using libraries like Graphviz or similar tools.


**##Multi-Layer Perceptron (MLP)

The implementation of simple ANN is a Multi-Layer Perceptron (MLP) from scratch using the NumPy library to solve the XOR problem, a classic non-linear dataset. We initialize a network architecture with 2 input neurons, 2 hidden neurons, and 1 output neuron. We use the Sigmoid activation function to introduce non-linearity, which is essential for solving XOR. The training process runs for 10,000 epochs.

In every iteration, two main phases occur:

-**Forward Propagation: The data flows from the input layer to the hidden layer and then to the output layer using dot products of inputs and weights, followed by the addition of biases and the application of the activation function.

-**Backpropagation: We calculate the error (the difference between the target y and the predicted_output). Using the Chain Rule of Calculus, we compute the gradient of the error with respect to the weights. We first calculate the delta at the output layer (d_predicted_output) and propagate this error backward to find the delta at the hidden layer (d_hidden_layer). Finally, we update the weights and biases by adding the product of the gradients and the learning_rate, effectively minimizing the error over time. The final print statement demonstrates that the network has learned to output values close to 0 for inputs [0,0] and [1,1], and values close to 1 for [0,1] and [1,0].
