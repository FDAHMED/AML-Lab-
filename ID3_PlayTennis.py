import pandas as pd 
import numpy as np #
from math import log2
from graphviz import Digraph

def calculate_entropy(data):
    labels = data.iloc[:, -1].value_counts()
    total = len(data)
    entropy = -sum((count / total) * log2(count / total) for count in labels)
    return entropy

def calculate_information_gain(data, attribute):
    total_entropy = calculate_entropy(data)
    values = data[attribute].unique()
    weighted_entropy = 0
    
    for value in values:
        subset = data[data[attribute] == value]
        weighted_entropy += (len(subset) / len(data)) * calculate_entropy(subset)
        
    return total_entropy - weighted_entropy

def id3(data, features):
    if len(data.iloc[:, -1].unique()) == 1:
        return data.iloc[0, -1]
    
    if len(features) == 0:
        return data.iloc[:, -1].mode()[0]
    
    gains = {feature: calculate_information_gain(data, feature) for feature in features}
    best_feature = max(gains, key=gains.get)
    
    tree = {best_feature: {}}
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        remaining_features = [feat for feat in features if feat != best_feature]
        tree[best_feature][value] = id3(subset, remaining_features)
        
    return tree

def visualize_tree(tree, parent=None, graph=None):
    if graph is None:
        graph = Digraph()
        
    for key, value in tree.items():
        if isinstance(value, dict):
            graph.node(key, key)
            for sub_key in value:
                graph.edge(key, sub_key)
                visualize_tree({sub_key: value[sub_key]}, key, graph)
        else:
            graph.node(value, value)
            graph.edge(parent, value)
    return graph

if __name__ == "__main__":
    # Sample dataset
    data = pd.DataFrame({
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Mild', 'Mild', 'Hot', 'Mild', 'Mild', 'Hot', 'Cool', 'Cool'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Windy': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
        'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    })
    
    features = list(data.columns[:-1])
    decision_tree = id3(data, features)
    print(decision_tree)

    #visualize_tree(decision_tree).view()