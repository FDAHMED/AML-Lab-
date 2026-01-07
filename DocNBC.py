# Task 4: Document/Text Classifier model using  Naive Bayesian Classifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics

# 1. Define Sample Training Data
# Categories: 0 = Tech, 1 = Politics
train_docs = [
    "The new smartphone has a high resolution screen and fast processor", # Tech
    "The software update fixed several bugs in the operating system",     # Tech
    "Artificial intelligence is changing the way we write code",          # Tech
    "The latest graphics card supports 4k gaming at high frame rates",    # Tech
    "The senator signed the new healthcare bill into law today",          # Politics
    "Voters are heading to the polls for the local election",             # Politics
    "The prime minister addressed the parliament regarding the budget",   # Politics
    "Diplomatic talks between the two nations failed to reach a treaty"    # Politics
]
train_labels = [0, 0, 0, 0, 1, 1, 1, 1]

# 2. Define Sample Test Data
test_docs = [
    "I need to upgrade my computer processor for better performance",    # Tech (Expected: 0)
    "The government passed a new law regarding taxes",                   # Politics (Expected: 1)
]
test_labels = [0, 1]

# 3. Build and Train the Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_clf.fit(train_docs, train_labels)

# 4. Predict and Evaluate
predicted = text_clf.predict(test_docs)

print(f"Predictions: {predicted}")
print(f"Accuracy: {metrics.accuracy_score(test_labels, predicted) * 100}%")

# 5. Try a custom sentence
new_sentence = ["The president is giving a speech about digital technology"]
prediction = text_clf.predict(new_sentence)
category = "Tech" if prediction[0] == 0 else "Politics"
print(f"\nCustom Test: '{new_sentence[0]}' -> Predicted Category: {category}")


#Important Note on Small Datasets
## In the "Custom Test" in the code above ("The president is giving a speech about digital technology"), 
## the model might struggle. It contains words from both categories ("President"[Politics] vs "Technology"[Tech]). 
## With very small datasets, the model is highly sensitive to which words it has seen before.
## To get better results, you would typically need dozens or hundreds of examples per category.