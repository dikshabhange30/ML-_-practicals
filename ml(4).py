# # Given a bank customer, build a neural network-based classifier that can determine whether they will leave or not in the next 6 months. Dataset Description:
# The case study is from an open-source dataset from Kaggle. The dataset contains 10,000 sample points with 14 distinct features such as Customer ID, 
# Credit Score, Geography, Gender, Age, Tenure, Balance, etc. Perform the following steps: 1. Read the dataset.
# 2. Distinguish the feature and target set and divide the data set into training and test sets. 3. Normalize the train and test data. 
# 4. Initialize and build the model. Identify the points of improvement and implement the same. 5. Print the accuracy score and confusion matrix (5 points).

# A Neural Network is a Machine Learning model inspired by the human brain. It learns patterns from data and makes predictions.

# Cell 1 – Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Cell 2 – Load dataset
df = pd.read_csv("Churn_Modelling.csv")
print(df.head())
print(df.shape)

# Cell 3 – Remove unnecessary columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
print(df.head())

# Cell 4 – Separate features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Cell 5 – Convert text into numbers
X = pd.get_dummies(X, drop_first=True)
print(X.head())

# Cell 6 – Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(X_train.shape)
print(X_test.shape)

# Cell 7 – Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Cell 8 – Build the basic Neural Network
model = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=300,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Cell 9 – Improved Neural Network
model2 = MLPClassifier(
    hidden_layer_sizes=(20, 10),
    max_iter=500,
    early_stopping=True,
    random_state=42
)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print("Improved Accuracy:", accuracy_score(y_test, y_pred2))

# Cell 10 – Confusion Matrix
cm = confusion_matrix(y_test, y_pred2)
print("Confusion Matrix:")
print(cm)

# Cell 11 – Final accuracy
accuracy = accuracy_score(y_test, y_pred2)
print("Final Accuracy:", accuracy * 100, "%")

# Cell 12 – Compare both models
print("Basic Model Accuracy:",
      accuracy_score(y_test, y_pred) * 100)
print("Improved Model Accuracy:",
      accuracy_score(y_test, y_pred2) * 100)



# Question	Easy answer
# 1. What is a Neural Network?	It is a machine learning model inspired by the human brain.
# 2. What is customer churn?	When a customer leaves or stops using a company's services.
# 3. What is ANN?	ANN stands for Artificial Neural Network.
# 4. What is MLP?	MLP stands for Multi-Layer Perceptron.
# 5. What is a neuron?	A basic processing unit of a neural network.
# 6. What is a hidden layer?	A layer between input and output layers.
# 7. What is an activation function?	It helps the neural network learn complex patterns.
# 8. What is normalization?	Bringing numerical values to a similar scale.
# 9. What is an epoch?	One complete pass through the training data.
# 10. What is a confusion matrix?	A table showing correct and incorrect predictions.
# 11. What is accuracy?	Percentage of correct predictions.
# 12. What is overfitting?	When a model performs well on training data but poorly on new data.
# 13. What is early stopping?	It stops training when validation performance stops improving.
# 14. What is classification?	Predicting categories such as Yes or No.
# 15. What is the target column?	Exited.

















































