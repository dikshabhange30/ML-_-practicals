# Implement the K-Nearest Neighbors algorithm on the diabetes.csv dataset. Compute confusion matrix, accuracy, error rate, precision and
# recall on the given dataset.
# KNN stands for K-Nearest Neighbors.
# It checks the nearest similar data points and predicts the class based on majority voting.
# For example:
# Suppose K = 5.
# Out of the 5 nearest patients:
# - 3 have diabetes.
# - 2 don't have diabetes.
# The model predicts Diabetes (1).

# Cell 1 – Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Cell 2 – Load dataset
df = pd.read_csv("diabetes.csv")
print(df.head())
print(df.shape)

# Cell 3 – Preprocess the dataset
df = df.dropna()
df = df.drop_duplicates()
print(df.shape)

# Cell 4 – Separate input and output
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Cell 5 – Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Cell 6 – Apply feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Cell 7 – Apply KNN algorithm
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(y_pred)

# Cell 8 – Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)



# Term	Meaning
# True Positive (TP)	Correctly predicted diabetes
# True Negative (TN)	Correctly predicted no diabetes
# False Positive (FP)	Predicted diabetes, but actually no diabetes
# False Negative (FN)	Predicted no diabetes, but actually diabetes

# Cell 9 – Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Cell 10 – Calculate error rate
error = 1 - accuracy
print("Error Rate:", error)

# Cell 11 – Calculate precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Cell 12 – Calculate recall
# Recall tells us how many actual diabetic patients were correctly identified.
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Cell 13 – Display all results together
print("Confusion Matrix:")
print(cm)
print("Accuracy:", accuracy)
print("Error Rate:", error)
print("Precision:", precision)
print("Recall:", recall)


# 5. Important formulas for viva
# Metric	Formula
# Accuracy	(TP + TN) / (TP + TN + FP + FN)
# Error Rate	1 − Accuracy
# Precision	TP / (TP + FP)
# Recall	TP / (TP + FN)


# Q1. What is KNN?
# KNN stands for K-Nearest Neighbors. It is a supervised machine learning algorithm used for classification and regression.

# Q2. What is K in KNN?
# K represents the number of nearest neighbours considered for prediction.

# Q3. Why do we use KNN?
# To classify new data based on its nearest similar data points.

# Q4. What is classification?
# Classification is used to predict categories or classes, such as diabetic or non-diabetic.

# Q5. What is the diabetes dataset?
# It contains patient information such as glucose, blood pressure, insulin, BMI and age.

# Q6. What is StandardScaler?
# It scales numerical data to a similar range.

# Q7. Why do we use train-test split?
# To train the model and test its performance on unseen data.

# Q8. What is a confusion matrix?
# It is a table that shows correct and incorrect predictions.

# Q9. What is accuracy?
# It measures the percentage of total correct predictions.

# Q10. What is precision?
# It measures how many positive predictions are actually positive.

# Q11. What is recall?
# It measures how many actual positive cases are correctly identified.

# Q12. What is error rate?
# It measures the percentage of incorrect predictions.

# Q13. What happens if K = 1?
# The model considers only the nearest one neighbour.

# Q14. Why is scaling important in KNN?
# Because KNN calculates distances, and features with larger numerical ranges can dominate the distance calculation.

# Q15. Is KNN supervised or unsupervised?
# KNN is a supervised machine learning algorithm
