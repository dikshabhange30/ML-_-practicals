# Classify the email using the binary classification method. Email Spam detection has two states:
# a) Normal State – Not Spam b) Abnormal State – Spam. Use K-Nearest Neighbors and Support Vector Machine for classification. Analyze their performance.

# Cell 1 – Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Cell 2 – Load dataset
df = pd.read_csv("spam.csv", encoding="latin1")
print(df.head())

# Cell 3 – Preprocess the dataset
df = df[['v1', 'v2']]
df.columns = ['Category', 'Message']
df = df.dropna()
df = df.drop_duplicates()
print(df.head())

# Cell 4 – Convert labels into numbers
df['Category'] = df['Category'].map({
    'ham': 0,
    'spam': 1
})
print(df.head())

# Cell 5 – Separate input and output
X = df['Message']
y = df['Category']

# Cell 6 – Convert text into numbers
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(X)
print(X.shape)

# Cell 7 – Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Cell 8 – Apply KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred1 = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred1))

# Cell 9 – Apply SVM
svm = LinearSVC()
svm.fit(X_train, y_train)
y_pred2 = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred2))

# Cell 10 – Confusion Matrix for KNN
cm1 = confusion_matrix(y_test, y_pred1)
print("KNN Confusion Matrix:")
print(cm1)

# Cell 11 – Confusion Matrix for SVM
cm2 = confusion_matrix(y_test, y_pred2)
print("SVM Confusion Matrix:")
print(cm2)

# Cell 12 – Compare both models
print("KNN Accuracy:",
      accuracy_score(y_test, y_pred1) * 100)
print("SVM Accuracy:",
      accuracy_score(y_test, y_pred2) * 100)

# Cell 13 – Final result
print("KNN Confusion Matrix:")
print(cm1)
print("\nSVM Confusion Matrix:")
print(cm2)
print("\nKNN Accuracy:",
      accuracy_score(y_test, y_pred1))
print("SVM Accuracy:",
      accuracy_score(y_test, y_pred2))

# # Expected practical conclusion
# Both KNN and SVM can classify messages into spam and normal categories.
# Compare their accuracy and confusion matrices to determine which model performs better.
# SVM often performs well on text classification, but the actual result depends on the dataset and test split.





# Question	Easy answer
# 1. What is spam detection?	Identifying unwanted or fraudulent messages.
# 2. What is binary classification?	Classification into two categories.
# 3. What is KNN?	K-Nearest Neighbors is an algorithm that predicts using nearest data points.
# 4. What is SVM?	Support Vector Machine is an algorithm that separates data using a decision boundary.
# 5. What is spam?	An unwanted or suspicious message.
# 6. What is ham?	A normal or genuine message.
# 7. What is TF-IDF?	It converts text into numerical values based on word importance.
# 8. What is training data?	Data used to train the model.
# 9. What is testing data?	Data used to test the model.
# 10. What is accuracy?	Percentage of correct predictions.
# 11. What is a confusion matrix?	A table showing correct and incorrect predictions.
# 12. What is K in KNN?	Number of nearest neighbours.
# 13. What is SVM used for?	Classification and regression.
# 14. What is supervised learning?	Learning using labelled data.
# 15. Which algorithm is better for spam detection?	SVM often performs well for text classification, but we compare actual accuracy to decide.










































