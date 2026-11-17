# 1)  Predict the price of the Uber ride from a given pickup point to the agreed drop-off location. Perform the following tasks: 1. Pre-process the dataset. 2. Identify outliers. 3. Check the correlation. 
# 4. Implement linear regression and random forest regression models. 5. Evaluate the models and compare their respective scores like R2, RMSE, etc. 
# first cell : Cell 1 – Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
# Cell 2 – Load dataset
df = pd.read_csv("uber.csv")
print(df.head())
print(df.shape)
# Cell 3 – Preprocess the dataset
df = df.dropna()
df = df.drop_duplicates()
df = df[df['fare_amount'] > 0]
df = df[df['passenger_count'] > 0]
print(df.head())
print(df.shape)
# Cell 4 – Calculate distance
def distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians,
                                  [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2

    return 2 * R * np.arcsin(np.sqrt(a))
df['distance'] = distance(
    df['pickup_latitude'],
    df['pickup_longitude'],
    df['dropoff_latitude'],
    df['dropoff_longitude']
)
print(df.head())
# Cell 5 – Identify outliers
sns.boxplot(x=df['fare_amount'])
plt.show()
# Remove extreme fare values using the IQR method:
Q1 = df['fare_amount'].quantile(0.25)
Q3 = df['fare_amount'].quantile(0.75)

IQR = Q3 - Q1

df = df[
    (df['fare_amount'] >= Q1 - 1.5 * IQR) &
    (df['fare_amount'] <= Q3 + 1.5 * IQR)
]
print(df.shape)
# Cell 6 – Check correlation
corr = df[['fare_amount', 'distance', 'passenger_count']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
# Cell 7 – Select input and output
X = df[['distance', 'passenger_count']]
y = df['fare_amount']
# Cell 8 – Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Cell 9 – Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred1 = lr.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred1))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred1)))

# Cell 10 – Random Forest Regression
rf = RandomForestRegressor(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)
y_pred2 = rf.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred2))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred2)))

# Cell 11 – Compare both models
print("Linear Regression")
print("R2:", r2_score(y_test, y_pred1))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred1)))

print("\nRandom Forest Regression")
print("R2:", r2_score(y_test, y_pred2))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred2)))
# 6. If libraries are not installed
pip install pandas numpy matplotlib seaborn scikit-learn






















