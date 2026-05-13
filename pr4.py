import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv("home_prices_simple.csv")

print("\n===== DATASET =====\n")
print(df)

# Features and target
X = df[['Rooms']]
y = df['Price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# -----------------------------
# PRINT ACTUAL VS PREDICTED IN TERMINAL
# -----------------------------
print("\n===== ACTUAL vs PREDICTED =====\n")

comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
print("\nActual vs Predicted:\n", comparison.head())

# Accuracy

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nRMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# -----------------------------
# GRAPH (ONLY ONE)
# -----------------------------
plt.figure(figsize=(6,4))
sns.regplot(x=df["Rooms"], y=df["Price"], line_kws={"color": "red"})
plt.title("Rooms vs Price Relationship")
plt.show()


# Rooms,Distance,Age,Tax,Price
# 2,5,20,200,15
# 3,7,15,220,18
# 4,10,30,250,22
# 2,4,25,180,14
# 3,6,18,210,17
# 5,12,10,300,30
# 1,3,40,150,10
# 4,9,22,240,24
# 3,8,28,230,20
# 2,5,35,190,16
# 5,11,12,320,32
# 4,10,18,260,26
# 3,7,24,220,19
# 2,4,30,200,15
# 5,13,8,350,35
# 4,9,20,250,25
# 3,6,26,210,18
# 2,5,32,190,14
# 4,10,15,270,27
# 3,8,22,225,21