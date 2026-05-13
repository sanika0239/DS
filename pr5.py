# ==========================================
# LOGISTIC REGRESSION - SOCIAL NETWORK ADS
# ==========================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# 2. Load Dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
print("Dataset Preview:\n", dataset.head())

# 3. Select Features & Target
X = dataset.iloc[:, [2, 3]].values   # Age, Salary
y = dataset.iloc[:, 4].values        # Purchased

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# 5. Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 6. Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. Prediction
y_pred = model.predict(X_test)

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# 9. Extract TP, FP, TN, FN
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]

print("\nTP:", TP)
print("FP:", FP)
print("TN:", TN)
print("FN:", FN)

# 10. Performance Metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
error_rate = 1 - accuracy
precision = TP / (TP + FP)
recall = TP / (TP + FN)

print("\nAccuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)

# ==========================================
# 11. CONFUSION MATRIX HEATMAP (SAVE AS PNG)
# ==========================================

plt.figure(figsize=(6, 5))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['No', 'Yes'],
            yticklabels=['No', 'Yes'])

plt.title(f"Confusion Matrix (Accuracy = {accuracy:.2f})")
plt.xlabel("Predicted")
plt.ylabel("Actual")


plt.show()

# ==========================================
# CONCLUSION
# ==========================================
print("\nConclusion: Logistic Regression model achieved high accuracy and effectively classified user purchase behavior.")

# User ID,Gender,Age,EstimatedSalary,Purchased
# 15624510,Male,19,19000,0
# 15810944,Male,35,20000,0
# 15668575,Female,26,43000,0
# 15603246,Female,27,57000,0
# 15804002,Male,19,76000,0
# 15728773,Male,27,58000,0
# 15598044,Female,27,84000,0
# 15694829,Female,32,150000,1
# 15600575,Male,25,33000,0
# 15727311,Female,35,65000,0
# 15570769,Female,26,80000,0
# 15606274,Female,26,52000,0
# 15746139,Male,20,86000,0
# 15704987,Male,32,18000,0
# 15628972,Male,18,82000,0
# 15697686,Male,29,80000,0
# 15733883,Male,47,25000,1
# 15617482,Male,45,26000,1
# 15704583,Male,46,28000,1
# 15621083,Female,48,29000,1
# 15649487,Male,45,22000,1
# 15736760,Female,47,49000,1
