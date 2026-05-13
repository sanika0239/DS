import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Academic_Performance_Dataset.csv")

print("\nFirst 5 Records:\n")
print(df.head())

print("\nMissing Values in Each Column:\n")
print(df.isnull().sum())


# 1. Gender (Categorical → Mode Imputation)
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)

# 2. Age (Numeric → Median Imputation)
df["Age"].fillna(df["Age"].median(), inplace=True)

# 3. Scores & Study Hours (Numeric → Mean Imputation)
numeric_cols = ["Math_Score","Science_Score",
                "English_Score","Attendance_Percentage",
                "Study_Hours"]


for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

df["Gender"] = df["Gender"].replace({"M": "Male", "F": "Female"})


df.loc[df["Attendance_Percentage"] > 100, "Attendance_Percentage"] = 100
df.loc[df["Attendance_Percentage"] < 0, "Attendance_Percentage"] = 0

df.loc[df["Age"] <= 0, "Age"] = df["Age"].median()

print("\nData cleaned for inconsistencies.")



plt.figure(figsize=(10,6))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplot before Outlier Treatment")
plt.show()


def treat_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    return np.where(column > upper, upper,
           np.where(column < lower, lower, column))

for col in numeric_cols:
    df[col] = treat_outliers_iqr(df[col])

print("\nOutliers treated using IQR Capping Method.")

plt.figure(figsize=(10,6))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplot After Outlier Treatment")
plt.show()



# Check skewness before transformation
print("\nSkewness Before Transformation:")
print(df["Study_Hours"].skew())



# Apply Log Transformation to reduce skewness
df["Study_Hours_Log"] = np.log1p(df["Study_Hours"])

print("\nSkewness After Log Transformation:")
print(df["Study_Hours_Log"].skew())


# Compare distributions
fig, ax = plt.subplots(1,2, figsize=(12,5))

sns.histplot(df["Study_Hours"], kde=True, ax=ax[0])
ax[0].set_title("Before Log Transformation")

sns.histplot(df["Study_Hours_Log"], kde=True, ax=ax[1])
ax[1].set_title("After Log Transformation")

plt.show()

print(df.head())



# Student_ID,Student_Name,Gender,Age,Math_Score,Science_Score,English_Score,Attendance_Percentage,Study_Hours
# 1,Aarav Sharma,Male,16,78,72,81,88,4
# 2,Diya Patel,Female,17,,60,74,92,
# 3,Rohan Verma,Male,18,88,75,,95,6
# 4,Ananya Singh,Female,16,54,58,69,80,2
# 5,Vihaan Gupta,,,91,84,89,97,7
# 6,Ishita Mehta,Female,15,73,,78,85,4
# 7,Arjun Nair,Male,19,60,55,68,75,2
# 8,Kavya Reddy,Female,,82,77,83,90,5
# 9,Aditya Joshi,Male,17,69,64,72,,3
# 10,Meera Iyer,Female,16,95,88,91,98,8
# 11,Siddharth Rao,Male,18,58,62,70,78,
# 12,Prisha Kapoor,Female,17,,74,80,89,4
# 13,Yash Malhotra,Male,16,84,79,86,93,6
# 14,Sneha Desai,Female,19,67,65,73,82,3
# 15,Karan Shah,Male,17,72,68,75,87,4
# 16,Riya Choudhary,Female,18,89,85,90,96,7
# 17,Manav Kulkarni,Male,16,61,,66,76,2
# 18,Anvi Bansal,Female,17,79,73,82,91,5