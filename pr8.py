# =========================================================
# Titanic Dataset Analysis using Seaborn
# =========================================================

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# =========================================================
# Load Titanic Dataset
# =========================================================

titanic = pd.read_csv('Titanic_Sample_Dataset.csv')

# Display first 5 rows
print(titanic.head())

# Dataset Information
print("\nDataset Shape:")
print(titanic.shape)

print("\nDataset Info:")
print(titanic.info())

# =========================================================
# Finding Patterns in Titanic Dataset
# =========================================================

# 1. Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=titanic)

plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")

plt.show()

# =========================================================
# 2. Survival based on Gender
# =========================================================

plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=titanic)

plt.title("Survival Based on Gender")
plt.xlabel("Gender")
plt.ylabel("Count")

plt.show()

# =========================================================
# 3. Survival based on Passenger Class
# =========================================================

plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=titanic)

plt.title("Survival Based on Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")

plt.show()

# =========================================================
# 4. Age Distribution
# =========================================================

plt.figure(figsize=(8,5))
sns.histplot(titanic['Age'].dropna(), bins=30, kde=True)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()

# =========================================================
# 5. Fare Distribution (Required Question)
# =========================================================

plt.figure(figsize=(8,5))

sns.histplot(titanic['Fare'], bins=30, kde=True)

plt.title("Fare Distribution of Titanic Passengers")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.show()


print("\nOverall Inference:")
print("1. Women and first-class passengers had higher survival rates.")
print("2. Most passengers belonged to lower fare categories.")
print("3. Age, gender, passenger class, and fare influenced survival chances.")



# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Fare,Embarked
# 1,0,3,John Smith,male,22,1,0,7.25,S
# 2,1,1,Anna Brown,female,38,1,0,71.28,C
# 3,1,3,Emily Davis,female,26,0,0,7.92,S
# 4,1,1,Michael Wilson,male,35,1,0,53.10,S
# 5,0,3,David Miller,male,35,0,0,8.05,S
# 6,0,3,Sophia Taylor,female,,0,0,8.45,Q
# 7,0,1,James Anderson,male,54,0,0,51.86,S
# 8,0,3,Olivia Thomas,female,2,3,1,21.07,S
# 9,1,3,Liam Jackson,male,27,0,2,11.13,S
# 10,1,2,Ava White,female,14,1,0,30.07,C
# 11,1,3,Noah Harris,male,,0,0,7.89,Q
# 12,0,2,Isabella Martin,female,45,1,1,26.00,S
# 13,1,1,William Thompson,male,58,0,0,35.50,S
# 14,0,3,Mia Garcia,female,20,0,0,7.22,C
# 15,1,2,Ethan Martinez,male,31,0,0,13.00,S