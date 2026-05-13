# =========================================================
# Data Visualization II - Titanic Dataset
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

# =========================================================
# Box Plot: Age Distribution with Gender and Survival
# =========================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    x='Sex',
    y='Age',
    hue='Survived',
    data=titanic
)

plt.title("Age Distribution by Gender and Survival")
plt.xlabel("Gender")
plt.ylabel("Age")

plt.legend(title="Survived", labels=["No", "Yes"])

plt.show()


print("\nObservations / Inference:\n")

print("1. Female passengers had a higher survival rate than male passengers.")

print("2. Younger passengers, especially children, had better chances of survival.")

print("3. Male passengers show a wider age distribution compared to females.")

print("4. Many older male passengers did not survive.")

print("5. The median age of survivors is slightly lower than non-survivors.")

print("6. Presence of outliers indicates some passengers were very old.")

print("7. The graph suggests that women and children were given priority during rescue operations.")


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