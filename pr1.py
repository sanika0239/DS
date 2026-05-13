
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Titanic_Sample_Dataset.csv")
print("Dataset Loaded Successfully!\n")

print("First 5 Rows of Dataset:\n")
print(df.head())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Dataset Dimensions ---")
print(df.shape)

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Dataset Information ---")
print(df.info())

print("\n--- Data Types Before Conversion ---")
print(df.dtypes)

# Fill missing Age values with median
df['Age'] = df['Age'].fillna(df['Age'].median())

print("\n--- Missing Values After Handling ---")
print(df.isnull().sum())

df['Sex'] = df['Sex'].astype('category')

df['Embarked'] = df['Embarked'].astype('category')

df['Pclass'] = df['Pclass'].astype('category')

print("\n--- Data Types After Conversion ---")
print(df.dtypes)


print("\n--- Before Encoding ---")
print(df.head())

label_encoder = LabelEncoder()

# Encoding Sex
df['Sex'] = label_encoder.fit_transform(df['Sex'])

# Encoding Embarked
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])

# Encoding Pclass
df['Pclass'] = df['Pclass'].cat.codes

print("\n--- After Encoding ---")
print(df.head())


scaler = StandardScaler()

numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\n--- After Normalization ---")
print(df.head())

print("\nProgram Executed Successfully!")



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