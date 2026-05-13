import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea


df=pd.read_csv("insurance.csv")
print("dataset: ")
print(df.head())


bins=[0,25,40,60,100]
labels=["young(0-25)","adult(25-40)","middleage(40-60)","senior(60+)"]
df["Age_grouped"]=pd.cut(df["age"],bins=bins,labels=labels)
print("grouped data")
print(df.head())


grouped_statee=df.groupby("Age_grouped",observed=True)["charges"].agg(["mean","median","max","min","std"])
print("Summary Statistics of Income (charges) grouped by Age Groups:\n")
print(grouped_statee)

mean_income_list=df.groupby("Age_grouped",observed=True)["charges"].mean().tolist()
print("mean:", mean_income_list)

plt.figure()
plt.hist(df["charges"],bins=30)
plt.title("Histogram of Insurance Charges")
plt.xlabel("charges")
plt.ylabel("frequence")
plt.show()



# column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df = pd.read_csv("iris.csv")
print(df.head())


setosa=df[df["species"]=="Iris-setosa"]
versicolor=df[df["species"]=="Iris-versicolor"]
virginica=df[df["species"]=="Iris-virginica"]


print("Setosa")
print(setosa.describe())

print("versicolor")
print(versicolor.describe())

print("virginica")
print(virginica.describe())

numeric_df=df.drop(columns=["species"])
corr=numeric_df.corr()

print(corr)
plt.figure(figsize=(6,5))
sea.heatmap(corr,annot=True)
plt.title("Correlation Heatmap - Iris Dataset")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
# plt.colorbar()
plt.show()


# insurance.csv
# age,sex,bmi,children,smoker,region,charges
# 19,female,27.9,0,yes,southwest,16884
# 25,male,28.5,1,no,southeast,3200
# 32,female,25.3,2,no,northwest,6400
# 45,male,30.1,3,yes,northeast,22000
# 52,female,29.7,1,no,southwest,12000
# 23,male,24.5,0,no,southeast,2800
# 36,female,31.2,2,yes,northwest,18000
# 41,male,26.8,1,no,northeast,9000
# 60,female,33.5,0,yes,southwest,26000
# 29,male,27.1,2,no,southeast,7000
# 48,female,32.9,3,yes,northwest,21000
# 55,male,29.4,1,no,northeast,14000
# 38,female,26.2,0,no,southwest,8500
# 27,male,23.9,1,no,southeast,4000
# 65,female,34.1,2,yes,northwest,30000


# iris.csv
# sepal_length,sepal_width,petal_length,petal_width,species
# 5.1,3.5,1.4,0.2,Iris-setosa
# 4.9,3.0,1.4,0.2,Iris-setosa
# 5.0,3.4,1.5,0.2,Iris-setosa
# 5.4,3.9,1.7,0.4,Iris-setosa
# 6.4,3.2,4.5,1.5,Iris-versicolor
# 6.9,3.1,4.9,1.5,Iris-versicolor
# 5.5,2.3,4.0,1.3,Iris-versicolor
# 6.5,2.8,4.6,1.5,Iris-versicolor
# 6.3,3.3,6.0,2.5,Iris-virginica
# 5.8,2.7,5.1,1.9,Iris-virginica
# 7.1,3.0,5.9,2.1,Iris-virginica
# 6.5,3.0,5.8,2.2,Iris-virginica
