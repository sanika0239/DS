# ==========================================
# IRIS DATASET - EXPLORATORY DATA ANALYSIS
# ==========================================

# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset
df = pd.read_csv('iris.csv')

# Display first few rows
print("First 5 rows of dataset:")
print(df.head())

# ==========================================
# 1. Features and Their Types
# ==========================================
print("\nFeature Types:")
print(df.dtypes)

print("\nFeature Description:")
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"{col} → Categorical (Nominal)")
    else:
        print(f"{col} → Numeric (Continuous)")

# ==========================================
# 2. Histograms for Each Feature
# ==========================================
print("\nGenerating Histograms...")

df.hist(figsize=(10,8))
plt.suptitle("Histograms of Iris Features")
plt.show()

# ==========================================
# 3. Boxplots for Each Feature
# ==========================================
print("\nGenerating Boxplots...")

for col in df.columns[:-1]:  # exclude species
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# ==========================================
# 4. Compare Distributions & Identify Outliers
# ==========================================
print("\nStatistical Summary:")
print(df.describe())

print("\nOutlier Detection using IQR Method:")

for col in df.columns[:-1]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    print(f"\n{col}:")
    print(f"Lower Bound = {lower_bound}, Upper Bound = {upper_bound}")
    print(f"Number of Outliers = {len(outliers)}")


sns.pairplot(df, hue='species')

plt.show()

# ---------------------------------------------------------
# Inference / Observations
# ---------------------------------------------------------

print("\nInference from Visualizations:\n")

print("1. Sepal Length:")
print("- Approximately normally distributed.")
print("- Moderate spread.")
print("- Few/no major outliers.")

print("\n2. Sepal Width:")
print("- Wider distribution.")
print("- Some outliers visible in boxplot.")

print("\n3. Petal Length:")
print("- Clearly separates species.")
print("- Setosa has very small petal length.")

print("\n4. Petal Width:")
print("- Strong distinction between species.")
print("- Useful feature for classification.")

print("\nPairplot Observations:")
print("- Setosa species is clearly separable.")
print("- Versicolor and Virginica overlap slightly.")
print("- Petal features provide better separation than sepal features.")

print("\nOverall Conclusion:")
print("- Sepal width contains some outliers.")
print("- Petal length and petal width are the most important features.")
print("- Pairplot helps compare relationships among all features.")




# sepal_length,sepal_width,petal_length,petal_width,species
# 5.1,3.5,1.4,0.2,setosa
# 4.9,3.0,1.4,0.2,setosa
# 5.0,3.4,1.5,0.2,setosa
# 5.4,3.9,1.7,0.4,setosa
# 5.8,4.0,1.2,0.2,setosa
# 6.4,3.2,4.5,1.5,versicolor
# 6.9,3.1,4.9,1.5,versicolor
# 5.5,2.3,4.0,1.3,versicolor
# 6.5,2.8,4.6,1.5,versicolor
# 5.7,2.8,4.5,1.3,versicolor
# 6.3,3.3,6.0,2.5,virginica
# 5.8,2.7,5.1,1.9,virginica
# 7.1,3.0,5.9,2.1,virginica
# 6.5,3.0,5.8,2.2,virginica
# 6.7,3.1,5.6,2.4,virginica