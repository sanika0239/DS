import pandas as pd
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

df=pd.read_csv("iris.csv")
X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values
X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.3,random_state=42, stratify=y
)

model=GaussianNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
print(cm)

accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,average="macro")
recall=recall_score(y_test,y_pred,average="macro")
error=1-accuracy

print("\nAccuracy:", accuracy)
print("Error Rate:", error)
print("Precision:", precision)
print("Recall:", recall)


for i in range(len(cm)):
    tp=cm[i][i]
    fp=sum(cm[:,i])-tp
    fn=sum(cm[i,:])-tp
    tn=sum(sum(cm))-(tp+fp+fn)
    
    print(f"Class {i}:")
    print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)
    print()


plt.figure(figsize=(6, 5))

sea.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['No', 'Yes'],
            yticklabels=['No', 'Yes'])

plt.title(f"Confusion Matrix (Accuracy = {accuracy:.2f})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# iris.csv
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