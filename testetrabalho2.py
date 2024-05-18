import numpy as np
import random
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_table, read_csv
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import sklearn.tree as tree


pd.set_option('future.no_silent_downcasting', True)


def outiline(x):
    for column in x.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        outliers = df[(df[column] < limite_inferior) |
                            (df[column] > limite_superior)][column]
        print("Quartil 1 (Q1):", Q1)
        print("Quartil 3 (Q3):", Q3)
        print("Diferença Interquartil (IQR):", IQR)  #ou posso simplesmente usar print(outliers)
        print("Limite Inferior:", limite_inferior)   #que no caso vai me dar o valor outliers: 107 5.0
        print("Limite Superior:", limite_superior)
        print("Outliers:", outliers)

# Função para avaliar e plotar a matriz de confusão
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    print(f"{model_name} Accuracy: {accuracy:.2f}")

    cm = confusion_matrix(y_test, prediction)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Dies", "Lives"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()
        

# Carregar e preparar os dados
df = read_csv("hcc_dataset.csv", sep=",", na_values='?')
df.fillna(df.mode().iloc[0], inplace=True)
df.replace('?', df.mode().iloc[0], inplace=True)
df.replace('Yes', 1, inplace=True)
df.replace('No', 0, inplace=True)
value_counts = df['Class'].value_counts()

print(value_counts)


x = df
if "Class" in x.columns:
    x = x.drop(columns=["Class"])
string_columns = x.select_dtypes(include=['object']).columns
x = pd.get_dummies(x, columns=string_columns)
x = x.astype(int)
scaler = MinMaxScaler()
cx = scaler.fit_transform(x)
x = pd.DataFrame(cx, columns=x.columns)
y = df[["Class",]]



# 
plt.figure(figsize=(15,10))
sns.heatmap(x.corr(), annot=False, square=True, cmap='coolwarm')
plt.show()

selector = SelectKBest(score_func=f_classif, k=12)  
x_new = selector.fit_transform(x, y.values.ravel())
selected_features = x.columns[selector.get_support(indices=True)]
x = pd.DataFrame(x_new, columns=selected_features)
x = pd.concat([x, y], axis=1)
x.replace('Lives', 1, inplace=True)
x.replace('Dies', 0, inplace=True)
corr_matrix = x.corr()
plt.figure(figsize=(15,10))
sns.heatmap(x.corr(), annot=True, square=True, cmap='coolwarm')
plt.show()
x = x.drop(columns='Class')
corr_matrix = x.corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# Find columns with correlation greater than a threshold (e.g., 0.95)
threshold = 0.81
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
# Drop highly correlated columns
x = x.drop(columns=to_drop)
#for negative correlation
threshold = -0.81
to_drop = [column for column in upper.columns if any(upper[column] < threshold)]
# Drop highly correlated columns
x = x.drop(columns=to_drop)
x = pd.concat([x, y], axis=1)
x.replace('Lives', 1, inplace=True)
x.replace('Dies', 0, inplace=True)
plt.figure(figsize=(15,10))
sns.heatmap(x.corr(), annot=False, square=True, cmap='coolwarm')
plt.show()
print(x.head(10))
for column in x.columns.drop('Class'):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=x, x=column, hue='Class', element="bars", bins=20, kde=True)
    plt.title(f'Histogram of {column} by Class')
    plt.show()
x = x.drop(columns='Class')




y = y.values.ravel()
# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random.randint(0,1000), stratify=y)

# Árvore de decisão
arv = tree.DecisionTreeClassifier(max_leaf_nodes=5, criterion="gini", random_state=random.randint(0,100))
evaluate_model(arv, X_train, X_test, y_train, y_test, "Decision Tree")
plt.figure(figsize=(8,8))
tree.plot_tree(arv, feature_names=x.columns.tolist(), class_names=["lives", "dies"], filled=True)
plt.show()


# KNN com k visinhos
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
evaluate_model(knn, X_train, X_test, y_train, y_test, "KNN")

# Random Forest
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random.randint(0,1000), stratify=y)
rf = RandomForestClassifier(n_estimators=1000, random_state=random.randint(0,1000))
evaluate_model(rf, X_train, X_test, y_train, y_test, "Random Forest")
