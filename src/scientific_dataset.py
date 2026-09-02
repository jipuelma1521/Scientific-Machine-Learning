import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# Cargar dataset
data = load_breast_cancer()

X = data.data
y = data.target


print("Forma de X:", X.shape)
print("Forma de y:", y.shape)

print()
print("Clases:")
print(data.target_names)

print()
print("Primeras 10 features:")
print(data.feature_names[:10])


# Train / test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


print()
print("Datos de entrenamiento:", X_train.shape)
print("Datos de test:", X_test.shape)


# Random Forest
modelo = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# Entrenar
modelo.fit(X_train, y_train)


# Predecir
y_pred = modelo.predict(X_test)


# Métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

matriz = confusion_matrix(y_test, y_pred)


print()
print("RESULTADOS")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)

print()
print("Matriz de confusión:")
print(matriz)


# Feature importance
importancias = modelo.feature_importances_

indices = np.argsort(importancias)[::-1]


print()
print("TOP 10 FEATURES MÁS IMPORTANTES")

for i in range(10):

    indice = indices[i]

    print(
        data.feature_names[indice],
        ":",
        importancias[indice]
    )


# Top 10
top_indices = indices[:10]

top_features = data.feature_names[top_indices]
top_importancias = importancias[top_indices]


# Gráfico
plt.figure(figsize=(10, 6))

plt.barh(
    top_features[::-1],
    top_importancias[::-1]
)

plt.xlabel("Importancia")
plt.ylabel("Feature")

plt.title(
    "Features más importantes - Random Forest"
)

plt.tight_layout()

plt.show()