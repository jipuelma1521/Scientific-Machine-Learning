import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Crear datos
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=2,
    class_sep=1.0,
    random_state=42
)


# Train / test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# Decision Tree
arbol = DecisionTreeClassifier(
    max_depth=None,
    random_state=42
)

arbol.fit(X_train, y_train)


# Random Forest
bosque = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

bosque.fit(X_train, y_train)


# Predicciones
pred_arbol_train = arbol.predict(X_train)
pred_arbol_test = arbol.predict(X_test)

pred_bosque_train = bosque.predict(X_train)
pred_bosque_test = bosque.predict(X_test)


# Accuracy
print("ÁRBOL")
print(
    "Accuracy train:",
    accuracy_score(y_train, pred_arbol_train)
)
print(
    "Accuracy test:",
    accuracy_score(y_test, pred_arbol_test)
)

print()

print("RANDOM FOREST")
print(
    "Accuracy train:",
    accuracy_score(y_train, pred_bosque_train)
)
print(
    "Accuracy test:",
    accuracy_score(y_test, pred_bosque_test)
)


# Feature importance
print()
print("IMPORTANCIA DE FEATURES")
print("Feature 1:", bosque.feature_importances_[0])
print("Feature 2:", bosque.feature_importances_[1])


# Malla
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = np.c_[xx.ravel(), yy.ravel()]


# Decision Tree
pred_grid_arbol = arbol.predict(grid)
pred_grid_arbol = pred_grid_arbol.reshape(xx.shape)

plt.figure()

plt.contourf(
    xx,
    yy,
    pred_grid_arbol,
    alpha=0.2
)

plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test,
    edgecolors="black"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Tree")

plt.show()


# Random Forest
pred_grid_bosque = bosque.predict(grid)
pred_grid_bosque = pred_grid_bosque.reshape(xx.shape)

plt.figure()

plt.contourf(
    xx,
    yy,
    pred_grid_bosque,
    alpha=0.2
)

plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test,
    edgecolors="black"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Random Forest")

plt.show()