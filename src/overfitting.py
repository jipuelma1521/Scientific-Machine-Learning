import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error


# Crear datos
np.random.seed(42)

X = np.linspace(-3, 3, 40)
y_real = X**2

ruido = np.random.normal(0, 2, size=len(X))
y = y_real + ruido

X = X.reshape(-1, 1)


# Separar train y test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# Modelo grado 2
modelo_grado_2 = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)


# Modelo grado 15 sin regularización
modelo_grado_15 = make_pipeline(
    PolynomialFeatures(degree=15),
    LinearRegression()
)


# Modelo grado 15 con Ridge
modelo_ridge = make_pipeline(
    PolynomialFeatures(degree=15),
    StandardScaler(),
    Ridge(alpha=1.0)
)


# Entrenar
modelo_grado_2.fit(X_train, y_train)
modelo_grado_15.fit(X_train, y_train)
modelo_ridge.fit(X_train, y_train)


# Predicciones
pred_train_2 = modelo_grado_2.predict(X_train)
pred_test_2 = modelo_grado_2.predict(X_test)

pred_train_15 = modelo_grado_15.predict(X_train)
pred_test_15 = modelo_grado_15.predict(X_test)

pred_train_ridge = modelo_ridge.predict(X_train)
pred_test_ridge = modelo_ridge.predict(X_test)


# RMSE
rmse_train_2 = np.sqrt(
    mean_squared_error(y_train, pred_train_2)
)

rmse_test_2 = np.sqrt(
    mean_squared_error(y_test, pred_test_2)
)

rmse_train_15 = np.sqrt(
    mean_squared_error(y_train, pred_train_15)
)

rmse_test_15 = np.sqrt(
    mean_squared_error(y_test, pred_test_15)
)

rmse_train_ridge = np.sqrt(
    mean_squared_error(y_train, pred_train_ridge)
)

rmse_test_ridge = np.sqrt(
    mean_squared_error(y_test, pred_test_ridge)
)


print("GRADO 2")
print("RMSE train:", rmse_train_2)
print("RMSE test:", rmse_test_2)

print()

print("GRADO 15 SIN REGULARIZACIÓN")
print("RMSE train:", rmse_train_15)
print("RMSE test:", rmse_test_15)

print()

print("GRADO 15 CON RIDGE")
print("RMSE train:", rmse_train_ridge)
print("RMSE test:", rmse_test_ridge)


# Curvas
X_grafico = np.linspace(-3, 3, 500).reshape(-1, 1)

y_2 = modelo_grado_2.predict(X_grafico)
y_15 = modelo_grado_15.predict(X_grafico)
y_ridge = modelo_ridge.predict(X_grafico)


# Gráfico
plt.scatter(X_train, y_train, label="Train")
plt.scatter(X_test, y_test, label="Test")

plt.plot(
    X_grafico,
    y_2,
    label="Polinomio grado 2"
)

plt.plot(
    X_grafico,
    y_15,
    label="Grado 15 sin regularización"
)

plt.plot(
    X_grafico,
    y_ridge,
    label="Grado 15 con Ridge"
)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Overfitting and Ridge Regularization")
plt.legend()

plt.ylim(-10, 30)

plt.show()