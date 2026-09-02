import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# Crear datos
np.random.seed(42)

X = np.linspace(0, 10, 100)
ruido = np.random.normal(0, 2, size=100)

y = 3 * X + 2 + ruido

X = X.reshape(-1, 1)


# Separar train y test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Crear modelo
modelo = LinearRegression()


# Entrenar
modelo.fit(X_train, y_train)


# Predecir
y_pred = modelo.predict(X_test)


# Métricas
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print("Pendiente encontrada:", modelo.coef_[0])
print("Intercepto encontrado:", modelo.intercept_)

print()
print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)


# Recta para visualizar
X_linea = np.linspace(0, 10, 100).reshape(-1, 1)
y_linea = modelo.predict(X_linea)


# Gráfico
plt.scatter(X_train, y_train, label="Entrenamiento")
plt.scatter(X_test, y_test, label="Test")
plt.plot(X_linea, y_linea, label="Modelo")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression")
plt.legend()

plt.show()