from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. Cargar dataset
# --------------------------------------------------

data = load_breast_cancer()

X = data.data
y = data.target


# --------------------------------------------------
# 2. Separar test final
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. Probar distintas profundidades
# --------------------------------------------------

profundidades = [2, 4, 6, 8, None]

mejor_score = 0
mejor_profundidad = None


for profundidad in profundidades:

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=profundidad,
        random_state=42
    )

    scores = cross_val_score(
        modelo,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    promedio = scores.mean()
    desviacion = scores.std()

    print(
        "max_depth =",
        profundidad,
        "| Accuracy CV =",
        promedio,
        "+/-",
        desviacion
    )

    if promedio > mejor_score:
        mejor_score = promedio
        mejor_profundidad = profundidad


# --------------------------------------------------
# 4. Mejor hiperparámetro
# --------------------------------------------------

print()
print("Mejor profundidad:", mejor_profundidad)
print("Mejor accuracy CV:", mejor_score)


# --------------------------------------------------
# 5. Modelo final
# --------------------------------------------------

modelo_final = RandomForestClassifier(
    n_estimators=200,
    max_depth=mejor_profundidad,
    random_state=42
)

modelo_final.fit(X_train, y_train)


# --------------------------------------------------
# 6. Evaluación final
# --------------------------------------------------

y_pred = modelo_final.predict(X_test)

accuracy_test = accuracy_score(
    y_test,
    y_pred
)


print()
print(
    "Accuracy final en test:",
    accuracy_test
)