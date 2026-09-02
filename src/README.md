# Scientific Machine Learning

A scientific computing project developed in Python to explore the fundamentals of supervised machine learning using `scikit-learn`.

The project covers regression, classification, overfitting, regularization, decision trees, random forests, feature importance, cross-validation, and scientific data analysis.

## Topics

This project includes:

- Linear regression
- Train/test splitting
- Model evaluation
- Mean Squared Error
- Root Mean Squared Error
- R² score
- Polynomial regression
- Overfitting
- Ridge regularization
- Logistic regression
- Binary classification
- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrices
- Decision trees
- Random forests
- Feature importance
- Hyperparameters
- Cross-validation
- Scientific data classification

## Machine Learning Workflow

A supervised machine learning model can be represented as:

\[
\hat{y} = f(X; \theta)
\]

where:

- `X` represents the input features
- `y` represents the target
- `θ` represents parameters learned from data
- `ŷ` represents the prediction

A typical workflow is:

```text
Data
  ↓
Training data
  ↓
Model training
  ↓
Validation
  ↓
Hyperparameter selection
  ↓
Final model
  ↓
Test data
  ↓
Generalization estimate
```

## Linear Regression

The project begins with linear regression.

A simple linear model has the form:

\[
\hat{y} = ax + b
\]

The model learns the slope `a` and intercept `b` from noisy observations.

The parameters are selected by minimizing prediction error.

A common loss function is the sum of squared errors:

\[
L = \sum_i (y_i - \hat{y}_i)^2
\]

## Regression Metrics

### Mean Squared Error

\[
MSE =
\frac{1}{N}
\sum_i
(y_i - \hat{y}_i)^2
\]

Large prediction errors are penalized strongly because the residuals are squared.

### Root Mean Squared Error

\[
RMSE = \sqrt{MSE}
\]

RMSE is particularly useful because it has the same units as the target variable.

### R² Score

The coefficient of determination measures how much of the variation in the target is explained by the model.

A value close to:

\[
R^2 = 1
\]

indicates strong predictive performance.

## Training and Test Data

Machine learning models should not be evaluated only using the data used for training.

The dataset is therefore divided into:

- Training data
- Test data

The model learns using the training set.

The test set contains observations that the model has never seen during training.

This allows us to estimate how well the model generalizes to new data.

## Generalization

The objective of machine learning is not simply to obtain the lowest possible training error.

The real objective is to perform well on unseen observations.

Therefore:

\[
\text{low training error}
\not\Rightarrow
\text{good model}
\]

A good model should capture useful structure in the data without simply memorizing the training set.

## Overfitting

Polynomial regression is used to demonstrate overfitting.

A polynomial model can have the form:

\[
\hat{y}
=
a_0
+
a_1x
+
a_2x^2
+
\cdots
+
a_nx^n
\]

A low-degree polynomial may capture the general structure of the data.

A high-degree polynomial has much greater flexibility and can begin fitting random fluctuations.

This produces:

- Very low training error
- Poor test performance

This behavior is known as **overfitting**.

## Underfitting

The opposite problem is underfitting.

An underfitted model is too simple to capture the relevant structure of the dataset.

The objective is therefore to find a balance between:

```text
Too simple
   ↓
Underfitting

Appropriate complexity
   ↓
Good generalization

Too complex
   ↓
Overfitting
```

## Regularization

Ridge Regression is used to reduce overfitting.

Instead of minimizing only prediction error, Ridge adds a penalty for large coefficients:

\[
L =
\sum_i
(y_i - \hat{y}_i)^2
+
\alpha
\sum_j
\beta_j^2
\]

The first term measures prediction error.

The second term penalizes model complexity.

The hyperparameter:

```python
alpha
```

controls the strength of regularization.

Large values of `alpha` produce stronger regularization.

Regularization can slightly worsen training performance while significantly improving performance on unseen data.

## Classification

Regression predicts continuous quantities.

Classification predicts discrete categories.

For example:

```text
Measurements
    ↓
Model
    ↓
Class 0 or Class 1
```

The project explores binary classification using Logistic Regression.

## Logistic Regression

Logistic Regression computes a linear combination of the features:

\[
z =
\beta_0
+
\beta_1x_1
+
\beta_2x_2
+
\cdots
\]

This value is transformed using the sigmoid function:

\[
\sigma(z)
=
\frac{1}{1 + e^{-z}}
\]

The result lies between 0 and 1 and can be interpreted as a probability.

A decision threshold can then convert this probability into a predicted class.

## Classification Metrics

### Accuracy

Accuracy measures the fraction of all predictions that were correct.

\[
Accuracy =
\frac{\text{correct predictions}}
{\text{total predictions}}
\]

Accuracy can be misleading when the classes are strongly imbalanced.

### Precision

Precision answers:

> When the model predicts the positive class, how often is it correct?

\[
Precision =
\frac{TP}
{TP + FP}
\]

### Recall

Recall answers:

> Of all real positive observations, how many were detected?

\[
Recall =
\frac{TP}
{TP + FN}
\]

### F1 Score

The F1 score combines precision and recall:

\[
F1 =
2
\frac{
Precision \cdot Recall
}{
Precision + Recall
}
\]

## Confusion Matrix

A binary confusion matrix contains:

```text
                Predicted 0    Predicted 1

Real 0              TN              FP

Real 1              FN              TP
```

where:

- `TP` = True Positive
- `TN` = True Negative
- `FP` = False Positive
- `FN` = False Negative

Different kinds of mistakes can have very different consequences depending on the scientific problem.

## Decision Trees

Decision Trees classify observations using a sequence of rules.

For example:

\[
x_1 < a?
\]

Depending on the answer, the observation follows one branch of the tree.

Additional rules divide the feature space into increasingly specific regions.

Decision Trees can therefore construct nonlinear decision boundaries.

However, unrestricted trees can easily memorize training data and overfit.

## Random Forest

Random Forest combines many Decision Trees.

Instead of relying on a single tree, many trees are trained using variations of the dataset and feature subsets.

For classification, the final prediction is based on the collective predictions of the trees.

Conceptually:

```text
Tree 1 ─┐
Tree 2 ─┤
Tree 3 ─┤
   ...  ├──> Voting ───> Final prediction
Tree N ─┘
```

This usually improves stability and generalization compared with a single unrestricted Decision Tree.

## Feature Importance

Random Forest can estimate the predictive importance of different input features.

For example:

```text
Feature 1: 0.70
Feature 2: 0.20
Feature 3: 0.10
```

Higher importance means that the feature was more useful for the model when separating observations.

However:

\[
\text{predictive importance}
\neq
\text{causal importance}
\]

A highly predictive variable does not necessarily cause the observed phenomenon.

Correlated features can also share or redistribute importance.

## Scientific Dataset

The project applies Machine Learning to the Wisconsin Diagnostic Breast Cancer dataset included in `scikit-learn`.

The dataset contains:

- 569 observations
- 30 numerical features
- 2 classes

Each observation contains measurements describing geometric and statistical characteristics extracted from cell nuclei.

Examples of features include:

- Mean radius
- Mean texture
- Mean perimeter
- Mean area
- Mean smoothness
- Mean compactness
- Mean concavity
- Mean concave points
- Worst radius
- Worst perimeter
- Worst area

The target classes are:

```text
malignant
benign
```

The dataset is used here as a scientific Machine Learning example.

## Scientific Dataset Results

Using a Random Forest classifier produced approximately:

```text
Accuracy: 0.958
Precision: 0.957
Recall: 0.978
F1: 0.967
```

The confusion matrix was:

```text
[[49  4]
 [ 2 88]]
```

The most predictive features included:

```text
worst perimeter
worst area
worst concave points
mean concave points
worst radius
mean perimeter
mean radius
mean concavity
mean area
worst concavity
```

These results demonstrate that a model can identify useful multidimensional patterns across many scientific measurements.

## Multidimensional Data

Earlier examples used a single feature:

\[
x \rightarrow y
\]

The scientific dataset contains 30 features:

\[
(x_1,x_2,\ldots,x_{30})
\rightarrow y
\]

The feature matrix therefore has the form:

\[
X =
\begin{pmatrix}
x_{11} & x_{12} & \cdots & x_{1,30} \\
x_{21} & x_{22} & \cdots & x_{2,30} \\
\vdots & \vdots & & \vdots
\end{pmatrix}
\]

Each row represents one observation.

Each column represents one feature.

## Parameters and Hyperparameters

An important distinction in Machine Learning is the difference between parameters and hyperparameters.

### Parameters

Parameters are learned automatically during training.

Examples include:

- Linear regression coefficients
- Logistic regression coefficients
- Decision Tree split thresholds

### Hyperparameters

Hyperparameters are selected before training.

Examples include:

```python
Ridge(alpha=1.0)
```

```python
DecisionTreeClassifier(max_depth=5)
```

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=8
)
```

The model does not automatically learn these values during normal training.

They must be selected using an external evaluation procedure.

## Validation

Using the test dataset repeatedly to choose models or hyperparameters would indirectly leak information from the test set into the modeling process.

A better structure is:

```text
Training
Validation
Test
```

The validation data is used for model selection.

The final test data is reserved for the final evaluation.

## Cross-Validation

The project uses 5-fold cross-validation.

The training data is divided into five subsets.

The model is trained five times.

```text
Fold 1:
VALIDATION | TRAIN | TRAIN | TRAIN | TRAIN

Fold 2:
TRAIN | VALIDATION | TRAIN | TRAIN | TRAIN

Fold 3:
TRAIN | TRAIN | VALIDATION | TRAIN | TRAIN

Fold 4:
TRAIN | TRAIN | TRAIN | VALIDATION | TRAIN

Fold 5:
TRAIN | TRAIN | TRAIN | TRAIN | VALIDATION
```

This generates five validation scores:

\[
A_1,A_2,A_3,A_4,A_5
\]

The average score is:

\[
\bar{A}
=
\frac{
A_1+A_2+A_3+A_4+A_5
}{5}
\]

Cross-validation provides a more robust estimate than relying on a single validation split.

## Hyperparameter Selection

Different Random Forest depths were tested using cross-validation.

Example results:

```text
max_depth = 2    Accuracy CV ≈ 0.947
max_depth = 4    Accuracy CV ≈ 0.947
max_depth = 6    Accuracy CV ≈ 0.956
max_depth = 8    Accuracy CV ≈ 0.960
max_depth = None Accuracy CV ≈ 0.960
```

The selected model used:

```python
max_depth=8
```

The final accuracy on the untouched test dataset was approximately:

```text
Accuracy test ≈ 0.956
```

The similarity between validation and test performance indicates reasonable generalization.

## Final Machine Learning Pipeline

The complete workflow explored in this project is:

```text
Scientific data
      ↓
Feature matrix X
Target vector y
      ↓
Train / Test split
      ↓
Training data
      ↓
Cross-validation
      ↓
Hyperparameter selection
      ↓
Final model training
      ↓
Untouched test data
      ↓
Performance metrics
      ↓
Scientific interpretation
```

## Machine Learning and Physics

Traditional physical modeling often begins with physical principles.

For example:

\[
F = ma
\]

A physical theory provides equations that describe the behavior of the system.

Machine Learning can instead begin directly from observations:

\[
(x_1,x_2,\ldots,x_n)
\rightarrow y
\]

The algorithm searches for predictive relationships in the data.

This means that a Machine Learning model can produce excellent predictions without necessarily providing a fundamental physical explanation.

For scientific applications, it is therefore important to distinguish between:

- Prediction
- Correlation
- Interpretation
- Causality
- Physical explanation

## Technologies

- Python
- NumPy
- Matplotlib
- scikit-learn

## Installation

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Project

Run:

```bash
python main.py
```

## Project Structure

```text
Scientific-Machine-Learning/
│
├── data/
│
├── results/
│
├── src/
│   ├── regression.py
│   ├── overfitting.py
│   ├── classification.py
│   ├── random_forest.py
│   └── scientific_dataset.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Main Concepts Learned

This project explores the foundations of supervised Machine Learning:

- Features and targets
- Training and prediction
- Regression
- Classification
- Train/test splitting
- Model evaluation
- MSE
- RMSE
- R²
- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrices
- Polynomial models
- Underfitting
- Overfitting
- Regularization
- Logistic Regression
- Decision Trees
- Random Forests
- Feature importance
- Parameters
- Hyperparameters
- Validation
- Cross-validation
- Generalization
- Multidimensional scientific data
- Predictive importance versus causality
- Machine Learning versus physical modeling