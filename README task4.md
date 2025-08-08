# Logistic Regression - Breast Cancer Classification

This project uses the Breast Cancer Wisconsin dataset to build a binary classifier using Logistic Regression.

## Steps Performed:
1. Loaded and cleaned the dataset.
2. Split the dataset into training and testing sets.
3. Standardized the feature values.
4. Trained a Logistic Regression model.
5. Evaluated the model using:
   - Confusion Matrix
   - Precision
   - Recall
   - ROC-AUC Score
6. Plotted the ROC Curve.

## Results:
- **Confusion Matrix**: [[70, 1], [2, 41]]
- **Precision**: 0.976
- **Recall**: 0.953
- **ROC AUC Score**: 0.997

## Tools Used:
- Pandas, Numpy
- Scikit-learn
- Matplotlib


---

## ✔️ Task Checklist and Explanation

### 1. Chose a Binary Classification Dataset
- **Dataset**: Breast Cancer Wisconsin
- **Target**: 'diagnosis' (Malignant = 1, Benign = 0)

### 2. Train/Test Split and Standardization
- Used 80% training and 20% testing split.
- Standardized features using `StandardScaler`.

### 3. Logistic Regression Model
- Model trained using `LogisticRegression` from Scikit-learn.
- Used probability predictions and class labels for evaluation.

### 4. Model Evaluation
- **Confusion Matrix**: [[70, 1], [2, 41]]
- **Precision**: 0.976  
- **Recall**: 0.953  
- **ROC-AUC Score**: 0.997  
- **ROC Curve**: Plotted and saved as `roc_curve.png`.

### 5. Threshold Tuning and Sigmoid Function
- Probabilities (`y_probs`) calculated using `predict_proba`.
- Threshold can be adjusted (e.g., `y_probs > 0.4`).
- **Sigmoid Function**:
  	Sigmoid(z) = 1 / (1 + e^(-z))  
  - Converts linear model output into probabilities.
  - Core of logistic regression for binary classification.


