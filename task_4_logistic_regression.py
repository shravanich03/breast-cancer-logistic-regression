import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("cancer dataset.xlsx", sheet_name='in')

# Data preparation
X = df.drop(columns=['id', 'diagnosis'])
y = df['diagnosis'].map({'M': 1, 'B': 0})  # Malignant = 1, Benign = 0

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_probs = model.predict_proba(X_test_scaled)[:, 1]
y_preds = model.predict(X_test_scaled)

# Evaluation
conf_matrix = confusion_matrix(y_test, y_preds)
precision = precision_score(y_test, y_preds)
recall = recall_score(y_test, y_preds)
roc_auc = roc_auc_score(y_test, y_probs)

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")

# Save results
df.to_csv("cleaned_cancer_dataset.csv", index=False)
pd.DataFrame({'Actual': y_test.values, 'Predicted': y_preds, 'Probabilities': y_probs}).to_csv("predictions.csv", index=False)
