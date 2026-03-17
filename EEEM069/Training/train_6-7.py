import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Input cleaned dataset
FILE_PATH = "/Users/louisilett/EEEM069/ELSA Dataset/Predictive/dataset_wave6_predict_wave7_clean.csv"

df = pd.read_csv(FILE_PATH)

# Separate features and target
X = df.drop(columns=["future_depression"])
y = df["future_depression"]

print("Dataset shape:", df.shape)
print("Feature count:", X.shape[1])
print("Target distribution:")
print(y.value_counts())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# -----------------------------
# Logistic Regression
# -----------------------------
log_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)
log_prob = log_model.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression Results ===")
print(classification_report(y_test, log_pred, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, log_pred))
print("ROC AUC:", round(roc_auc_score(y_test, log_prob), 4))

# -----------------------------
# Random Forest
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    min_samples_split=10,
    min_samples_leaf=5
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n=== Random Forest Results ===")
print(classification_report(y_test, rf_pred, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print("ROC AUC:", round(roc_auc_score(y_test, rf_prob), 4))

# -----------------------------
# Feature Importance
# -----------------------------
importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nTop 10 Random Forest Features:")
print(importance_df.head(10).to_string(index=False))

# -----------------------------
# Feature Importance Plot
# -----------------------------

importance_df.head(10).plot(
    kind="barh",
    x="feature",
    y="importance",
    legend=False
)

plt.title("Top Predictors of Future Depression")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# -----------------------------
# ROC Curve
# -----------------------------

fpr_log, tpr_log, _ = roc_curve(y_test, log_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

plt.plot(fpr_log, tpr_log, label="Logistic Regression")
plt.plot(fpr_rf, tpr_rf, label="Random Forest")
plt.plot([0,1],[0,1], linestyle="--", label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
