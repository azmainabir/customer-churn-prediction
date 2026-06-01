import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('data/telco_churn.csv')
print(f"Dataset shape: {df.shape}")
print(df.head())

# ─────────────────────────────────────────
# 2. CLEAN DATA
# ─────────────────────────────────────────
print("\nCleaning data...")

# Fix TotalCharges column (has spaces instead of numbers)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Drop customerID (not useful for prediction)
df.drop('customerID', axis=1, inplace=True)

# Convert target to binary (Yes=1, No=0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print(f"Cleaned dataset shape: {df.shape}")
print(f"Churn rate: {df['Churn'].mean()*100:.1f}%")

# ─────────────────────────────────────────
# 3. ENCODE CATEGORICAL COLUMNS
# ─────────────────────────────────────────
print("\nEncoding categorical columns...")

categorical_cols = df.select_dtypes(include='object').columns.tolist()
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

print(f"Encoded columns: {categorical_cols}")

# ─────────────────────────────────────────
# 4. SPLIT DATA
# ─────────────────────────────────────────
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape}")
print(f"Testing set: {X_test.shape}")

# ─────────────────────────────────────────
# 5. TRAIN 3 MODELS
# ─────────────────────────────────────────
print("\nTraining models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'auc': auc
    }
    print(f"\n{name} — AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────
# 6. SAVE BEST MODEL (XGBoost)
# ─────────────────────────────────────────
print("\nSaving best model...")
os.makedirs('models', exist_ok=True)
joblib.dump(results['XGBoost']['model'], 'models/xgboost_model.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
print("Model saved to models/xgboost_model.pkl")

# ─────────────────────────────────────────
# 7. SAVE CHARTS TO ASSETS FOLDER
# ─────────────────────────────────────────
os.makedirs('assets', exist_ok=True)

# Chart 1 — Confusion Matrix
print("\nSaving confusion matrix...")
cm = confusion_matrix(y_test, results['XGBoost']['y_pred'])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.title('Confusion Matrix — XGBoost')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('assets/confusion_matrix.png', dpi=150)
plt.close()

# Chart 2 — ROC Curve (all 3 models)
print("Saving ROC curve...")
plt.figure(figsize=(8, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Model Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('assets/roc_curve.png', dpi=150)
plt.close()

# Chart 3 — Feature Importance
print("Saving feature importance chart...")
xgb_model = results['XGBoost']['model']
feat_imp = pd.Series(xgb_model.feature_importances_, index=X.columns)
feat_imp = feat_imp.sort_values(ascending=False).head(15)
plt.figure(figsize=(8, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title('Top 15 Feature Importances — XGBoost')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('assets/feature_importance.png', dpi=150)
plt.close()

# Chart 4 — Churn Distribution
print("Saving churn distribution chart...")
plt.figure(figsize=(5, 4))
df['Churn'].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'], edgecolor='black')
plt.title('Churn Distribution')
plt.xticks([0, 1], ['No Churn', 'Churn'], rotation=0)
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('assets/churn_distribution.png', dpi=150)
plt.close()

print("\n✅ All done! Charts saved to assets/ folder.")
print("✅ Model saved to models/ folder.")
print("\nModel Performance Summary:")
for name, res in results.items():
    print(f"  {name}: AUC = {res['auc']:.4f}")