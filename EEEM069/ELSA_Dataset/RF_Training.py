"""
MSc AI & Health Group Project

Predicts depression at Wave 7 using features from Wave 6

Author: [Louis Ilett]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

print("=" * 60)
print("ELSA DEPRESSION PREDICTION - RANDOM FOREST PIPELINE")
print("=" * 60)

w6 = pd.read_csv('ELSA_Clean/clean_wave_6_elsa_data_v2.csv')
w7 = pd.read_csv('ELSA_Clean/clean_wave_7_elsa_data.csv')

print(f"\nWave 6: {w6.shape[0]:,} participants, {w6.shape[1]} columns")
print(f"Wave 7: {w7.shape[0]:,} participants, {w7.shape[1]} columns")


# ─────────────────────────────────────────────
# 2. BUILD TARGET VARIABLE (Wave 7 Depression)
# ─────────────────────────────────────────────

# GHQ-6 items in ELSA:
#   hepsyde = depression
#   hepsyan = anxiety
#   hepsyem = emotional problems
#   hepsyps = positive mental health (lower = worse)
#   hepsymo = morale
#   hepsyma = mastery/control
HEPSY_COLS = ['hepsyde', 'hepsyan', 'hepsyem', 'hepsyps', 'hepsymo', 'hepsyma']

w7_target = w7[['idauniq'] + HEPSY_COLS].copy()

# Replace ELSA missing value codes with NaN
for col in HEPSY_COLS:
    w7_target[col] = w7_target[col].replace([-1, -8, -9], np.nan)

# Keep only participants with complete psych data at wave 7
w7_target = w7_target.dropna(subset=HEPSY_COLS)

# Sum score: range 0-6; threshold >= 2 indicates likely depression
w7_target['depression_score_w7'] = w7_target[HEPSY_COLS].sum(axis=1)
w7_target['depressed_w7'] = (w7_target['depression_score_w7'] >= 2).astype(int)

print(f"\nWave 7 participants with valid psych data: {len(w7_target):,}")
print(f"Depressed (score >= 2): {w7_target['depressed_w7'].sum()} ({w7_target['depressed_w7'].mean()*100:.1f}%)")
print(f"Not depressed:          {(w7_target['depressed_w7'] == 0).sum()} ({(1-w7_target['depressed_w7'].mean())*100:.1f}%)")


# ─────────────────────────────────────────────
# 3. BUILD FEATURE SET (Wave 6)
# ─────────────────────────────────────────────

# Feature descriptions:
# DiSex     - Sex (1=male, 2=female)
# DiMar     - Marital status
# HeAge     - Age (mostly -1 = age banded / not given directly)
# DiMaedu   - Education level
# MiLive    - Living situation
# WhoSo1-5  - Social contact indicators (binary)
# Hehelf    - Self-rated health (1=excellent to 5=poor)
# Heill     - Long-standing illness (1=yes, 2=no)
# Helim     - Health limits activities
# HePain    - Pain (1=yes, 2=no)
# HeSmk     - Smoking status
# HeActa-c  - Physical activity (vigorous/moderate/mild)
# Heiqa-e   - Cognitive/IQ items (mostly sparse)
# hepsyde-ma - Wave 6 psych scores (prior depression - strong predictor)

FEATURE_COLS = [
    'DiSex', 'DiMar', 'DiMaedu',
    'WhoSo1', 'WhoSo2', 'WhoSo3', 'WhoSo4', 'WhoSo5',
    'Hehelf', 'Heill', 'Helim', 'HePain', 'HeSmk',
    'HeActa', 'HeActb', 'HeActc',
    # Prior wave psych scores — likely strongest predictors
    'hepsyde', 'hepsyan', 'hepsyem', 'hepsyps', 'hepsymo', 'hepsyma'
]

w6_features = w6[['idauniq'] + FEATURE_COLS].copy()

# Replace ELSA missing codes with NaN
MISSING_CODES = [-1, -8, -9]
for col in FEATURE_COLS:
    w6_features[col] = w6_features[col].replace(MISSING_CODES, np.nan)

# Replace hepsy negative values separately (same codes)
for col in HEPSY_COLS:
    w6_features[col] = w6_features[col].replace(MISSING_CODES, np.nan)

# Add wave 6 depression score as a summary feature
w6_features['depression_score_w6'] = w6_features[HEPSY_COLS].sum(axis=1, min_count=1)

print(f"\nFeatures used from Wave 6: {len(FEATURE_COLS) + 1}")  # +1 for score


# ─────────────────────────────────────────────
# 4. MERGE WAVES
# ─────────────────────────────────────────────

df = pd.merge(
    w6_features,
    w7_target[['idauniq', 'depressed_w7', 'depression_score_w7']],
    on='idauniq',
    how='inner'
)

print(f"\nMerged dataset (participants in both waves with valid W7 target): {len(df):,}")

# Drop ID columns before modelling
X = df.drop(columns=['idauniq', 'depressed_w7', 'depression_score_w7'])
y = df['depressed_w7']

print(f"Feature matrix shape: {X.shape}")
print(f"\nMissing values per feature:")
missing = X.isnull().sum()
missing = missing[missing > 0]
for col, count in missing.items():
    print(f"  {col}: {count} ({count/len(X)*100:.1f}%)")


# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

# Stratified split to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train):,} participants")
print(f"Test set:     {len(X_test):,} participants")
print(f"Train class balance: {y_train.mean()*100:.1f}% depressed")
print(f"Test class balance:  {y_test.mean()*100:.1f}% depressed")


# ─────────────────────────────────────────────
# 6. BUILD PIPELINE & TRAIN
# ─────────────────────────────────────────────

# Pipeline: median imputation → Random Forest
# class_weight='balanced' handles the class imbalance automatically
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('rf', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining Random Forest...")
pipeline.fit(X_train, y_train)
print("Training complete.")


# ─────────────────────────────────────────────
# 7. EVALUATE
# ─────────────────────────────────────────────

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE ON TEST SET")
print("=" * 60)
print(f"\nROC-AUC Score: {auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Depressed', 'Depressed']))

# Cross-validation for more robust estimate
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"5-Fold Cross-Validation AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")


# ─────────────────────────────────────────────
# 8. PLOTS
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('ELSA Depression Prediction — Random Forest Results', fontsize=14, fontweight='bold')

# --- Plot 1: Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Depressed', 'Depressed'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix')

# --- Plot 2: ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc:.3f}')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

# --- Plot 3: Feature Importance ---
rf_model = pipeline.named_steps['rf']
feature_names = X.columns.tolist()
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:15]  # Top 15

bars = axes[2].barh(
    range(len(indices)),
    importances[indices][::-1],
    color='steelblue', alpha=0.8
)
axes[2].set_yticks(range(len(indices)))
axes[2].set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=9)
axes[2].set_xlabel('Feature Importance (Gini)')
axes[2].set_title('Top 15 Most Important Features')
axes[2].grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('elsa_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as elsa_results.png")

# --- Print top features ---
print("\nTop 15 Most Important Features:")
print(f"{'Rank':<5} {'Feature':<25} {'Importance':<10}")
print("-" * 40)
for rank, idx in enumerate(indices, 1):
    print(f"{rank:<5} {feature_names[idx]:<25} {importances[idx]:.4f}")

print("\nPipeline complete.")
