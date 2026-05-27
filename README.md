# House Prices - Advanced Regression Techniques
**Kaggle Public Score: 0.12031 | (324th place)**

---

## 🔗 Links
- [Kaggle Notebook](https://www.kaggle.com/code/thinkrunner/notebook7f7c063978)
- [Kaggle Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

## Project Summary

End-to-end ML pipeline for predicting house sale prices using stacking ensemble.  
Starting from a baseline score of 0.13640, each step was carefully designed to improve performance.

| Step | Technique | Score |
|------|-----------|-------|
| Baseline | First submission | 0.13640 |
| Optuna Tuning | LightGBM hyperparameter optimization | 0.12652 |
| Stacking | LGB + XGB + GBM + CatBoost + ElasticNet | 0.12257 |
| Target Encoding | Neighborhood, Exterior1st, Exterior2nd, MSSubClass | 0.12057 |

---

## Pipeline Overview

```
Raw Data
  → Data Loading & Missing Value Imputation
  → ID Separation
  → Outlier Removal
  → Feature Engineering (numerical + categorical)
  → Target Encoding (high-cardinality categoricals)
  → Skewness Transform (log1p)
  → StandardScaler + One-Hot Encoding
  → Feature Selection (importance > 0 intersection → 79 features)
  → Stacking Ensemble (5-Fold CV)
  → Final Prediction & Submission
```

---

## Cross-Validation Strategy

- **Method**: KFold(n_splits=5, shuffle=True, random_state=42)
- **Metric**: RMSLE (log1p transform applied to target before scoring)
- **Leakage prevention**: All encoders and scalers fitted on train fold only; test set uses `transform()` exclusively

---

## Key Design Decisions

### 1. Outlier Removal
Removed extreme outliers before preprocessing to prevent distortion:
```python
train_df = train_df[train_df['GrLivArea'] <= 4500]
train_df = train_df[train_df['LotArea'] <= 200000]
train_df = train_df[train_df['LotFrontage'] <= 300]
```

### 2. Feature Engineering
Derived features based on correlation analysis (Pearson correlation & correlation ratio):

**Numerical features**

| Feature | Formula | Weight Rationale |
|---------|---------|-----------|
| `new_GrLivArea` | `1stFlrSF × 1.2 + 2ndFlrSF × 0.7` | Weights manually tuned based on Pearson correlation with SalePrice (1stFlr > 2ndFlr); outperforms raw GrLivArea |
| `Porch_Deck` | `OpenPorchSF × 1.05 + 3SsnPorch × 0.37 + ScreenPorch × 0.37 + WoodDeckSF × 0.58` | Weights reflect per-type correlation with price; EnclosedPorch excluded (negative correlation) |
| `HouseAge` | `YrSold - YearBuilt` | Age captures depreciation more directly than raw year |
| `SaleCondition_flag` | `1 if Normal or Partial else 0` | Separates standard transactions from distressed/atypical sales |

**Categorical features (Ordinal Encoding)**

| Feature | Mapping | Rationale |
|---------|---------|-----------|
| `ExterQual`, `KitchenQual`, `GarageQual`, etc. | `Ex=5, Gd=4, TA=3, Fa=2, Po=1, NA=0` | Natural ordinal relationship with price |
| `BsmtCond`, `FireplaceQu` | `Ex=5, Gd=4, TA=3, Fa=2, Po=0, NA=1` | NA ranked above Po(poor) based on price analysis |

### 3. Target Encoding
Applied to high-cardinality categorical features to preserve target relationship:
```python
target_enc_cols = ['Neighborhood', 'Exterior1st', 'Exterior2nd', 'MSSubClass']
te = TargetEncoder(cols=target_enc_cols, smoothing=10)
te.fit(X_train[target_enc_cols], y_log)  # fit on train only (no leakage)
```

**Why these columns?**
- Neighborhood: 25 unique values → strong price signal by area
- MSSubClass: 15 unique values → dwelling type strongly affects price
- Exterior1st/2nd: 15-16 unique values → material quality signal

### 4. Feature Selection
Used intersection of all model importances to keep only universally important features:
```python
inter_features = lgb_features & xgb_features & gbm_features & catboost_features
# Selection criterion: importance > 0 in ALL 4 models → 79 features
```
- Each model: all features with importance > 0 selected individually
- Final feature set = intersection across all 4 models → **79 features**
- Rationale: features with zero importance in any model are likely noise or redundant

### 5. Stacking Ensemble
Meta-model learns optimal combination of base model predictions:
```python
stack_model = StackingRegressor(
    estimators=[
        ('lgb', LGBMRegressor),    # leaf-wise boosting
        ('xgb', XGBRegressor),     # depth-wise boosting
        ('gbm', GradientBoosting), # original boosting
        ('cat', CatBoostRegressor),# ordered boosting
        ('elastic', ElasticNet),   # linear model for diversity
    ],
    final_estimator=RidgeCV(),  # meta-model
    cv=5
)
```

Base models cover both tree-based and linear approaches for diversity:
- Tree-based: LGB, XGB, GBM, CatBoost
- Linear: ElasticNet (L1+L2 regularization)

**No data leakage**: train and test preprocessed independently.  
Test set uses only `transform()`, never `fit_transform()`.

---

## Model Performance

| Model | Local CV RMSLE | Kaggle Score | Gap |
|-------|---------------|--------------|-----|
| LightGBM (single) | 0.1169 | 0.12647 | 0.0096 |
| XGBoost (single) | 0.1128 | 0.12970 | 0.0169 |
| GBM (single) | 0.1124 | 0.13080 | 0.0184 |
| CatBoost (single) | 0.1104 | 0.12974 | 0.0193 |
| ElasticNet (single) | 0.1125 | 0.13493 | 0.0224 |
| **Stacking (LGB+XGB+GBM+CAT+ElasticNet)** | **0.1107** | **0.12031** | **0.0096** |

> **Gap analysis**: ElasticNet showed the largest gap (0.0224), reflecting linear model limitations
> on non-linear data. Among tree-based models, CatBoost had the largest gap (0.0193),
> followed by GBM (0.0184) and XGBoost (0.0169).
> Stacking reduced the gap to 0.0096, matching LightGBM's generalization while improving overall score.

---

## What Worked

- Target Encoding on high-cardinality features → biggest single improvement (-0.002)
- Stacking over simple weighted average → consistent improvement
- Feature intersection across models → reduced noise
- No leakage in preprocessing → reliable generalization

## What Didn't Work

- Ridge as standalone model → 0.13550 (linear model limitation on non-linear data)
- XGBoost with aggressive feature selection → overfitting (local 0.112 vs Kaggle 0.133); train-specific patterns not generalizing
- Adding neighbor's domain features → 0.12144 (worse than current 0.12031); added noise outweighed signal

---

## Environment

```
Python      3.11.15
LightGBM    4.6.0
XGBoost     3.2.0
CatBoost    1.2.10
scikit-learn 1.8.0
category_encoders 2.9.0
optuna      4.8.0
```
