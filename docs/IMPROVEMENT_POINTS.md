# Improvement Points for Large Prediction Errors

> How to reduce large errors (e.g., summer 130%, winter 53%) and improve MAPE

---

## 1. Feature Weighting (Sample Weights)

**Problem**: Summer and winter show large errors (e.g., summer 2022-06-25: actual 1385 → predicted 3196, 130% error).

**Solution**: Use `sample_weight` to upweight hard seasons or downweight outliers.

### LightGBM / CatBoost

```python
# In training: give higher weight to summer samples (or seasons with high error)
sample_weight = np.ones(len(y))
for i, row in train_df.iterrows():
    month = pd.to_datetime(str(row["date"]), format="%Y%m%d").month
    if month in [6, 7, 8]:  # summer
        sample_weight[i] = 2.0  # 2x weight for summer
    elif month in [12, 1, 2]:  # winter
        sample_weight[i] = 1.5  # 1.5x for winter

model.fit(X, y, sample_weight=sample_weight)
```

### Config-based approach

Add to `config/training_config.yaml`:

```yaml
target:
  use_log: true
  outlier_clip_pct: 1.0
  # Season weights: summer/winter get higher weight
  season_weights:
    spring: 1.0
    summer: 2.0   # upweight summer (high MAPE)
    fall: 1.0
    winter: 1.5   # upweight winter
```

---

## 2. Feature Importance / Manual Weighting

**Problem**: Some features (e.g., lag1, seasonal lags) may be underused.

**Solution**: Force important features via `feature_fraction` or `min_data_in_leaf` per feature (LightGBM), or use **feature subsampling** that always includes key features.

### LightGBM `forced_features`

```python
# Always use lag1, lag7, seasonal lags
forced = ["price_per_kg_mean_lag1", "price_per_kg_mean_lag7",
          "price_per_kg_mean_lag364", "price_per_kg_mean_lag365"]
lgb_params = {
    "feature_fraction": 0.8,
    "forcedsplits": None,  # or provide split file
}
# Alternative: ensure these are in top-N features from selection
```

### CatBoost `per_object_feature_penalties`

```python
# Penalize less important features (or boost important ones via interaction)
model = CatBoostRegressor(
    per_object_feature_penalties={
        "price_per_kg_mean_lag1": 0.0,   # no penalty = high importance
        "price_per_kg_mean_lag365": 0.0,
    }
)
```

---

## 3. Seasonal / Dummy Features

**Problem**: Model may not capture seasonal patterns (summer low, winter high).

**Solution**: Add explicit seasonal features.

```python
# In feature engineering
df["month"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.month
df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
```

Include these in `final_features` or feature selection.

---

## 4. Robust Target (Reduce Outlier Impact)

**Problem**: MAPE is sensitive to small actual values (e.g., actual 1385 → predicted 3196 gives 130% error).

**Solution**:

- Use **sMAPE** or **log-MAPE** as the main metric for tuning.
- Clip extreme predictions: `pred = np.clip(pred, actual * 0.5, actual * 2.0)` (post-hoc).
- Use **Quantile Regression** instead of MSE to model median and reduce outlier influence.

---

## 5. Season-Specific Models

**Problem**: Single model struggles with all seasons.

**Solution**: Train separate models per season, then ensemble.

```python
# Train spring_model, summer_model, fall_model, winter_model
# At inference: select model by month
```

---

## 6. Post-Processing Bounds

**Problem**: Predictions sometimes far outside reasonable range.

**Solution**: Clip predictions using historical percentiles.

```python
# After prediction
lo = np.percentile(train_prices, 1)
hi = np.percentile(train_prices, 99)
pred = np.clip(pred, lo, hi)
```

---

## 7. Recommended Priority

| Priority | Action | Expected Effect |
|----------|--------|-----------------|
| 1 | Add `sample_weight` for summer/winter | Summer MAPE -5~10%p |
| 2 | Add `month_sin`, `month_cos`, `is_summer` | Better seasonal fit |
| 3 | Post-hoc clip predictions (e.g., ±50% of lag1) | Reduce extreme errors |
| 4 | Season-specific models | Further 3~5%p |
| 5 | Quantile regression / sMAPE loss | More robust to outliers |

---

## 8. Quick Config Changes

In `config/training_config.yaml`:

```yaml
# Increase iterations for tree models
lgb:
  n_estimators: 1000
catboost:
  iterations: 1000

# Add more regularization to reduce overfitting
lgb:
  min_data_in_leaf: 40
  lambda_l1: 1
  lambda_l2: 1
```
