# Retail Demand Forecasting: A Comparative Machine Learning Study

## Project overview

This project compares machine-learning models and a seasonal-naive baseline for daily retail demand forecasting using a reproducible, category-stratified sample of 50 product-store series from the M5 Forecasting Accuracy dataset.

The goal is to demonstrate a practical forecasting workflow: preparing daily sales data, joining calendar and price information, engineering lag-based features, evaluating forecasts chronologically, and interpreting model performance under two different forecasting settings.

## Dataset and sample

- Source: Kaggle M5 Forecasting Accuracy dataset (Walmart retail sales).
- Sample: 50 product-store series selected deterministically across FOODS, HOBBIES, and HOUSEHOLD.
- Sample composition: 17 FOODS, 17 HOBBIES, and 16 HOUSEHOLD series.
- Daily observations per series: 1,941 in the source sales table.
- Missing sell prices are flagged and imputed using medians learned from the training data; sales observations are retained.

This is a sample-based portfolio experiment, not an analysis of the full M5 dataset.

## Models

- Random Forest Regressor
- LightGBM Regressor
- XGBoost Regressor
- Seasonal-naive baseline using sales from seven days earlier

## Features

The notebook creates within-series lag features (1, 7, 14, and 28 days), lagged rolling means (7 and 28 days), calendar variables, SNAP indicators, sell price, and a missing-price flag. The chronological split and feature construction are designed to prevent using a day's target value to construct its own predictors.

## Evaluation designs

### 1. Rolling one-step-ahead evaluation

The test period runs from 18 October 2014 to 22 May 2016. For each test date, the model predicts using features that can include actual sales observed on earlier test dates. This represents a rolling operational setting in which new actual sales become available over time.

### 2. Fixed-origin 28-day evaluation

The forecast origin is 24 April 2016, and the forecast horizon is 25 April to 22 May 2016. Models are fitted using data before the forecast horizon and then forecast recursively. Within-horizon lag features use each model's own prior predictions, not actual sales from the forecast horizon. Future calendar and scheduled price fields are treated as known covariates.

The two evaluation designs answer different questions; their metric values should not be compared as if they were the same test task.

## Saved results from the executed notebook

### Rolling one-step-ahead

| Model | MAE | RMSE | SMAPE (%) | RMSLE |
|---|---:|---:|---:|---:|
| Random Forest | 0.7881 | 1.5082 | 149.9490 | 0.4782 |
| LightGBM | 0.7754 | 1.5146 | 150.5699 | 0.4714 |
| XGBoost | 0.7966 | 1.5367 | 150.6044 | 0.4818 |
| Seasonal Naive (lag 7) | 0.9398 | 2.1682 | 74.9930 | 0.6205 |

### Fixed-origin 28-day forecast

| Model | MAE | RMSE | SMAPE (%) | RMSLE | WAPE (%) |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.9837 | 1.8728 | 137.4443 | 0.5349 | 91.5686 |
| Random Forest | 0.9442 | 1.9007 | 140.6977 | 0.5219 | 87.8929 |
| XGBoost | 0.9275 | 1.9405 | 142.5330 | 0.5175 | 86.3393 |
| Seasonal Naive (lag 7) | 1.1071 | 2.2329 | 90.0876 | 0.6765 | 103.0585 |

Lower values indicate better performance for all metrics shown. The best model depends on the metric and forecast setting: Random Forest has the lowest rolling RMSE, LightGBM the lowest fixed-origin RMSE, and XGBoost the lowest fixed-origin MAE and WAPE. The seasonal-naive baseline's much lower SMAPE alongside higher MAE/RMSE should be explicitly discussed, particularly because zero-sales days are common.

These numbers are copied from the notebook's saved outputs and have not been independently reproduced outside the original Kaggle run.

## Limitations

- Only 50 product-store series are included; results may not generalize to all M5 items and stores.
- The models are compared using one chronological split rather than multiple rolling-origin backtests.
- The selected models and hyperparameters are not the result of a comprehensive tuning study.
- Price and calendar information during the fixed-origin horizon are treated as known. This assumption should be stated when interpreting the results.
- Metrics can tell different stories on intermittent, zero-heavy demand series. MAE, RMSE, WAPE, RMSLE, and the exact SMAPE definition should be reported together rather than relying on a single metric.

## How to run

1. Open the notebook in Kaggle.
2. Attach the M5 Forecasting Accuracy dataset.
3. Confirm that `DATA_PATH` points to the attached dataset directory and contains `calendar.csv`, `sales_train_evaluation.csv`, and `sell_prices.csv`.
4. Run all cells from top to bottom.

The notebook requires pandas, NumPy, Matplotlib, scikit-learn, and (for the corresponding model comparisons) LightGBM and XGBoost.

## Suggested citation

Cite the M5 Forecasting Accuracy competition and dataset using the citation or dataset reference provided on its Kaggle page. Do not redistribute the raw dataset in this repository unless its terms permit it.
