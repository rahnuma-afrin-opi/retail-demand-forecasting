# Retail Demand Forecasting: A Comparative Machine Learning Study

## Project Overview

This project compares three machine-learning models with a seasonal-naive baseline for daily retail demand forecasting using a reproducible, category-stratified sample of 50 product-store series from the [M5 Forecasting Accuracy dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy).

The project demonstrates an end-to-end forecasting workflow, including sales-data preparation, calendar and price integration, lag-based feature engineering, chronological evaluation, and model comparison under two forecasting settings.

**[View the executed Jupyter notebook](retail-demand-forecasting-portfolio-v2.ipynb)**

## Dataset and Sample

- **Source:** [M5 Forecasting Accuracy — Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
- **Retailer:** Walmart
- **Sample:** 50 product-store time series selected deterministically across three product categories.
- **Category distribution:** 17 FOODS, 17 HOBBIES, and 16 HOUSEHOLD series.
- **Observation period:** February 26, 2011 – May 22, 2016.
- **Daily observations:** 1,941 per series in the source sales table.

Missing sell prices are flagged and imputed using medians learned from the training data. Sales observations are retained.

This is a sample-based portfolio experiment, not an analysis of the complete M5 dataset. Results should therefore be interpreted within the scope of the selected series.

## Models Compared

1. Random Forest Regressor
2. LightGBM Regressor
3. XGBoost Regressor
4. Seasonal-naive baseline using sales from seven days earlier

## Feature Engineering

The notebook constructs features including:

- Within-series sales lags: 1, 7, 14, and 28 days
- Lagged rolling means: 7 and 28 days
- Calendar variables
- SNAP indicators
- Sell prices
- A missing-price indicator

The chronological evaluation design is intended to prevent a day's target value from being used to construct that same day's predictors.

## Evaluation Methodology

### 1. Rolling One-Step-Ahead Evaluation

The test period runs from October 18, 2014, to May 22, 2016.

For each test date, the model predicts demand using features that may include actual sales observed on earlier test dates. This represents a rolling operational setting in which new actual sales become available over time.

### 2. Fixed-Origin 28-Day Forecast

The forecast origin is April 24, 2016, and the forecast horizon is April 25–May 22, 2016.

Models are fitted using data before the forecast horizon and then forecast recursively. Within-horizon lag features use each model's own previous predictions rather than actual sales from the forecast horizon. Future calendar and scheduled price fields are treated as known covariates.

**These evaluation designs address different forecasting tasks.** Their metric values should not be compared as if they came from the same test setting.

## Results

The following tables report the saved results from the executed notebook.

### Rolling One-Step-Ahead Evaluation

| Model | MAE | RMSE | SMAPE (%) | RMSLE |
|---|---:|---:|---:|---:|
| Random Forest | 0.7881 | 1.5082 | 149.9490 | 0.4782 |
| LightGBM | 0.7754 | 1.5146 | 150.5699 | 0.4714 |
| XGBoost | 0.7966 | 1.5367 | 150.6044 | 0.4818 |
| Seasonal Naive (lag 7) | 0.9398 | 2.1682 | 74.9930 | 0.6205 |

### Fixed-Origin 28-Day Forecast

| Model | MAE | RMSE | SMAPE (%) | RMSLE | WAPE (%) |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.9837 | 1.8728 | 137.4443 | 0.5349 | 91.5686 |
| Random Forest | 0.9442 | 1.9007 | 140.6977 | 0.5219 | 87.8929 |
| XGBoost | 0.9275 | 1.9405 | 142.5330 | 0.5175 | 86.3393 |
| Seasonal Naive (lag 7) | 1.1071 | 2.2329 | 90.0876 | 0.6765 | 103.0585 |

Lower values indicate better performance for all metrics shown.

### Key Findings

- **Rolling one-step-ahead:** Random Forest has the lowest RMSE, while LightGBM has the lowest MAE and RMSLE among the machine-learning models.
- **Fixed-origin forecasting:** XGBoost has the lowest MAE and WAPE, while LightGBM has the lowest RMSE. XGBoost also has the lowest RMSLE.
- **Baseline comparison:** The seasonal-naive baseline has higher MAE and RMSE than the machine-learning models in both evaluation settings, but substantially lower reported SMAPE.
- **Metric sensitivity:** The differences in model rankings show why forecasting performance should be assessed using multiple metrics rather than a single score.

The seasonal-naive baseline's lower SMAPE despite its higher MAE and RMSE deserves particular attention because retail demand can contain many zero-sales observations. SMAPE interpretation depends on the exact formula and treatment of zero denominators, so its implementation should be checked before drawing conclusions from this metric.

These results are taken from the notebook's saved outputs. They have not been independently reproduced outside the Kaggle execution.

## Limitations

- Only 50 product-store series are included; the findings may not generalize to the complete M5 dataset.
- The evaluation uses a chronological split rather than multiple rolling-origin backtests.
- The selected models and hyperparameters do not represent an exhaustive tuning study.
- The fixed-origin evaluation assumes future calendar and scheduled price information is available.
- Intermittent and zero-heavy demand can produce different model rankings across metrics. MAE, RMSE, WAPE, RMSLE, and a clearly defined SMAPE should be considered together.
- The findings are specific to the selected series, features, models, and evaluation designs.

## How to Run the Notebook

1. Open the [Jupyter notebook in this repository](retail-demand-forecasting-portfolio-v2.ipynb).
2. To execute it, open or import the notebook in Kaggle.
3. Attach the [M5 Forecasting Accuracy dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy).
4. Confirm that `DATA_PATH` points to the directory containing `calendar.csv`, `sales_train_evaluation.csv`, and `sell_prices.csv`.
5. Run all cells from top to bottom.

The notebook uses Python libraries including pandas, NumPy, Matplotlib, scikit-learn, LightGBM, and XGBoost.

## Tools and Technologies

Python · pandas · NumPy · scikit-learn · LightGBM · XGBoost · Matplotlib · Jupyter Notebook · Kaggle

## Dataset Citation

The dataset is from the [M5 Forecasting Accuracy competition on Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy). Refer to the competition page for the dataset citation and applicable terms of use.

Raw dataset files are not included in this repository.

---

*This project was developed as a machine-learning portfolio project focused on retail demand forecasting, chronological evaluation, and model comparison.*

