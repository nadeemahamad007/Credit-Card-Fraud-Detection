# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Models-F7931E?style=for-the-badge&logo=scikitlearn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Machine Learning project focused on Credit Card Fraud Detection, transaction risk classification, fraud pattern analysis, and model evaluation using Python.
An end-to-end machine learning project that identifies fraudulent credit card transactions from highly imbalanced transactional data. This repository now includes both the original notebook and a clean Python training pipeline so the project is easier to understand, reproduce, and showcase in a portfolio.

## Why this project matters

Credit card fraud detection is a real-world classification problem where the positive class is rare but very costly. That makes model selection and evaluation more important than raw accuracy alone. In this project, the focus is on building a practical fraud detection workflow and comparing multiple models on fraud-specific metrics.

## Project highlights

- Cleaned the original notebook workflow into a reusable training script.
- Compared Logistic Regression, Decision Tree, and Random Forest models.
- Used preprocessing pipelines with imputation, scaling, and one-hot encoding.
- Handled class imbalance with class-weighted models.
- Saved evaluation artifacts such as confusion matrices, metrics, and the best model.

## Dataset

Source: [Fraud Detection Dataset on Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

Files used:

- `fraudTrain.csv`
- `fraudTest.csv`

The raw dataset is not stored in this repository because of file size. Setup steps are available in [data/README.md](data/README.md).

## Tech stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Project structure

```text
.
|-- CREDIT CARD FRAUD  DETECTION.ipynb
|-- train_model.py
|-- requirements.txt
|-- data/
|   `-- README.md
`-- artifacts/   # generated after training
```

## Features used

After removing personally identifiable or low-value columns, the project trains on a mix of transaction and customer context features such as:

- `amt`
- `merchant`
- `category`
- `gender`
- `state`
- `zip`
- `lat`, `long`
- `city_pop`
- `unix_time`
- `merch_lat`, `merch_long`

## Model pipeline

1. Load `fraudTrain.csv` as the training set and `fraudTest.csv` as the holdout test set.
2. Drop identifier and privacy-sensitive fields.
3. Impute missing values if present.
4. Scale numeric features and one-hot encode categorical features.
5. Train and compare three baseline models.
6. Evaluate using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
7. Save the best model and result artifacts.

## Sample results

These results were produced from the current cleaned workflow on the provided train/test split:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.9782 | 0.1210 | 0.7403 | 0.2080 | 0.9325 |
| Decision Tree | 0.9557 | 0.0760 | 0.9389 | 0.1407 | 0.9686 |
| Logistic Regression | 0.9245 | 0.0366 | 0.7338 | 0.0698 | 0.8911 |

Note: Because fraud detection is extremely imbalanced, accuracy alone is misleading. Recall, precision, F1-score, and ROC-AUC are more useful for comparing model quality.

## How to run

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the Kaggle dataset and place the CSV files in the `data/` folder.
4. Run the training script:

```bash
python train_model.py
```

Optional:

```bash
python train_model.py --train-path "data/fraudTrain.csv" --test-path "data/fraudTest.csv" --output-dir "artifacts"
```

## Outputs

After running the script, the `artifacts/` folder will contain:

- `model_metrics.csv`
- `metrics_summary.json`
- `best_fraud_model.joblib`
- confusion matrix plots
- top feature importance files for the best tree-based model

## What improved in this version

- Removed machine-specific dataset paths from the workflow.
- Replaced repeated notebook-only experimentation with a reusable script.
- Made the evaluation process easier for recruiters and reviewers to follow.
- Added reproducible output artifacts instead of only notebook cells.

## Future improvements

- Add cross-validation and threshold tuning for better fraud recall/precision trade-offs.
- Try gradient boosting models such as XGBoost or LightGBM.
- Package the best model behind a Streamlit or Flask demo app.

## Author

Nadeem Ahamad

Machine Learning project focused on Credit Card Fraud Detection, transaction risk classification, fraud pattern analysis, and model evaluation using Python by CodSoft Internship Project.
