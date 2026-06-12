# Bank Marketing Subscription Prediction

Predict whether a client will subscribe to a term deposit using the UCI Bank Marketing dataset.

## Dataset
- 45,211 rows, 17 columns (10 categorical, 7 numerical)
- Target: `y` — binary (yes/no), 88.3% / 11.7% imbalanced
- Missing values as `"unknown"` in 4 columns (job, education, contact, poutcome)

## Project Structure
```
├── data/            # bank-full.csv, train.csv, test.csv
├── notebooks/       # eda.ipynb
├── src/             # preprocessing.py, model.py, utils.py
├── requirements.txt
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Results

| Model | CV F1 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| LogisticRegression (baseline) | 0.5507 | 0.8460 | 0.4186 | 0.8138 | 0.5528 | 0.9078 |
| GradientBoosting + GridSearch | 0.5472 | 0.9077 | 0.6469 | 0.4641 | 0.5405 | 0.9301 |
| XGBoost + GridSearch | 0.5451 | 0.9091 | 0.6625 | 0.4546 | 0.5392 | 0.9333 |
| RandomForest + GridSearch | 0.4906 | 0.9045 | 0.6554 | 0.3866 | 0.4863 | 0.9272 |
| DecisionTree | 0.4743 | 0.8777 | 0.4783 | 0.4991 | 0.4884 | 0.7135 |
| AdaBoost + GridSearch | 0.4554 | 0.8987 | 0.6199 | 0.3469 | 0.4448 | 0.9073 |

Top features: duration, poutcome_success, balance, age, day, pdays, campaign, housing_yes, contact_unknown, previous.

**Note:** `duration` is included but causes data leakage (known only after call ends). Will be dropped for final model.

## Run
```bash
python run.py
```
