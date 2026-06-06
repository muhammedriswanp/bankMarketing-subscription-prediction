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
