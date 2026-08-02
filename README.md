# Loan Default Prediction

This project compares Logistic Regression, a Decision Tree and an Artificial Neural Network for predicting loan default.

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the project

Run the complete project with:

```bash
python run_all.py
```

The `run_all.py` script runs the three project steps in order:

1. `01_eda.py` creates the exploratory-analysis figures.
2. `02_modeling.py` prepares the data, trains the models and saves the evaluation results.
3. `03_generate_report.py` combines the results and figures into `Loan_Default_Prediction_Report.pdf`.

