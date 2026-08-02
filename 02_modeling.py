"""
In this file, I will be preparing data for modeling, training and evaluation of the three classifiers (logistic regression, decision tree and ANN) on the Credit Risk dataset.

I will be showing the confusion matrices, ROC curves and a metric comparison table for the three mention models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

DATA = "credit_risk_dataset.csv"
RANDOM_STATE = 42
PAL = ["#4C72B0", "#DD8452"]

plt.rcParams["figure.dpi"] = 110
sns.set_style("whitegrid")


def load_and_prepare():
    """Clean, encode and split. Imputation/scaling are fitted on TRAIN only."""
    df = pd.read_csv(DATA)

    n0 = len(df)
    df = df.drop_duplicates()
    df = df[(df["person_age"] <= 100) &
            ((df["person_emp_length"] <= 60) | df["person_emp_length"].isna())]
    print(f"Rows: {n0} -> {len(df)} (removed {n0 - len(df)})")

    df["loan_grade"] = df["loan_grade"].map({g: i for i, g in enumerate("ABCDEFG")})
    df["cb_person_default_on_file"] = (df["cb_person_default_on_file"] == "Y").astype(int)
    df = pd.get_dummies(df, columns=["person_home_ownership", "loan_intent"], drop_first=True)

    y = df["loan_status"]
    X = df.drop(columns="loan_status")
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    grade_med = Xtr.groupby("loan_grade")["loan_int_rate"].median()
    glob_rate = Xtr["loan_int_rate"].median()
    emp_med = Xtr["person_emp_length"].median()
    for part in (Xtr, Xte):
        part["loan_int_rate"] = (part["loan_int_rate"]
                                 .fillna(part["loan_grade"].map(grade_med))
                                 .fillna(glob_rate))
        part["person_emp_length"] = part["person_emp_length"].fillna(emp_med)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    return X.columns, Xtr, Xte, ytr, yte, Xtr_s, Xte_s


def train_models(Xtr, ytr, Xtr_s):
    logreg = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(Xtr_s, ytr)
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=RANDOM_STATE).fit(Xtr, ytr)
    ann = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", max_iter=400,
                        early_stopping=True, random_state=RANDOM_STATE).fit(Xtr_s, ytr)
    return logreg, tree, ann


def evaluate(models, yte):
    rows, cms, rocs = [], {}, {}
    for name, (m, Xt) in models.items():
        pred = m.predict(Xt)
        proba = m.predict_proba(Xt)[:, 1]
        rows.append([name, accuracy_score(yte, pred), precision_score(yte, pred),
                     recall_score(yte, pred), f1_score(yte, pred), roc_auc_score(yte, proba)])
        cms[name] = confusion_matrix(yte, pred)
        rocs[name] = roc_curve(yte, proba)
    res = pd.DataFrame(
        rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    ).set_index("Model")
    return res, cms, rocs


def plot_confusion(cms):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for i, (name, cm) in enumerate(cms.items()):
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax[i], cbar=False,
                    xticklabels=["Good", "Default"], yticklabels=["Good", "Default"])
        ax[i].set_title(name); ax[i].set_xlabel("Predicted"); ax[i].set_ylabel("Actual")
    plt.suptitle("Confusion matrices (test set)", fontweight="bold")
    plt.tight_layout(); plt.savefig("model_confusion.png", bbox_inches="tight"); plt.close()


def plot_roc(rocs, res):
    plt.figure(figsize=(6, 5.5))
    for name, (fpr, tpr, _) in rocs.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={res.loc[name, 'ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC curves"); plt.legend()
    plt.tight_layout(); plt.savefig("model_roc.png", bbox_inches="tight"); plt.close()


def main():
    cols, Xtr, Xte, ytr, yte, Xtr_s, Xte_s = load_and_prepare()
    logreg, tree, ann = train_models(Xtr, ytr, Xtr_s)

    models = {
        "Logistic Regression": (logreg, Xte_s),
        "Decision Tree": (tree, Xte),
        "ANN (MLP)": (ann, Xte_s),
    }
    res, cms, rocs = evaluate(models, yte)

    print("\n=== RESULTS (test set) ===")
    print(res.round(4).to_string())

    plot_confusion(cms)
    plot_roc(rocs, res)
    res.round(4).to_csv("results.csv")

    fi = pd.Series(tree.feature_importances_, index=cols).sort_values(ascending=False)
    print("\nTop tree feature importances:")
    print(fi.head(8).round(3).to_string())
    print("\nReadable decision rules (depth 3):")
    shallow = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE).fit(Xtr, ytr)
    print(export_text(shallow, feature_names=list(cols), max_depth=3))

    print("Saved: model_confusion.png, model_roc.png, results.csv")


if __name__ == "__main__":
    main()
