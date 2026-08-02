"""
In this file, I will be trying to understand the data and perform some exploratory data analysis (EDA) to get a better understanding of the dataset.
I will configure few pictures to visualize the data and understand the relationships:

    eda_1_target.png       - target class balance
    eda_2_numeric.png      - numeric feature distributions
    eda_3_categorical.png  - default rate by categorical feature
    eda_4_drivers.png      - strongest numeric drivers + correlation heatmap

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA = "credit_risk_dataset.csv"
PAL = ["#4C72B0", "#DD8452"]

plt.rcParams["figure.dpi"] = 110
sns.set_style("whitegrid")


def main():
    df = pd.read_csv(DATA)
    print("Shape:", df.shape)

    print("\nMissing values:")
    print(df.isna().sum()[df.isna().sum() > 0])
    print("\nTarget balance (%):")
    print((df["loan_status"].value_counts(normalize=True) * 100).round(2))
    print("Duplicate rows:", df.duplicated().sum())

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    vc = df["loan_status"].value_counts().sort_index()
    ax[0].bar(["Good (0)", "Default (1)"], vc.values, color=PAL)
    for i, v in enumerate(vc.values):
        ax[0].text(i, v + 300, f"{v:,}", ha="center", fontweight="bold")
    ax[0].set_title("Target counts"); ax[0].set_ylabel("applicants")
    ax[1].pie(vc.values, labels=["Good 78.2%", "Default 21.8%"], colors=PAL,
              startangle=90, wedgeprops=dict(width=0.45))
    ax[1].set_title("Class balance")
    plt.suptitle("Target: loan_status", fontweight="bold")
    plt.tight_layout(); plt.savefig("eda_1_target.png", bbox_inches="tight"); plt.close()

    num = ["person_age", "person_income", "person_emp_length", "loan_amnt",
           "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for i, c in enumerate(num):
        d = df[c].dropna()
        if c == "person_income":
            axes.flat[i].hist(np.log10(d), bins=40, color=PAL[0])
            axes.flat[i].set_xlabel("log10(income)")
        else:
            axes.flat[i].hist(d, bins=40, color=PAL[0])
        axes.flat[i].set_title(c, fontsize=9)
    axes.flat[-1].axis("off")
    plt.suptitle("Numeric feature distributions "
                 "(note age & emp_length tails = data errors)", fontweight="bold")
    plt.tight_layout(); plt.savefig("eda_2_numeric.png", bbox_inches="tight"); plt.close()

    cats = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for i, c in enumerate(cats):
        g = df.groupby(c)["loan_status"].mean().sort_values(ascending=False) * 100
        axes.flat[i].bar(g.index, g.values, color=PAL[1])
        for j, v in enumerate(g.values):
            axes.flat[i].text(j, v + 0.5, f"{v:.0f}%", ha="center", fontsize=8)
        axes.flat[i].axhline(21.82, color="gray", ls="--", lw=1, label="overall 21.8%")
        axes.flat[i].set_title(f"Default rate by {c}", fontsize=10)
        axes.flat[i].set_ylabel("% default"); axes.flat[i].legend(fontsize=7)
        axes.flat[i].tick_params(axis="x", rotation=30)
    plt.suptitle("How default rate varies across categories", fontweight="bold")
    plt.tight_layout(); plt.savefig("eda_3_categorical.png", bbox_inches="tight"); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col in [(axes[0], "loan_percent_income"), (axes[1], "loan_int_rate")]:
        tmp = df.dropna(subset=[col]).copy()
        tmp["bin"] = pd.qcut(tmp[col], 6, duplicates="drop")
        g = tmp.groupby("bin", observed=True)["loan_status"].mean() * 100
        ax.plot(range(len(g)), g.values, "o-", color=PAL[0])
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels([f"{iv.mid:.2f}" for iv in g.index], rotation=30, fontsize=8)
        ax.set_title(f"Default rate vs {col}", fontsize=10); ax.set_ylabel("% default")
    corr = df[num + ["loan_status"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[2],
                cbar=False, annot_kws={"size": 7})
    axes[2].set_title("Correlation (numeric + target)", fontsize=10)
    plt.suptitle("Strongest numeric relationships with default", fontweight="bold")
    plt.tight_layout(); plt.savefig("eda_4_drivers.png", bbox_inches="tight"); plt.close()

    print("\nSaved: eda_1_target.png, eda_2_numeric.png, eda_3_categorical.png, eda_4_drivers.png")


if __name__ == "__main__":
    main()
