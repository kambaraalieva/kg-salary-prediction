"""
Shared data preparation for all notebooks.

Input:  data/vacancies_clean.csv  (output of kg-job-market-analysis/notebooks/02_cleaning)
Target: log(salary_kgs), monthly salary in som.

Only features that also exist for vacancies WITHOUT a salary are used,
otherwise the model could not be applied to hidden salaries (data leakage).
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

DATA = "../data/vacancies_clean.csv"
SEED = 42

CAT_FEATURES = ["category", "sub_category", "city", "source", "education",
                "gender_req", "schedule"]
NUM_FEATURES = ["experience_start", "n_skills", "text_len"]
BOOL_FEATURES = ["remote", "commission", "kyrgyz_text"]
TEXT_FEATURE = "text"            # name + requirements, for text models

# Columns that exist only when a salary is shown: never use as features.
LEAKY = ["compensation_from", "compensation_to", "compensation_currency",
         "salary_type", "pay_period", "salary_outlier", "salary_hidden", "has_salary"]


MONEY_WORDS = r"\b(сом\w*|тыс\w*|т\.?с\.?|usd|долл\w*|\$|руб\w*|евро|€|kgs|kzt|тенге)\b"


def strip_money(s):
    """Remove digits and currency words from a lowercased text Series."""
    s = s.str.replace(r"\d+", " ", regex=True)
    s = s.str.replace(MONEY_WORDS, " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def load(path=DATA):
    df = pd.read_csv(path)
    for c in CAT_FEATURES:
        df[c] = df[c].fillna("не указано").astype(str)
    for c in BOOL_FEATURES:
        df[c] = df[c].astype(int)
    # Text without numbers and currency words: ads with a visible salary often repeat it in the
    # text ("зарплата 30 000 сом"), ads with a hidden salary do not. Keeping the numbers would let
    # the model read the answer from the text (leakage).
    clean = lambda s: strip_money(s.fillna("").str.lower())
    df[TEXT_FEATURE] = clean(df["name"]) + ". " + clean(df["requirements"])
    # richer text used from notebook 03 on: title alone + title, requirements and responsibilities
    df["title"] = clean(df["name"])
    df["full"] = df["title"] + ". " + clean(df["requirements"]) + ". " + clean(df["responsibilities"])
    # group key for near-duplicates: same title + same beginning of the text
    norm = lambda s: re.sub(r"[^а-яёa-z]+", " ", str(s).lower()).strip()
    df["dup_group"] = (df["name"].map(norm) + "|" + df["requirements"].fillna("").map(norm).str.slice(0, 80))
    return df


def labeled(df, verbose=False):
    """Vacancies with a plausible salary: these train and test the model.

    Extra label-noise filter for lalafo: a salary more than 5x above or below the median
    of its profession is dropped (500 000 for a seamstress, 10 000 'salary' that is a taxi fee,
    business-for-sale ads). hh is not filtered: there such gaps are plausible
    (e.g. a senior developer paid in USD).
    """
    d = df[df["has_salary"]].copy()
    ratio = d["salary_kgs"] / d.groupby("sub_category")["salary_kgs"].transform("median")
    noise = ((ratio > 5) | (ratio < 0.2)) & (d["source"] == "lalafo")
    if verbose:
        print(f"Убрано как шум в разметке (lalafo, зарплата в 5+ раз от медианы профессии): {noise.sum()}")
    d = d[~noise]
    d["y"] = np.log(d["salary_kgs"])
    return d.reset_index(drop=True)


def hidden(df):
    """Vacancies where the employer hid the salary: the model predicts these."""
    return df[df["salary_hidden"]].copy().reset_index(drop=True)


def split(d, test_size=0.2, seed=SEED):
    """Train/test split that keeps near-duplicate ads in the same part."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(d, groups=d["dup_group"]))
    return d.iloc[tr].reset_index(drop=True), d.iloc[te].reset_index(drop=True)


def cv_folds(d, n_splits=5):
    """5-fold CV on the training part, grouped the same way."""
    return list(GroupKFold(n_splits=n_splits).split(d, groups=d["dup_group"]))


def metrics(y_true_log, y_pred_log):
    """Errors in som: the model predicts log(salary), report in som per month."""
    yt, yp = np.exp(y_true_log), np.exp(y_pred_log)
    err = np.abs(yt - yp)
    ss_res = ((y_true_log - y_pred_log) ** 2).sum()
    ss_tot = ((y_true_log - y_true_log.mean()) ** 2).sum()
    return {
        "MAE, сом": err.mean(),
        "медианная ошибка, сом": np.median(err),
        "MAPE, %": (err / yt).mean() * 100,
        "доля ошибок < 20%": (err / yt < 0.2).mean(),
        "R² (log)": 1 - ss_res / ss_tot,
    }
