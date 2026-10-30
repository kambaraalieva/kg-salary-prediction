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


# ---------------------------------------------------------------- level (notebook 07)
LEVELS = [
    ("lead", r"\b(lead|тимлид|team ?lead|ведущ\w*|head|руководител\w*|principal|architect|архитектор)\b"),
    ("senior", r"\b(senior|сеньор|синьор|старш\w*|sr)\b"),
    ("middle", r"\b(middle|мидл|mid)\b"),
    ("junior", r"\b(junior|джун\w*|младш\w*|jr|стаж[её]р\w*|intern|trainee|ассистент)\b"),
]
LEVEL_FROM_EXPERIENCE = {0: "junior", 1: "middle", 3: "senior", 6: "lead"}


def title_level(title):
    """Level written in the job title (Senior, Junior...), or None."""
    t = str(title).lower()
    for name, pattern in LEVELS:
        if re.search(pattern, t):
            return name
    return None


def add_level(df):
    """level: from the title if it is there, otherwise from the required experience.
    cat_exp / cat_level: sector x experience and sector x level, so that experience can be
    worth more in IT than in retail."""
    df["level_in_title"] = df["name"].map(title_level)
    # experience_start = -1 means «not stated»: then the level is unknown too, nothing is assumed
    df["level"] = df["level_in_title"].fillna(df["experience_start"].map(LEVEL_FROM_EXPERIENCE)).fillna("не указан")
    df["cat_exp"] = df["category"].astype(str) + "|" + df["experience_start"].astype(str)
    df["cat_level"] = df["category"].astype(str) + "|" + df["level"]
    return df


# ---------------------------------------------------------------- master dataset (kg-job-market-analysis, notebook 09)
MASTER = "../data/vacancies_2024_2026_master.csv.gz"
SOURCE_KEY = {"hh.kg": "headhunter", "lalafo.kg": "lalafo",
              "Telegram @codifynews": "telegram_codifynews", "Telegram @kgworkzone": "telegram_kgworkzone"}


def load_master(path=MASTER):
    """Master dataset in the column names the models use.

    Empty values stay empty in the data. Inside the model only:
      experience_start = -1 and experience_known = 0 when experience is not stated,
      city / education / gender / schedule = «не указано».
    Text for the model: title + additional_info (everything in the ad except the salary line and the employer line),
    with digits and currency words removed (see strip_money)."""
    m = pd.read_csv(path, low_memory=False, parse_dates=["reference_date", "published_at"])
    df = pd.DataFrame({
        "id": m["vacancy_id"], "link": m["url"], "published": m["reference_date"],
        "source": m["source"].map(SOURCE_KEY), "name": m["job_title"], "company": m["employer"],
        "category": m["sector"], "sub_category": m["profession"],
        "city": m["city_group"], "area": m["locality"],
        "experience_start": m["experience_min_years"].fillna(-1).astype(int),
        "experience_known": m["experience_min_years"].notna().astype(int),
        "education": m["education_required"], "gender_req": m["gender_required"], "schedule": m["schedule"],
        "remote": (m["work_format"] == "удалённо или гибрид"), "commission": m["commission_or_bonus"].astype(bool),
        "skills": m["skills"], "requirements": m["additional_info"], "responsibilities": "",
        "job_type": m["work_format_raw"], "extra": "",
        "compensation_from": m["salary_from"], "compensation_to": m["salary_to"], "compensation_currency": m["currency"],
        "salary_status": m["salary_status"], "salary_kgs": m["salary_month_kgs"],
        "times_published": m["times_published"],
    })
    df["has_salary"] = df["salary_kgs"].notna()
    df["salary_hidden"] = df["salary_status"] == "скрыта"
    text = (df["name"].fillna("") + " " + df["requirements"].fillna("")).str.lower()
    df["kyrgyz_text"] = text.str.contains("[өүң]")
    df["n_skills"] = df["skills"].fillna("").str.split(",").map(lambda xs: len([x for x in xs if x.strip()]))
    df["text_len"] = df["requirements"].fillna("").str.len()
    for c in CAT_FEATURES:
        df[c] = df[c].fillna("не указано").astype(str)
    for c in BOOL_FEATURES:
        df[c] = df[c].astype(int)
    clean = lambda s: strip_money(s.fillna("").str.lower())
    df[TEXT_FEATURE] = clean(df["name"]) + ". " + clean(df["requirements"])
    df["title"] = clean(df["name"])
    df["full"] = df["title"] + ". " + clean(df["requirements"])
    norm = lambda s: re.sub(r"[^а-яёa-z]+", " ", str(s).lower()).strip()
    df["dup_group"] = (df["name"].map(norm) + "|" + df["requirements"].fillna("").map(norm).str.slice(0, 80))
    return df
