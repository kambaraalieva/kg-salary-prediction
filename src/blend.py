"""
Data loading for 2024-2026 and the blended model of notebook 08:
Ridge on TF-IDF text + one-hot features, and LightGBM on target-encoded title / employer,
mixed as  w * LightGBM + (1 - w) * Ridge  on log(salary).
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import prep

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RIDGE_CAT = prep.CAT_FEATURES + ["company", "level", "cat_exp", "cat_level", "fmt"]
RIDGE_NUM = prep.NUM_FEATURES + prep.BOOL_FEATURES + ["experience_known", "month_n"]
TE_COLS = ["tnorm", "company", "sub_category", "cat_level"]
GBM_CAT = ["category", "level", "source", "fmt", "city"]
GBM_NUM = ["experience_start", "month_n", "remote", "experience_known", "commission"]


def _norm(s):
    return s.fillna("").astype(str).str.lower().str.replace(r"[^а-яёa-z0-9]+", " ", regex=True).str.strip()


def add_fields(df):
    """Columns built from what a vacancy shows, also when the salary is hidden."""
    for c in ["job_type", "extra"]:
        if c not in df:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str).str.lower()
    df["fmt"] = df["job_type"].str.split(",").str[0].str.strip().replace("", "не указан")
    clean = lambda s: prep.strip_money(s.fillna("").astype(str).str.lower())
    df["full2"] = df["full"] + ". " + clean(df["job_type"]) + ". " + clean(df["extra"])
    df["tnorm"] = _norm(df["name"])
    return df


def load_2026(path=os.path.join(ROOT, "data", "vacancies_2024_2026_master.csv.gz"), verbose=True):
    """All vacancies since 2024 (df) and the labeled, de-duplicated part (d), from the master dataset."""
    df = prep.load_master(path)
    df = df[df["published"] >= "2024-01-01"].copy()
    df["company"] = df["company"].fillna("не указано").astype(str).str.lower().str.strip()
    df["month_n"] = (df["published"].dt.year - 2024) * 12 + df["published"].dt.month - 1
    df = prep.add_level(df)
    df = add_fields(df)

    d = df[df["has_salary"]].copy()
    ratio = d["salary_kgs"] / d.groupby("sub_category")["salary_kgs"].transform("median")
    noise = ((ratio > 5) | (ratio < 0.2)) & d["source"].isin(["lalafo", "telegram_kgworkzone"])
    d = d[~noise].copy()
    d["y"] = np.log(d["salary_kgs"])
    d["vac_key"] = _norm(d["name"]) + "|" + _norm(d["company"].replace("не указано", "")) + "|" + _norm(d["requirements"]).str.slice(0, 100)
    d = d.sort_values("published")
    gap = d.groupby([d["vac_key"], d["salary_kgs"].round(-2)])["published"].diff()
    repost = gap <= pd.Timedelta(days=30)
    d = d[~repost].reset_index(drop=True)
    if verbose:
        print(f"Вакансий: {len(df)} | с зарплатой после очистки: {len(d)} (шум {noise.sum()}, повторы {repost.sum()})")
    return df.reset_index(drop=True), d


def time_split(d, start, end=None):
    """Train: everything before `start`. Eval: [start, end). Eval rows whose employer+title
    (or text) already appeared in train are dropped, so the model cannot recall a repost."""
    tr = d[d["published"] < start]
    ev = d[d["published"] >= start]
    if end is not None:
        ev = ev[ev["published"] < end]
    ev = ev[~ev["vac_key"].isin(set(tr["vac_key"]))]
    return tr.reset_index(drop=True), ev.reset_index(drop=True)


def recency_weight(d, half_life):
    if not half_life:
        return np.ones(len(d))
    age = d["month_n"].max() - d["month_n"]
    return 0.5 ** (age.values / half_life)


class BlendModel:
    def __init__(self, p, alpha=5.0, half_life=12, w_gbm=0.4, te_k=5, seed=42):
        self.p, self.alpha, self.half_life, self.w_gbm, self.te_k, self.seed = p, alpha, half_life, w_gbm, te_k, seed

    # ---------------- parts
    def _ridge(self):
        p = self.p
        ct = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=p["min_freq"]), RIDGE_CAT),
            ("num", StandardScaler(), RIDGE_NUM),
            ("t", TfidfVectorizer(min_df=p["min_df"], sublinear_tf=True, ngram_range=(1, p["word_ngram"]),
                                  max_features=p["word_max"]), "full2"),
            ("ti", TfidfVectorizer(min_df=p["min_df"], sublinear_tf=True, analyzer="char_wb",
                                   ngram_range=(2, p["char_max"])), "title"),
        ], transformer_weights={"ti": p["title_weight"]})
        return make_pipeline(ct, Ridge(alpha=self.alpha))

    def _te_table(self, d, col):
        g = d.groupby(col)["y"].agg(["sum", "count"])
        te = (g["sum"] + self.te_k * self.gm_) / (g["count"] + self.te_k)
        # обычный dict: модель загружается в любой версии pandas
        return {str(k): float(v) for k, v in te.items()}

    def _gbm_X(self, x, te):
        X = pd.DataFrame(index=x.index)
        for col in TE_COLS:
            X["te_" + col] = te[col] if isinstance(te[col], np.ndarray) else x[col].astype(str).map(te[col]).fillna(self.gm_).values
        for c in GBM_CAT:
            X[c] = pd.Categorical(x[c].astype(str), categories=self.cats_[c])
        for c in GBM_NUM:
            X[c] = x[c].astype(float).values
        return X

    # ---------------- API
    def fit(self, d):
        w = recency_weight(d, self.half_life)
        self.ridge_ = self._ridge().fit(d, d["y"], ridge__sample_weight=w)
        if self.w_gbm > 0:
            self.gm_ = float(d["y"].mean())
            self.cats_ = {c: sorted(d[c].astype(str).unique().tolist()) for c in GBM_CAT}
            # out-of-fold target encoding for training rows (otherwise LightGBM sees its own answer)
            oof = {c: np.zeros(len(d)) for c in TE_COLS}
            for tr, va in KFold(5, shuffle=True, random_state=self.seed).split(d):
                part = d.iloc[tr]
                for c in TE_COLS:
                    oof[c][va] = d.iloc[va][c].astype(str).map(self._te_table(part, c)).fillna(self.gm_).values
            self.te_ = {c: self._te_table(d, c) for c in TE_COLS}
            self.gbm_ = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=30,
                                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                          random_state=self.seed, verbose=-1)
            self.gbm_.fit(self._gbm_X(d, oof), d["y"], sample_weight=w)
        return self

    def predict_parts(self, x):
        r = self.ridge_.predict(x)
        g = self.gbm_.predict(self._gbm_X(x, self.te_)) if self.w_gbm > 0 else r
        return r, g

    def predict(self, x):
        r, g = self.predict_parts(x)
        return self.w_gbm * g + (1 - self.w_gbm) * r
