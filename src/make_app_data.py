"""
Prepares the tables the app shows, so the app opens instantly.

    python src/make_app_data.py

Trains the final model configuration on vacancies before July 2026 (as in notebook 08) and saves
its predictions for July–October 2026 to reports/app/. Takes about a minute.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))
import blend  # noqa: E402
import prep  # noqa: E402

OUT = os.path.join(ROOT, "reports", "app")
os.makedirs(OUT, exist_ok=True)
meta = json.load(open(os.path.join(ROOT, "models", "model_2026_meta.json"), encoding="utf-8"))
TEST_FROM = pd.Timestamp("2026-07-01")

df, d = blend.load_2026()
tr, te = blend.time_split(d, TEST_FROM)
m = blend.BlendModel(meta["params"], alpha=meta["alpha"], half_life=meta["half_life"], w_gbm=meta["w_gbm"]).fit(tr)
p = m.predict(te)
Q = meta["interval_log_quantiles"]
lo50 = np.array([Q[s]["50%"][0] for s in te["source"]]); hi50 = np.array([Q[s]["50%"][1] for s in te["source"]])
lo80 = np.array([Q[s]["80%"][0] for s in te["source"]]); hi80 = np.array([Q[s]["80%"][1] for s in te["source"]])

out = pd.DataFrame({
    "дата": te["published"].dt.date.astype(str),
    "вакансия": te["name"],
    "работодатель": te["company"].replace("не указано", ""),
    "канал": te["source"].str.replace("telegram_", "", regex=False),
    "профессия": te["sub_category"],
    "сфера": te["category"],
    "опыт, лет": te["experience_start"],
    "реальная зарплата": te["salary_kgs"].round(-2),
    "прогноз": np.exp(p).round(-3),
    "80% от": np.exp(p + lo80).round(-3),
    "80% до": np.exp(p + hi80).round(-3),
    "в 50% диапазоне": (te["y"].values >= p + lo50) & (te["y"].values <= p + hi50),
    "в 80% диапазоне": (te["y"].values >= p + lo80) & (te["y"].values <= p + hi80),
})
out["ошибка, сом"] = (out["прогноз"] - out["реальная зарплата"]).abs()
out["ошибка, %"] = (out["прогноз"] / out["реальная зарплата"] - 1) * 100
out.sort_values("дата", ascending=False).to_csv(os.path.join(OUT, "test_predictions.csv"), index=False)

mm = prep.metrics(te["y"].values, p)
summary = {"test_from": "2026-07-01", "test_to": str(te["published"].max().date()), "n_test": int(len(te)),
           "n_train": int(len(tr)), "n_total": int(len(d)),
           "mae": float(mm["MAE, сом"]), "median_error": float(mm["медианная ошибка, сом"]), "mape": float(mm["MAPE, %"]),
           "within20": float(mm["доля ошибок < 20%"]), "r2_log": float(mm["R² (log)"]),
           "coverage50": float(out["в 50% диапазоне"].mean()), "coverage80": float(out["в 80% диапазоне"].mean()),
           "years": d["published"].dt.year.value_counts().sort_index().astype(int).to_dict()}
summary["years"] = {str(k): v for k, v in summary["years"].items()}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
