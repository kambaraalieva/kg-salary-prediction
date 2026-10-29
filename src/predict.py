"""
Live salary estimate for one vacancy with the 2026 model.

    python src/predict.py "Менеджер по продажам" --exp 1
    python src/predict.py "Повар" --exp 0 --source telegram_kgworkzone
    python src/predict.py "Python developer" --exp 3 --company "o!" --text "Django, PostgreSQL, удалённо"

If --sub_category is not given, the profession is taken from the most similar job title
in the training data (models/title_lookup.csv).
"""
import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, "src"))
import prep  # noqa: E402

MODELS = os.path.join(ROOT, "models")


class Predictor:
    def __init__(self):
        self.model = joblib.load(os.path.join(MODELS, "model_2026.joblib"))
        self.meta = json.load(open(os.path.join(MODELS, "model_2026_meta.json"), encoding="utf-8"))
        self.lookup = pd.read_csv(os.path.join(MODELS, "title_lookup.csv"))
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.X = self.vec.fit_transform(self.lookup["title"])

    def profession(self, title):
        """Category and profession of the most similar known title."""
        sim = (self.vec.transform([title.lower()]) @ self.X.T).toarray()[0]
        i = int(sim.argmax())
        r = self.lookup.iloc[i]
        return r["category"], r["sub_category"], r["title"], float(sim[i])

    def predict(self, title, experience=1, source="telegram_codifynews", city="Бишкек",
                company="не указано", text="", category=None, sub_category=None, remote=0):
        match = None
        if sub_category is None:
            category, sub_category, match, _ = self.profession(title)
        row = {c: "не указано" for c in prep.CAT_FEATURES}
        row.update(category=category, sub_category=sub_category, city=city, source=source,
                   company=company.lower().strip(), experience_start=experience, n_skills=0,
                   text_len=len(text), remote=int(remote), commission=0, kyrgyz_text=0,
                   experience_known=1, month_n=self.meta["month_n_now"], name=title)
        clean = lambda s: prep.strip_money(pd.Series([s.lower()]))[0]
        row["title"] = clean(title)
        row["full"] = row["title"] + ". " + clean(text) + ". "
        x = prep.add_level(pd.DataFrame([row]))   # уровень из заголовка, а если его нет, из опыта
        p = self.model.predict(x)[0]
        q = self.meta["interval_log_quantiles"].get(source, self.meta["interval_log_quantiles"]["telegram_codifynews"])
        r = lambda v: int(round(np.exp(v), -3))
        return {"вакансия": title, "профессия": sub_category, "похожий заголовок": match, "опыт, лет": experience, "уровень": x["level"][0],
                "прогноз": r(p), "50%": (r(p + q["50%"][0]), r(p + q["50%"][1])),
                "80%": (r(p + q["80%"][0]), r(p + q["80%"][1]))}


if __name__ == "__main__":
    a = argparse.ArgumentParser(description="Оценка зарплаты вакансии (модель 2026)")
    a.add_argument("title")
    a.add_argument("--exp", type=int, default=1, help="опыт: 0, 1, 3 или 6 лет")
    a.add_argument("--source", default="telegram_codifynews",
                   help="telegram_codifynews (офис, IT, продажи) или telegram_kgworkzone (сфера услуг)")
    a.add_argument("--city", default="Бишкек")
    a.add_argument("--company", default="не указано")
    a.add_argument("--text", default="", help="текст требований (по желанию)")
    a.add_argument("--remote", action="store_true")
    a.add_argument("--sub_category", default=None)
    a.add_argument("--category", default=None)
    x = a.parse_args()
    res = Predictor().predict(x.title, x.exp, x.source, x.city, x.company, x.text, x.category, x.sub_category, x.remote)
    f = lambda v: f"{v:,}".replace(",", " ")
    print(f"\n{res['вакансия']}  (профессия: {res['профессия']}; опыт {res['опыт, лет']} г.; уровень {res['уровень']})")
    print(f"  прогноз:      {f(res['прогноз'])} сом")
    print(f"  50% диапазон: {f(res['50%'][0])} – {f(res['50%'][1])} сом")
    print(f"  80% диапазон: {f(res['80%'][0])} – {f(res['80%'][1])} сом\n")
