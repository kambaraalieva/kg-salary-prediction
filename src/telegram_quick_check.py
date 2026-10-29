"""
First look at collected Telegram posts: is there enough to compare with 2024?

    python src/telegram_quick_check.py
"""

import json
import re
from pathlib import Path

import pandas as pd

rows = [json.loads(l) for l in Path("data/telegram/raw_posts.jsonl").open(encoding="utf-8")]
df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
t = df["text"].str.lower()

salary_like = t.str.contains(r"\d[\d\s]{2,}\s*(?:сом|с\b|kgs|\$|usd|тыс|к\b)|зарплат\w*\s*[:\-]?\s*(?:от\s*)?\d|з/п\s*[:\-]?\s*\d|оклад")
vacancy_like = t.str.contains(r"требу|ваканси|ищем|нужен|нужна|керек|приглашаем|набор")

print(f"Posts: {len(df)} | channels: {df['channel'].nunique()}")
print(f"Period: {df['date'].min():%Y-%m-%d} .. {df['date'].max():%Y-%m-%d}")
print(f"Look like vacancies: {vacancy_like.mean():.0%}")
print(f"Mention a salary:    {salary_like.mean():.0%}  ->  ~{int((salary_like & vacancy_like).sum())} usable posts")
print("\nBy channel:")
print(df.assign(sal=salary_like, vac=vacancy_like).groupby("channel")
        .agg(posts=("text", "size"), vacancies=("vac", "mean"), with_salary=("sal", "mean")).round(2).to_string())
print("\nThree random posts with a salary:")
for s in df[salary_like & vacancy_like].sample(min(3, int((salary_like & vacancy_like).sum())), random_state=1)["text"]:
    print("-" * 60); print(s[:500])
n = int((salary_like & vacancy_like).sum())
print("\n" + "=" * 60)
print("VERDICT:", "enough for a comparison" if n >= 500 else
      "borderline: add 2-3 more channels" if n >= 200 else "too few: add more channels or an earlier --since")
