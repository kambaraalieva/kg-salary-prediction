"""
Extracts the template fields "Тип" and "Дополнительно" from codifynews posts.

    python src/extract_codify_fields.py ../kg-job-market-analysis/data/raw/telegram_posts.jsonl

Writes data/codify_fields.csv (link, job_type, extra). The vacancy table of the analysis project
keeps only a remote flag from "Тип"; the full field also says office / remote / internship / project
and the job level, which helps the salary model (notebook 08).
"""
import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "..", "kg-job-market-analysis", "data", "raw", "telegram_posts.jsonl")
rows = []
for line in open(src, encoding="utf-8"):
    r = json.loads(line)
    if not r["channel"].lower().endswith("codifynews"):
        continue
    t = r.get("text") or ""
    m = re.search(r"Тип:\s*(.+)", t)
    a = re.search(r"Дополнительно:\s*(.+)", t)
    rows.append({"link": r["link"], "job_type": m.group(1).strip().lower() if m else "",
                 "extra": a.group(1).strip().lower() if a else ""})
out = pd.DataFrame(rows).drop_duplicates("link")
out.to_csv(os.path.join(ROOT, "data", "codify_fields.csv"), index=False)
print(f"codify posts: {len(out)} | with Тип: {(out['job_type'] != '').mean():.0%}")
