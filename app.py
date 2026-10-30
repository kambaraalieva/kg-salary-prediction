"""
Приложение: оценка зарплаты вакансии и итоги проекта на одной странице.

    pip install streamlit
    streamlit run app.py
"""
import json
import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "src"))
from predict import Predictor  # noqa: E402

FIG = os.path.join(ROOT, "reports", "figures")
APP = os.path.join(ROOT, "reports", "app")

st.set_page_config(page_title="Зарплаты в вакансиях Кыргызстана", page_icon="💼", layout="wide")


def som(x):
    return f"{x:,.0f}".replace(",", " ") + " сом"


@st.cache_resource
def predictor():
    return Predictor()


@st.cache_data
def summary():
    return json.load(open(os.path.join(APP, "summary.json"), encoding="utf-8"))


@st.cache_data
def test_table():
    return pd.read_csv(os.path.join(APP, "test_predictions.csv"))


@st.cache_data
def market_2026():
    """Реальные зарплаты 2026 года по профессиям: для сравнения с прогнозом."""
    d = pd.read_csv(os.path.join(ROOT, "data", "vacancies_2024_2026.csv"), low_memory=False,
                    usecols=["sub_category", "published", "has_salary", "salary_kgs", "source"])
    d = d[d["has_salary"] & (d["published"] >= "2026-01-01")]
    return d


def image(name, caption=None):
    path = os.path.join(FIG, name)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)


S = summary()
st.title("💼 Зарплаты в вакансиях Кыргызстана")
st.caption(f"Модель обучена на {S['n_total']:,} вакансиях с зарплатой".replace(",", " ")
           + " (hh, lalafo, Telegram), январь 2024 – октябрь 2026.")

tab1, tab2, tab3, tab4 = st.tabs(["🔮 Оценить вакансию", "📏 Точность модели", "🧪 Проверка на свежих вакансиях", "📈 Рынок 2024–2026"])

# ------------------------------------------------------------------ 1. прогноз
with tab1:
    left, right = st.columns([1, 1.3])
    with left:
        title = st.text_input("Название вакансии", "Менеджер по продажам")
        exp_label = st.radio("Требуемый опыт", ["без опыта", "от 1 года", "от 3 лет", "от 6 лет"], index=1, horizontal=True)
        exp = {"без опыта": 0, "от 1 года": 1, "от 3 лет": 3, "от 6 лет": 6}[exp_label]
        sector = st.radio("Тип вакансии", ["Офис, IT, продажи, маркетинг", "Сфера услуг: общепит, торговля, склад"], horizontal=False)
        source = "telegram_codifynews" if sector.startswith("Офис") else "telegram_kgworkzone"
        fmt = st.selectbox("Формат работы", ["полная занятость", "office", "remote", "можно удаленно",
                                             "частичная занятость", "project", "internship"],
                           disabled=source != "telegram_codifynews")
        with st.expander("Дополнительно (необязательно)"):
            company = st.text_input("Работодатель", "")
            text = st.text_area("Требования из объявления", "")

    P = predictor()
    res = P.predict(title, exp, source=source, company=company or "не указано", text=text,
                    job_type=fmt if source == "telegram_codifynews" else "")
    with right:
        st.subheader(title)
        st.caption(f"Профессия: {res['профессия']} · уровень: {res['уровень']}"
                   + (f" · похожий заголовок в данных: «{res['похожий заголовок']}»" if res["похожий заголовок"] else ""))
        st.metric("Ожидаемая зарплата", som(res["прогноз"]))
        c1, c2 = st.columns(2)
        c1.metric("С вероятностью 50%", f"{res['50%'][0]/1000:.0f}–{res['50%'][1]/1000:.0f} тыс.")
        c2.metric("С вероятностью 80%", f"{res['80%'][0]/1000:.0f}–{res['80%'][1]/1000:.0f} тыс.")

        curve = pd.DataFrame([{"опыт": lbl, "прогноз, тыс. сом": P.predict(title, e, source=source, company=company or "не указано",
                                                                             text=text, job_type=fmt if source == "telegram_codifynews" else "")["прогноз"] / 1000}
                              for lbl, e in [("0", 0), ("1 год", 1), ("3 года", 3), ("6 лет", 6)]])
        st.caption("Как прогноз меняется с опытом")
        st.bar_chart(curve, x="опыт", y="прогноз, тыс. сом", height=220)

        m = market_2026()
        real = m[m["sub_category"] == res["профессия"]]["salary_kgs"]
        if len(real) >= 10:
            st.info(f"Для сравнения: реальные вакансии «{res['профессия']}» в 2026 году: медиана {som(real.median())}, "
                    f"половина от {real.quantile(.25)/1000:.0f} до {real.quantile(.75)/1000:.0f} тыс. ({len(real)} вакансий).")
    st.caption("Это ориентир по рынку, а не точная зарплата: одинаковые вакансии у разных работодателей различаются в среднем на 25%.")

# ------------------------------------------------------------------ 2. точность
with tab2:
    st.subheader("Проверка на будущем")
    st.write(f"Модель обучена на вакансиях до июля 2026 и проверена на **{S['n_test']:,}** вакансиях июля–октября 2026, "
             "которых она не видела. Повторные объявления того же работодателя из проверки удалены.".replace(",", " "))
    c = st.columns(5)
    c[0].metric("Медианная ошибка", som(S["median_error"]))
    c[1].metric("Средняя ошибка (MAE)", som(S["mae"]))
    c[2].metric("Ошибка меньше 20%", f"{S['within20']:.0%}")
    c[3].metric("Попало в 50% диапазон", f"{S['coverage50']:.0%}")
    c[4].metric("Попало в 80% диапазон", f"{S['coverage80']:.0%}")

    st.subheader("Сравнение моделей (MAE на вакансиях июля–октября 2026)")
    comp = pd.DataFrame([
        ["Медиана рынка за последние 6 месяцев", 34365, 20000, "простейший ориентир"],
        ["Медиана профессии за последние 6 месяцев", 30886, 20000, "базовая линия"],
        ["Модель 2024 года без переобучения", 39072, 24666, "занижает на 36%: зарплаты выросли"],
        ["Ridge + TF-IDF, обучение 2024–2026", 27780, 17520, "ноутбук 07"],
        ["+ месяц публикации, работодатель, уровень", 26693, 16541, "ноутбук 07"],
        ["+ формат вакансии, вес свежих вакансий", 25745, 16155, "ноутбук 08"],
        ["+ смесь с LightGBM (финальная)", round(S["mae"]), round(S["median_error"]), "ноутбук 08"],
    ], columns=["модель", "MAE, сом", "медианная ошибка, сом", "комментарий"])
    st.dataframe(comp.style.format({"MAE, сом": "{:,.0f}", "медианная ошибка, сом": "{:,.0f}"}), hide_index=True,
                 use_container_width=True)
    st.caption("Числа из ноутбуков 07 и 08; при повторном обучении LightGBM они могут сдвинуться на несколько сомов.")
    a, b = st.columns(2)
    with a:
        image("07_time_models.png", "Ноутбук 07: модель 2024 года против переобученной")
    with b:
        image("08_steps.png", "Ноутбук 08: вклад каждого улучшения")

    st.subheader("Где модель точнее")
    t = test_table()
    t["уровень зарплаты"] = pd.cut(t["реальная зарплата"], [0, 25e3, 40e3, 60e3, 100e3, 1e9],
                                   labels=["до 25 тыс.", "25–40 тыс.", "40–60 тыс.", "60–100 тыс.", "больше 100 тыс."])
    seg = t.groupby("уровень зарплаты", observed=True).agg(вакансий=("прогноз", "size"),
                                                           медианная_ошибка=("ошибка, сом", "median"),
                                                           в_80_диапазоне=("в 80% диапазоне", "mean"))
    st.dataframe(seg.style.format({"медианная_ошибка": "{:,.0f}", "в_80_диапазоне": "{:.0%}"}), use_container_width=True)
    st.write("Точнее всего для типичных зарплат 40–100 тыс. Редкие дорогие вакансии модель занижает, "
             "очень дешёвые завышает: она тянет крайние значения к середине.")

# ------------------------------------------------------------------ 3. свежие вакансии
with tab3:
    t = test_table()
    t["работодатель"] = t["работодатель"].fillna("")
    st.write("Реальные вакансии июля–октября 2026: зарплата из объявления против прогноза модели, "
             "обученной только на данных до июля.")
    f1, f2, f3 = st.columns(3)
    ch = f1.selectbox("Канал", ["все", "codifynews", "kgworkzone"])
    q = f2.text_input("Поиск по названию", "")
    only_miss = f3.checkbox("Только промахи (вне 80% диапазона)")
    v = t.copy()
    if ch != "все":
        v = v[v["канал"] == ch]
    if q:
        v = v[v["вакансия"].str.contains(q, case=False, na=False)]
    if only_miss:
        v = v[~v["в 80% диапазоне"]]
    c = st.columns(3)
    c[0].metric("Вакансий", len(v))
    if len(v):
        c[1].metric("Медианная ошибка", som(v["ошибка, сом"].median()))
        c[2].metric("В 80% диапазоне", f"{v['в 80% диапазоне'].mean():.0%}")
    show = v[["дата", "вакансия", "работодатель", "канал", "опыт, лет", "реальная зарплата", "прогноз", "80% от", "80% до",
              "ошибка, %", "в 80% диапазоне"]]
    st.dataframe(show.style.format({"реальная зарплата": "{:,.0f}", "прогноз": "{:,.0f}", "80% от": "{:,.0f}",
                                    "80% до": "{:,.0f}", "ошибка, %": "{:+.0f}"}),
                 hide_index=True, use_container_width=True, height=520)

# ------------------------------------------------------------------ 4. рынок
with tab4:
    c = st.columns(4)
    c[0].metric("Вакансий в данных", "44 656")
    c[1].metric("Скрывают зарплату (2024)", "52%")
    c[2].metric("Рост зарплат в вакансиях", "+31%", help="та же вакансия, I кв. 2024 → конец 2026, с поправкой на состав")
    c[3].metric("Официальная средняя зарплата", "+61%", help="Нацстатком: 34 046 → 54 864 сом")
    a, b = st.columns(2)
    with a:
        image("market_08_index.png", "Та же вакансия подорожала на ~31%; официальная средняя выросла сильнее")
        image("market_08_shares.png", "Доля скрытых зарплат и состав вакансий по кварталам (codifynews)")
    with b:
        image("market_08_professions.png", "Изменение зарплат по профессиям, 2024 → 2026")
        image("market_03_q3_experience.png", "Зарплата и опыт (2024)")
    st.caption("Подробности: репозиторий kg-job-market-analysis, ноутбуки 03 и 08.")
