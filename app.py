"""app.py – Dashboard Streamlit: analiza arm slotu miotaczy (MLB / Statcast).
Uruchomienie:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data_utils as du
import insights as ins

st.set_page_config(page_title="Arm Slot Lab", page_icon="⚾", layout="wide")
DATA_DIR = Path(__file__).parent / "data"
L = du.LABELS


def show(fig):
    st.plotly_chart(fig, width="stretch")


def table(df: pd.DataFrame):
    st.dataframe(df.rename(columns=L), width="stretch", hide_index=True)


# ----------------------------------------------------------------------------- dane
@st.cache_data(show_spinner="Wczytuję i łączę dane…")
def load(sources: tuple):
    return du.build_dataset(list(sources))


st.title("⚾ Arm Slot Lab")
st.caption("Jak kąt ramienia wpływa na IVB, HB, prędkość i spin – liga, miotacze, zmiany rok do roku.")

with st.sidebar:
    st.header("📂 Dane")
    uploaded = st.file_uploader("Wgraj pliki CSV (arm angle / movement / enriched)",
                                type="csv", accept_multiple_files=True)
    use_folder = st.checkbox("Wczytaj też pliki z folderu `data/`", value=True)

sources = []
if use_folder and DATA_DIR.exists():
    sources += [(p.name, p.read_bytes()) for p in sorted(DATA_DIR.glob("*.csv"))]
sources += [(f.name, f.getvalue()) for f in (uploaded or [])]

if not sources:
    st.info("Wgraj pliki CSV w panelu bocznym albo umieść je w folderze `data/`.\n\n"
            "Obsługiwane: **surowe arm angle** (`pitcher_arm_angles_2024.csv`…), **movement/arsenal** z Baseball Savant "
            "oraz **enriched** (arm_angle + ivb + hb + velo + spin + pitch_type). Szczegóły w README.")
    st.stop()

raw, msgs = load(tuple(sources))
with st.expander("🔎 Diagnostyka wczytanych plików", expanded=bool(raw.empty)):
    for lvl, m in msgs:
        getattr(st, lvl)(m)
if raw.empty:
    st.error("Nie udało się zbudować zbioru danych – sprawdź komunikaty powyżej.")
    st.stop()

# ---- Kontrola kolumn
present = {c: (c in raw.columns and raw[c].notna().any()) for c in ["arm_angle", "ivb", "hb", "velo", "spin"]}
present["pitch_type"] = raw["pitch_type"].nunique() > 1 or (raw["pitch_type"] != "ALL").any()
cols = st.columns(len(present))
for col, (name, ok) in zip(cols, present.items()):
    col.markdown(f"{'✅' if ok else '❌'} `{name}`")
if not present["arm_angle"]:
    st.error("Brak danych o arm slocie (kolumna `arm_angle` / `ball_angle`). Wgraj pliki arm angle.")
    st.stop()
missing = [m for m in du.CORE_METRICS if not present[m]]
if missing:
    st.warning(f"Brakuje kolumn: **{', '.join(missing)}**. Wykresy dla tych metryk będą niedostępne. "
               "Dociągnij dane z Savant (movement / arsenal) – patrz README.")
if not present["pitch_type"]:
    st.warning("Brak informacji o typie pitcha – dostępna tylko analiza „Overall”.")
METRICS = [m for m in du.CORE_METRICS + du.OUTCOME_METRICS if m in raw.columns and raw[m].notna().any()]

# ----------------------------------------------------------------------------- filtry
all_seasons = sorted(raw.season.unique())
types_present = [t for t in du.PITCH_ORDER if t in set(raw.pitch_type)] + \
                sorted(t for t in set(raw.pitch_type) if t not in du.PITCH_ORDER and t != "ALL")

with st.sidebar:
    st.header("🎛️ Filtry")
    seasons = st.multiselect("Sezony", all_seasons, default=all_seasons)
    level_lbl = st.radio("Poziom analizy", ["Wg typu pitcha", "Overall (wszystkie pitche)"],
                         index=0 if present["pitch_type"] else 1,
                         help="Overall = średnia ważona po wszystkich pitchach miotacza (mieszanka repertuaru!).")
    level = "type" if level_lbl.startswith("Wg") else "overall"
    sel_types = st.multiselect("Typy pitchów", types_present, default=types_present, disabled=(level == "overall"))
    hands = st.multiselect("Ręka rzutowa", ["R", "L"], default=["R", "L"]) if raw.pitch_hand.notna().any() else []
    hb_abs = st.checkbox("HB jako |wartość bezwzględna|", value=False,
                         help="Znak HB zależy od ręki miotacza – |HB| ułatwia porównania R vs L.")
    width = st.select_slider("Szerokość bucketu arm slotu (°)", [1, 2, 3, 4, 5], value=2)
    min_p = st.slider("Min. pitchy (wiersz miotacz×sezon×pitch)", 0, 500, 50, 10)

df = raw[raw.arm_angle.notna()].copy()
if hb_abs and "hb" in df:
    df["hb"] = df["hb"].abs()
if hands:
    df = df[df.pitch_hand.isin(hands) | df.pitch_hand.isna()]
base_all = du.prepare_level(df, level, METRICS)
base_all = base_all[base_all.pitches.fillna(np.inf) >= min_p]
filtered = base_all[base_all.season.isin(seasons)]
if level == "type":
    filtered = filtered[filtered.pitch_type.isin(sel_types)]
if filtered.empty:
    st.warning("Brak danych po zastosowaniu filtrów – poluzuj filtry w panelu bocznym.")
    st.stop()

k = st.columns(4)
k[0].metric("Wierszy", f"{len(filtered):,}")
k[1].metric("Miotaczy", f"{filtered.name_key.nunique():,}")
k[2].metric("Śr. arm slot", f"{du.wmean(filtered.arm_angle, filtered.pitches):.1f}°")
k[3].metric("Zakres arm slotu", f"{filtered.arm_angle.min():.0f}–{filtered.arm_angle.max():.0f}°")


def pitcher_picker(base: pd.DataFrame, key: str):
    o = base.sort_values("pitches", ascending=False, na_position="last").drop_duplicates("name_key")
    names = dict(zip(o.pitcher_name, o.name_key))
    sel = st.selectbox("Miotacz", list(names), key=key)
    return sel, names[sel]


def type_picker(d: pd.DataFrame, key: str):
    opts = [t for t in du.PITCH_ORDER if t in set(d.pitch_type)] + \
           sorted(t for t in set(d.pitch_type) if t not in du.PITCH_ORDER)
    return st.selectbox("Typ pitcha", opts, key=key)


tabs = st.tabs(["📊 Liga wg arm slotu", "📈 Korelacje", "🧑‍🎓 Miotacz", "🔁 Zmiany YoY", "🤖 AI Insights", "🗂️ Dane"])

# ----------------------------------------------------------------------------- 1. Liga
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    per_season = c1.checkbox("Rozdziel sezony", True)
    split_type = c2.checkbox("Rozdziel typy pitchów", value=False, disabled=(level != "type"))
    min_pitchers = c3.slider("Min. miotaczy w buckecie", 1, 20, 3)
    extra = (["season"] if per_season else []) + (["pitch_type"] if split_type and level == "type" else [])
    tbl = du.league_by_bucket(filtered, width, METRICS, extra)
    tbl = tbl[tbl.pitchers >= min_pitchers]
    if tbl.empty:
        st.info("Za mało miotaczy w bucketach – zmniejsz próg lub zwiększ szerokość bucketu.")
    else:
        tbl["grupa"] = tbl[extra].astype(str).agg(" · ".join, axis=1) if extra else "Liga"
        order = tbl.sort_values("bucket_low").bucket.unique().tolist()
        if METRICS:
            m = st.selectbox("Metryka", METRICS, format_func=lambda x: L[x], key="t1m")
            show(px.line(tbl, x="bucket", y=m, color="grupa", markers=True, category_orders={"bucket": order},
                         labels={"bucket": "Arm slot", m: L[m], "grupa": ""}))
        ycol = st.radio("Wolumen", ["pitches", "pitchers"], format_func=lambda x: L[x], horizontal=True)
        show(px.bar(tbl, x="bucket", y=ycol, color="grupa", barmode="group", category_orders={"bucket": order},
                    labels={"bucket": "Arm slot", ycol: L[ycol], "grupa": ""}))
        table(tbl[["bucket"] + extra + METRICS + ["pitches", "pitchers"]].round(2))

# ----------------------------------------------------------------------------- 2. Korelacje
with tabs[1]:
    if not METRICS:
        st.warning("Brak kolumn movement – wgraj dane enriched albo pliki movement.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        sel_m = c1.selectbox("Metryka", METRICS, format_func=lambda x: L[x], key="t2m")
        color_opts = ["pitch_type", "season"] + (["pitch_hand"] if filtered.pitch_hand.notna().any() else [])
        color_by = c2.selectbox("Kolorowanie", color_opts, key="t2c")
        trend_each = c3.checkbox("Trend per grupa", False)
        grid = c4.checkbox("Wszystkie metryki (siatka)", False)
        mets = METRICS if grid else [sel_m]
        cols2 = st.columns(2) if len(mets) > 1 else [st.container()]
        for i, m in enumerate(mets):
            d = filtered.dropna(subset=[m]).copy()
            d[color_by] = d[color_by].astype(str)
            with cols2[i % len(cols2)]:
                if d.empty:
                    st.info(f"Brak danych: {L[m]}")
                    continue
                fig = px.scatter(d, x="arm_angle", y=m, color=color_by, opacity=0.65, hover_name="pitcher_name",
                                 hover_data={"season": True, "pitch_type": True, "pitches": True},
                                 labels={"arm_angle": L["arm_angle"], m: L[m]})
                fit = du.linear_fit(d.arm_angle, d[m])
                if fit:
                    xs = np.array([d.arm_angle.min(), d.arm_angle.max()])
                    fig.add_trace(go.Scatter(x=xs, y=fit["intercept"] + fit["slope"] * xs, mode="lines",
                                             name="Trend (wszystkie)", line=dict(color="black", dash="dash")))
                    st.caption(f"**{L[m]}**: nachylenie {ins.num(fit['slope'], 2, True)} / 1°, "
                               f"r = {ins.num(fit['r'], 2)} ({ins.strength(fit['r'])}), n = {fit['n']}")
                if trend_each:
                    for tr in list(fig.data):
                        g = d[d[color_by] == tr.name]
                        f2 = du.linear_fit(g.arm_angle, g[m])
                        if f2:
                            xs = np.array([g.arm_angle.min(), g.arm_angle.max()])
                            fig.add_trace(go.Scatter(x=xs, y=f2["intercept"] + f2["slope"] * xs, mode="lines",
                                                     line=dict(color=tr.marker.color, width=2), showlegend=False,
                                                     hoverinfo="skip"))
                show(fig)

# ----------------------------------------------------------------------------- 3. Miotacz
with tabs[2]:
    c1, c2, c3 = st.columns(3)
    with c1:
        pname, pkey = pitcher_picker(base_all, "t3p")
    pdata = base_all[base_all.name_key == pkey]
    with c2:
        pt = type_picker(pdata, "t3t")
    window = c3.slider("Okno porównania ± (°)", 1.0, 6.0, 2.0, 0.5)
    pt_df = pdata[pdata.pitch_type == pt].sort_values("season")
    league_pt = base_all[base_all.pitch_type == pt]
    st.caption("Widok miotacza obejmuje wszystkie wczytane sezony (filtr sezonów go nie dotyczy).")

    fig = px.line(pt_df, x="season", y="arm_angle", markers=True, text=pt_df.arm_angle.round(1),
                  labels={"season": "Sezon", "arm_angle": L["arm_angle"]}, title=f"{pname} – arm slot wg sezonu")
    fig.update_traces(textposition="top center")
    fig.update_xaxes(dtick=1)
    show(fig)

    mets3 = [m for m in METRICS if pt_df[m].notna().any()]
    if mets3:
        cc = st.columns(2)
        for i, m in enumerate(mets3):
            f = go.Figure()
            lg = league_pt.dropna(subset=[m])
            f.add_trace(go.Scatter(x=lg.arm_angle, y=lg[m], mode="markers", name="Liga", hoverinfo="skip",
                                   marker=dict(color="lightgray", size=5, opacity=0.5)))
            f.add_trace(go.Scatter(x=pt_df.arm_angle, y=pt_df[m], mode="lines+markers+text", name=pname,
                                   text=pt_df.season.astype(str), textposition="top center",
                                   marker=dict(size=11, color="#d62728")))
            f.update_layout(title=f"{L[m]} vs arm slot ({pt})", xaxis_title=L["arm_angle"], yaxis_title=L[m],
                            margin=dict(t=50, b=20))
            with cc[i % 2]:
                show(f)
        rows = []
        for _, r in pt_df.iterrows():
            ref = du.league_reference(league_pt, r.season, pt, r.arm_angle, window, mets3, exclude_key=pkey)
            for m in mets3:
                refv, n = ref.get(m, (np.nan, 0))
                rows.append({"Sezon": r.season, "Arm slot": round(r.arm_angle, 1), "Metryka": L[m],
                             "Miotacz": round(r[m], 1) if pd.notna(r[m]) else None,
                             f"Liga (±{window:g}°)": round(refv, 1) if pd.notna(refv) else None,
                             "Δ vs liga": round(r[m] - refv, 1) if pd.notna(r[m]) and pd.notna(refv) else None,
                             "Miotaczy w próbie": n})
        st.subheader("Porównanie z ligą dla podobnego arm slotu")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.warning("Brak metryk movement dla tego miotacza / typu pitcha.")

# ----------------------------------------------------------------------------- 4. YoY
with tabs[3]:
    if len(all_seasons) < 2:
        st.info("Potrzebne są co najmniej dwa sezony.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            pt4 = type_picker(base_all, "t4t")
        s_from = c2.selectbox("Sezon od", all_seasons, index=len(all_seasons) - 2)
        s_to = c3.selectbox("Sezon do", all_seasons, index=len(all_seasons) - 1)
        c4, c5, c6 = st.columns(3)
        direction = c4.radio("Kierunek", ["Wszystkie", "Wzrost arm slotu", "Spadek arm slotu"], horizontal=True)
        min_d = c5.slider("Min. |Δ arm slot| (°)", 0.0, 10.0, 2.0, 0.5)
        top_n = c6.slider("Top N", 10, 100, 25, 5)
        yo = du.yoy_changes(base_all[base_all.pitch_type == pt4], s_from, s_to, METRICS, min_p)
        if s_from >= s_to:
            st.warning("„Sezon od” musi być wcześniejszy niż „Sezon do”.")
        elif yo.empty:
            st.info("Brak miotaczy spełniających kryteria w obu sezonach.")
        else:
            yo = yo[yo.d_arm.abs() >= min_d]
            if direction == "Wzrost arm slotu":
                yo = yo[yo.d_arm > 0]
            elif direction == "Spadek arm slotu":
                yo = yo[yo.d_arm < 0]
            yo = yo.reindex(yo.d_arm.abs().sort_values(ascending=False).index).head(top_n)
            has_ivb = "d_ivb" in yo.columns
            if has_ivb:
                thr = st.slider("Próg „poprawy IVB” (cale)", 0.0, 5.0, 1.0, 0.5)
                yo["Poprawa IVB"] = np.where(yo.d_ivb >= thr, "✅", "—")
                st.caption(f"Poprawa IVB: {(yo.d_ivb >= thr).mean():.0%} z {len(yo)} miotaczy.")
            else:
                st.warning("Brak IVB – pokazuję tylko zmiany arm slotu.")
            if len(yo):
                f = px.bar(yo.sort_values("d_arm"), x="d_arm", y="pitcher_name", orientation="h",
                           color="d_ivb" if has_ivb else None, color_continuous_scale="RdYlGn",
                           color_continuous_midpoint=0 if has_ivb else None,
                           labels={"d_arm": "Δ arm slot (°)", "pitcher_name": "", "d_ivb": "Δ IVB"},
                           title=f"Największe zmiany arm slotu {s_from}→{s_to} ({pt4})")
                f.update_layout(height=max(350, 22 * len(yo)))
                show(f)
                dm = [m for m in METRICS if f"d_{m}" in yo.columns]
                if dm:
                    sm = st.selectbox("Zależność Δ arm slot vs", dm, format_func=lambda x: L[x], key="t4m")
                    allc = du.yoy_changes(base_all[base_all.pitch_type == pt4], s_from, s_to, METRICS, min_p)
                    fs = px.scatter(allc, x="d_arm", y=f"d_{sm}", hover_name="pitcher_name",
                                    labels={"d_arm": "Δ arm slot (°)", f"d_{sm}": f"Δ {L[sm]}"})
                    fit = du.linear_fit(allc.d_arm, allc[f"d_{sm}"])
                    if fit:
                        xs = np.array([allc.d_arm.min(), allc.d_arm.max()])
                        fs.add_trace(go.Scatter(x=xs, y=fit["intercept"] + fit["slope"] * xs, mode="lines",
                                                line=dict(color="black", dash="dash"), name="Trend"))
                        st.caption(f"Wszyscy miotacze (n = {fit['n']}): nachylenie {ins.num(fit['slope'], 2, True)}, "
                                   f"r = {ins.num(fit['r'], 2)}")
                    show(fs)
                show_cols = ["pitcher_name", "arm_angle_from", "arm_angle_to", "d_arm"] + \
                            [f"d_{m}" for m in dm] + (["Poprawa IVB"] if has_ivb else [])
                out = yo[show_cols].copy()
                out.columns = ["Miotacz", f"Arm {s_from}", f"Arm {s_to}", "Δ arm"] + \
                              [f"Δ {L[m]}" for m in dm] + (["Poprawa IVB"] if has_ivb else [])
                st.dataframe(out.round(2), width="stretch", hide_index=True)

# ----------------------------------------------------------------------------- 5. Insights
with tabs[4]:
    if not any(m in METRICS for m in du.CORE_METRICS):
        st.warning("Insights wymagają kolumn movement (ivb/hb/velo/spin).")
    else:
        scope = st.radio("Zakres", ["Cała liga", "Wybrany miotacz"], horizontal=True)
        c1, c2 = st.columns(2)
        with c1:
            pt5 = type_picker(base_all, "t5t")
        league5 = base_all[base_all.pitch_type == pt5]
        cm = [m for m in du.CORE_METRICS if m in METRICS]
        if scope == "Cała liga":
            d5 = league5[league5.season.isin(seasons)]
            yo5 = du.yoy_all(league5, cm, min_p, seasons=seasons)
            items = ins.league_insights(d5, pt5, cm, yo5)
        else:
            with c2:
                pname5, pkey5 = pitcher_picker(base_all, "t5p")
            w5 = st.slider("Okno porównania ± (°)", 1.0, 6.0, 2.0, 0.5, key="t5w")
            items = ins.pitcher_insights(league5[league5.name_key == pkey5], league5, pt5, cm, w5)
        for t in items:
            (st.warning if t.startswith("⚠️") else st.info)(t)
        with st.expander("Jak powstają te wnioski?"):
            st.markdown("- Regresja liniowa metryki względem arm slotu (r i n podane przy każdym wniosku).\n"
                        "- Porównanie górnej i dolnej trzeciej stawki arm slotu oraz najlepszy bucket 5°.\n"
                        "- Historyczne zmiany YoY pogrupowane wg wielkości zmiany arm slotu (min. 8 przypadków).\n"
                        "- Dla miotacza: porównanie z ligą w oknie ±N° arm slotu i oczekiwana zmiana wg ligowej regresji.")

# ----------------------------------------------------------------------------- 6. Dane
with tabs[5]:
    st.dataframe(filtered.drop(columns=["name_key"]).round(2), width="stretch", hide_index=True)
    st.download_button("⬇️ Pobierz przefiltrowane dane (CSV)", filtered.to_csv(index=False).encode("utf-8"),
                       "arm_slot_filtered.csv", "text/csv")
    st.download_button("⬇️ Pobierz pełny enriched (CSV)", du.to_enriched(raw).to_csv(index=False).encode("utf-8"),
                       "enriched_all.csv", "text/csv")
