import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Arm Slot Analytics Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚾ Arm Slot Analytics Dashboard")
st.markdown("Analiza wpływu **arm slotu** na movement (IVB / HB) i wyniki. Dane z Baseball Savant.")

# ============================================================
# POMOCNICZE FUNKCJE
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizuje nazwy kolumn z różnych źródeł Savant."""
    col_map = {
        # Arm angle
        "arm_angle": "arm_angle",
        "Arm Angle": "arm_angle",
        "arm angle": "arm_angle",
        # Player
        "player_name": "player_name",
        "Player": "player_name",
        "player": "player_name",
        "last_name, first_name": "player_name",
        "pitcher": "player_name",
        # Year / Season
        "year": "year",
        "Year": "year",
        "season": "year",
        "Season": "year",
        # Pitch type
        "pitch_type": "pitch_type",
        "Pitch Type": "pitch_type",
        "pitch_name": "pitch_name",
        "api_pitch_type": "pitch_type",
        # Movement - Induced Vertical Break
        "induced_vertical_break": "ivb",
        "Induced Vertical Break": "ivb",
        "ivb": "ivb",
        "pfx_z": "ivb",                 # często w stopach → pomnożymy
        "avg_break_z": "ivb",
        "ff_avg_break_z": "ivb",
        "vertical_break": "vb",
        "Vertical Break": "vb",
        # Horizontal Break
        "horizontal_break": "hb",
        "Horizontal Break": "hb",
        "pfx_x": "hb",
        "avg_break_x": "hb",
        "ff_avg_break_x": "hb",
        "api_break_x_arm": "hb_arm",
        # Velocity / Spin
        "release_speed": "velo",
        "avg_speed": "velo",
        "velocity": "velo",
        "release_spin_rate": "spin",
        "avg_spin": "spin",
        # Results
        "whiff_percent": "whiff_pct",
        "whiff%": "whiff_pct",
        "Whiff %": "whiff_pct",
        "woba": "woba",
        "xwoba": "xwoba",
        "run_value": "run_value",
        "pitches": "pitches",
        "n_pitches": "pitches",
        "pitch_count": "pitches",
    }

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_dict = {}
    for orig, new in col_map.items():
        for col in df.columns:
            if col.lower() == orig.lower() or col == orig:
                rename_dict[col] = new
                break

    df = df.rename(columns=rename_dict)

    # Konwersja jednostek (pfx_* często w stopach → cale)
    if "ivb" in df.columns and df["ivb"].abs().mean() < 5:
        df["ivb"] = df["ivb"] * 12
    if "hb" in df.columns and df["hb"].abs().mean() < 5:
        df["hb"] = df["hb"] * 12

    # Arm angle do float
    if "arm_angle" in df.columns:
        df["arm_angle"] = pd.to_numeric(df["arm_angle"], errors="coerce")

    return df

def create_arm_buckets(series: pd.Series, bin_size: float = 2.0) -> pd.Series:
    """Tworzy buckety arm slotu np. 44-46, 46-48 itd."""
    min_val = np.floor(series.min() / bin_size) * bin_size
    max_val = np.ceil(series.max() / bin_size) * bin_size
    bins = np.arange(min_val, max_val + bin_size, bin_size)
    labels = [f"{int(b)}-{int(b+bin_size)}" for b in bins[:-1]]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)

# ============================================================
# SIDEBAR – UPLOAD + FILTRY
# ============================================================

with st.sidebar:
    st.header("📂 Dane")
    uploaded_files = st.file_uploader(
        "Wrzuć CSV z Baseball Savant (można wiele)",
        type=["csv"],
        accept_multiple_files=True
    )

    st.markdown("---")
    st.header("⚙️ Ustawienia")

    bin_size = st.slider("Rozmiar bucketu arm slotu (°)", 1.0, 5.0, 2.0, 0.5)
    min_pitches = st.number_input("Min. liczba pitchy (filtrowanie)", 0, 1000, 50)

    st.markdown("---")
    st.markdown("""
    **Rekomendowane CSV-y:**
    1. Arm Angle Leaderboard
    2. Custom Leaderboard (arm_angle + breaki + wyniki)
    3. Pitch Movement
    4. Statcast Search (pitch-level z `arm_angle`)
    """)

# ============================================================
# ŁADOWANIE DANYCH
# ============================================================

if not uploaded_files:
    st.info("👈 Wrzuć co najmniej jeden plik CSV z Baseball Savant, żeby zacząć.")
    st.stop()

dfs = []
for f in uploaded_files:
    try:
        temp = pd.read_csv(f)
        temp = normalize_columns(temp)
        temp["_source"] = f.name
        dfs.append(temp)
    except Exception as e:
        st.error(f"Błąd przy pliku {f.name}: {e}")

if not dfs:
    st.error("Nie udało się wczytać żadnego pliku.")
    st.stop()

df = pd.concat(dfs, ignore_index=True)

# Sprawdzenie kluczowych kolumn
required = ["arm_angle"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Brakuje kluczowych kolumn: {missing}. Sprawdź czy CSV zawiera arm_angle.")
    st.write("Dostępne kolumny:", list(df.columns))
    st.stop()

# Opcjonalne kolumny
has_player = "player_name" in df.columns
has_year = "year" in df.columns
has_pitch_type = "pitch_type" in df.columns or "pitch_name" in df.columns
has_ivb = "ivb" in df.columns
has_hb = "hb" in df.columns
has_velo = "velo" in df.columns
has_whiff = "whiff_pct" in df.columns
has_pitches = "pitches" in df.columns

# Filtrowanie min pitches
if has_pitches:
    df = df[df["pitches"].fillna(0) >= min_pitches]

df = df.dropna(subset=["arm_angle"])

# Buckety
df["arm_bucket"] = create_arm_buckets(df["arm_angle"], bin_size)

st.success(f"Wczytano **{len(df):,}** wierszy z **{len(uploaded_files)}** pliku/ów.")

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview & Buckets",
    "📈 Arm Slot vs Movement",
    "👤 Indywidualni miotacze",
    "🔄 Największe zmiany",
    "💡 Insights & Tips"
])

# ----------------------------------------------------------
# TAB 1 – Overview & Buckets
# ----------------------------------------------------------
with tab1:
    st.subheader("Średnie wartości według Arm Slot Bucket")

    agg_cols = {"arm_angle": "mean"}
    if has_ivb:
        agg_cols["ivb"] = "mean"
    if has_hb:
        agg_cols["hb"] = "mean"
    if has_velo:
        agg_cols["velo"] = "mean"
    if has_whiff:
        agg_cols["whiff_pct"] = "mean"
    if has_pitches:
        agg_cols["pitches"] = "sum"

    bucket_stats = (
        df.groupby("arm_bucket", observed=True)
        .agg(agg_cols)
        .reset_index()
        .sort_values("arm_bucket")
    )

    st.dataframe(
        bucket_stats.style.format({
            "arm_angle": "{:.1f}",
            "ivb": "{:.1f}",
            "hb": "{:.1f}",
            "velo": "{:.1f}",
            "whiff_pct": "{:.1f}",
            "pitches": "{:.0f}"
        }),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        if has_ivb:
            fig = px.bar(
                bucket_stats, x="arm_bucket", y="ivb",
                title="Średni Induced Vertical Break (cale) vs Arm Slot",
                labels={"arm_bucket": "Arm Slot Bucket", "ivb": "IVB (cale)"},
                color="ivb", color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if has_hb:
            fig = px.bar(
                bucket_stats, x="arm_bucket", y="hb",
                title="Średni Horizontal Break (cale) vs Arm Slot",
                labels={"arm_bucket": "Arm Slot Bucket", "hb": "HB (cale)"},
                color="hb", color_continuous_scale="RdBu"
            )
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# TAB 2 – Arm Slot vs Movement (scatter)
# ----------------------------------------------------------
with tab2:
    st.subheader("Zależność Arm Angle → Movement")

    if has_pitch_type:
        pitch_col = "pitch_type" if "pitch_type" in df.columns else "pitch_name"
        selected_pitches = st.multiselect(
            "Filtruj typy piłek",
            options=sorted(df[pitch_col].dropna().unique()),
            default=list(df[pitch_col].dropna().unique())[:5]
        )
        plot_df = df[df[pitch_col].isin(selected_pitches)] if selected_pitches else df
    else:
        plot_df = df
        pitch_col = None

    col1, col2 = st.columns(2)

    with col1:
        if has_ivb:
            fig = px.scatter(
                plot_df, x="arm_angle", y="ivb",
                color=pitch_col if pitch_col else None,
                hover_data=["player_name"] if has_player else None,
                title="Arm Angle vs Induced Vertical Break",
                labels={"arm_angle": "Arm Angle (°)", "ivb": "IVB (cale)"},
                trendline="ols" if len(plot_df) > 10 else None
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if has_hb:
            fig = px.scatter(
                plot_df, x="arm_angle", y="hb",
                color=pitch_col if pitch_col else None,
                hover_data=["player_name"] if has_player else None,
                title="Arm Angle vs Horizontal Break",
                labels={"arm_angle": "Arm Angle (°)", "hb": "HB (cale)"},
                trendline="ols" if len(plot_df) > 10 else None
            )
            st.plotly_chart(fig, use_container_width=True)

    # Movement plot w stylu Savant (HB vs IVB z kolorowaniem po arm angle)
    if has_ivb and has_hb:
        st.subheader("Movement Plot (kolor = Arm Angle)")
        fig = px.scatter(
            plot_df, x="hb", y="ivb",
            color="arm_angle",
            color_continuous_scale="Turbo",
            hover_data=["player_name", "arm_angle"] if has_player else ["arm_angle"],
            title="Horizontal Break vs Induced Vertical Break (kolor = Arm Slot)",
            labels={"hb": "Horizontal Break (cale)", "ivb": "IVB (cale)"}
        )
        fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# TAB 3 – Indywidualni miotacze
# ----------------------------------------------------------
with tab3:
    st.subheader("Analiza indywidualna miotacza")

    if not has_player:
        st.warning("Brak kolumny z nazwiskiem miotacza w danych.")
    else:
        players = sorted(df["player_name"].dropna().unique())
        selected_player = st.selectbox("Wybierz miotacza", players)

        player_df = df[df["player_name"] == selected_player].copy()

        st.markdown(f"### {selected_player}")

        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Średni Arm Angle", f"{player_df['arm_angle'].mean():.1f}°")
        if has_ivb:
            metrics_cols[1].metric("Średni IVB", f"{player_df['ivb'].mean():.1f}\"")
        if has_hb:
            metrics_cols[2].metric("Średni HB", f"{player_df['hb'].mean():.1f}\"")
        if has_velo:
            metrics_cols[3].metric("Średnia Velo", f"{player_df['velo'].mean():.1f} mph")

        if has_year and player_df["year"].nunique() > 1:
            st.subheader("Zmiana w czasie (sezon do sezonu)")
            yearly = player_df.groupby("year").agg({
                "arm_angle": "mean",
                **({"ivb": "mean"} if has_ivb else {}),
                **({"hb": "mean"} if has_hb else {}),
                **({"velo": "mean"} if has_velo else {}),
            }).reset_index()

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=yearly["year"], y=yearly["arm_angle"], name="Arm Angle", mode="lines+markers"),
                secondary_y=False
            )
            if has_ivb:
                fig.add_trace(
                    go.Scatter(x=yearly["year"], y=yearly["ivb"], name="IVB", mode="lines+markers"),
                    secondary_y=True
                )
            fig.update_layout(title="Arm Angle & IVB w czasie")
            fig.update_yaxes(title_text="Arm Angle (°)", secondary_y=False)
            fig.update_yaxes(title_text="IVB (cale)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(player_df, use_container_width=True)

# ----------------------------------------------------------
# TAB 4 – Największe zmiany
# ----------------------------------------------------------
with tab4:
    st.subheader("Miotacze z największymi zmianami Arm Slotu")

    if not (has_player and has_year):
        st.warning("Potrzebne kolumny: player_name + year, żeby policzyć zmiany sezon do sezonu.")
    else:
        # Średni arm angle per player-year
        yearly_arm = (
            df.groupby(["player_name", "year"])["arm_angle"]
            .mean()
            .reset_index()
            .sort_values(["player_name", "year"])
        )

        yearly_arm["prev_arm"] = yearly_arm.groupby("player_name")["arm_angle"].shift(1)
        yearly_arm["delta_arm"] = yearly_arm["arm_angle"] - yearly_arm["prev_arm"]
        yearly_arm = yearly_arm.dropna(subset=["delta_arm"])

        # Dołącz movement jeśli jest
        if has_ivb:
            yearly_ivb = df.groupby(["player_name", "year"])["ivb"].mean().reset_index()
            yearly_arm = yearly_arm.merge(yearly_ivb, on=["player_name", "year"], how="left")
            yearly_arm["prev_ivb"] = yearly_arm.groupby("player_name")["ivb"].shift(1)
            yearly_arm["delta_ivb"] = yearly_arm["ivb"] - yearly_arm["prev_ivb"]

        if has_hb:
            yearly_hb = df.groupby(["player_name", "year"])["hb"].mean().reset_index()
            yearly_arm = yearly_arm.merge(yearly_hb, on=["player_name", "year"], how="left")
            yearly_arm["prev_hb"] = yearly_arm.groupby("player_name")["hb"].shift(1)
            yearly_arm["delta_hb"] = yearly_arm["hb"] - yearly_arm["prev_hb"]

        st.markdown("#### Top 20 największych spadków Arm Angle (niższy slot)")
        top_drop = yearly_arm.nsmallest(20, "delta_arm")
        st.dataframe(
            top_drop.style.format({
                "arm_angle": "{:.1f}", "prev_arm": "{:.1f}", "delta_arm": "{:+.1f}",
                **({"ivb": "{:.1f}", "delta_ivb": "{:+.1f}"} if has_ivb else {}),
                **({"hb": "{:.1f}", "delta_hb": "{:+.1f}"} if has_hb else {}),
            }),
            use_container_width=True
        )

        st.markdown("#### Top 20 największych wzrostów Arm Angle (wyższy slot)")
        top_rise = yearly_arm.nlargest(20, "delta_arm")
        st.dataframe(
            top_rise.style.format({
                "arm_angle": "{:.1f}", "prev_arm": "{:.1f}", "delta_arm": "{:+.1f}",
                **({"ivb": "{:.1f}", "delta_ivb": "{:+.1f}"} if has_ivb else {}),
                **({"hb": "{:.1f}", "delta_hb": "{:+.1f}"} if has_hb else {}),
            }),
            use_container_width=True
        )

        # Scatter delta arm vs delta movement
        if has_ivb:
            fig = px.scatter(
                yearly_arm, x="delta_arm", y="delta_ivb",
                hover_data=["player_name", "year"],
                title="Zmiana Arm Angle vs Zmiana IVB",
                labels={"delta_arm": "Δ Arm Angle (°)", "delta_ivb": "Δ IVB (cale)"},
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# TAB 5 – Insights
# ----------------------------------------------------------
with tab5:
    st.subheader("💡 Wskazówki i proste insights")

    st.markdown("""
    ### Jak interpretować arm slot
    - **0°** ≈ pure sidearm / submarine
    - **~20–35°** ≈ low three-quarters / sidearmish
    - **~35–50°** ≈ klasyczny three-quarters
    - **~50–70°** ≈ high three-quarters / over-the-top
    - **90°** ≈ pure overhand

    ### Typowe zależności (fizyka + dane)
    - **Wyższy arm slot** → zazwyczaj więcej **IVB** (ride) na four-seam, mniej horizontal run.
    - **Niższy arm slot** → więcej **arm-side run** (HB), często lepszy sinker / two-seam, trudniejszy angle dla hitters.
    - Break piłki zwykle „podąża” za linią arm slotu (widać to na movement plotach Savant).

    ### Co sprawdzać przy zmianie slotu
    1. Czy IVB/HB poszło w oczekiwanym kierunku?
    2. Czy whiff% / xwOBA się poprawiły?
    3. Czy velocity się utrzymała?
    4. Czy nie spadła komenda (zone%).

    ### Proste reguły (do dalszej analizy)
    - Duży spadek arm angle + wzrost HB na sinkerze/sliderze często pomaga.
    - Podniesienie slotu + wzrost IVB na FF często daje więcej „ride” i whiffów up in the zone.
    - Zmiana > 8–10° to już poważna przebudowa mechaniki – warto patrzeć na consistency i zdrowie.
    """)

    if has_ivb and has_hb:
        corr_ivb = df["arm_angle"].corr(df["ivb"])
        corr_hb = df["arm_angle"].corr(df["hb"])
        st.metric("Korelacja Arm Angle ↔ IVB (cały dataset)", f"{corr_ivb:.3f}")
        st.metric("Korelacja Arm Angle ↔ HB (cały dataset)", f"{corr_hb:.3f}")

    st.markdown("---")
    st.caption("Dashboard stworzony pod dane Baseball Savant. Możesz rozbudować o AI insights (np. podłączenie LLM-a do generowania rekomendacji per miotacz).")