import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

st.set_page_config(
    page_title="Arm Slot Analytics Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚾ Arm Slot Analytics Dashboard")
st.markdown("Analiza **arm slotu** miotaczy (dane z Baseball Savant Arm Angle Leaderboard)")

# ============================================================
# NORMALIZACJA KOLUMN (dostosowana do Twoich CSV)
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    col_map = {
        # Arm angle
        "ball_angle": "arm_angle",
        "arm_angle": "arm_angle",
        "Arm Angle": "arm_angle",

        # Player
        "pitcher_name": "player_name",
        "player_name": "player_name",
        "pitcher": "pitcher_id",

        # Pitches
        "n_pitches": "pitches",
        "pitches": "pitches",

        # Hand
        "pitch_hand": "pitch_hand",

        # Release points
        "relative_release_ball_x": "release_x",
        "release_ball_z": "release_z",
        "relative_shoulder_x": "shoulder_x",
        "shoulder_z": "shoulder_z",

        # Team
        "team_id": "team_id",
    }

    rename_dict = {}
    for orig, new in col_map.items():
        for col in df.columns:
            if col.lower() == orig.lower():
                rename_dict[col] = new
                break

    df = df.rename(columns=rename_dict)

    # Typy
    if "arm_angle" in df.columns:
        df["arm_angle"] = pd.to_numeric(df["arm_angle"], errors="coerce")
    if "pitches" in df.columns:
        df["pitches"] = pd.to_numeric(df["pitches"], errors="coerce")
    if "release_z" in df.columns:
        df["release_z"] = pd.to_numeric(df["release_z"], errors="coerce")
    if "release_x" in df.columns:
        df["release_x"] = pd.to_numeric(df["release_x"], errors="coerce")

    return df


def create_arm_buckets(series: pd.Series, bin_size: float = 2.0) -> pd.Series:
    """Buckety arm slotu np. 44-46, 46-48"""
    min_val = np.floor(series.min() / bin_size) * bin_size
    max_val = np.ceil(series.max() / bin_size) * bin_size
    bins = np.arange(min_val, max_val + bin_size, bin_size)
    labels = [f"{int(b)}–{int(b + bin_size)}" for b in bins[:-1]]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("📂 Dane")
    uploaded_files = st.file_uploader(
        "Wrzuć CSV (2024 / 2025 / 2026)",
        type=["csv"],
        accept_multiple_files=True
    )

    st.markdown("---")
    st.header("⚙️ Ustawienia")

    bin_size = st.slider("Rozmiar bucketu arm slotu (°)", 1.0, 5.0, 2.0, 0.5)
    min_pitches = st.number_input("Min. liczba pitchy", 0, 2000, 100, 50)

    st.markdown("---")
    st.info(
        "Te pliki zawierają **arm angle + release point**.\n\n"
        "Do analizy IVB / HB / whiff% potrzebujesz dodatkowo "
        "CSV z Pitch Movement lub Custom Leaderboard."
    )

# ============================================================
# ŁADOWANIE DANYCH
# ============================================================

if not uploaded_files:
    st.info("👈 Wrzuć pliki CSV z Baseball Savant (Arm Angle Leaderboard).")
    st.stop()

dfs = []
for f in uploaded_files:
    try:
        temp = pd.read_csv(f)
        temp = normalize_columns(temp)
        temp["_source"] = f.name

        # Rok z nazwy pliku
        year_match = re.search(r"20\d{2}", f.name)
        if year_match:
            temp["year"] = int(year_match.group())
        else:
            temp["year"] = None

        dfs.append(temp)
    except Exception as e:
        st.error(f"Błąd przy pliku **{f.name}**: {e}")

if not dfs:
    st.error("Nie udało się wczytać żadnego pliku.")
    st.stop()

df = pd.concat(dfs, ignore_index=True)

# Sprawdzenie
if "arm_angle" not in df.columns:
    st.error("Brak kolumny arm_angle / ball_angle po normalizacji.")
    st.write("Dostępne kolumny:", list(df.columns))
    st.stop()

# Filtrowanie
df = df.dropna(subset=["arm_angle"])
if "pitches" in df.columns:
    df = df[df["pitches"].fillna(0) >= min_pitches]

df["arm_bucket"] = create_arm_buckets(df["arm_angle"], bin_size)

has_player = "player_name" in df.columns
has_year = "year" in df.columns and df["year"].notna().any()
has_hand = "pitch_hand" in df.columns
has_release = "release_z" in df.columns and "release_x" in df.columns

st.success(
    f"Wczytano **{len(df):,}** wierszy z **{len(uploaded_files)}** pliku/ów "
    f"| Sezony: {sorted(df['year'].dropna().unique()) if has_year else 'brak'}"
)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview & Buckets",
    "📈 Arm Slot Distribution",
    "👤 Indywidualni miotacze",
    "🔄 Największe zmiany",
    "💡 Insights"
])

# ----------------------------------------------------------
# TAB 1 – Overview & Buckets
# ----------------------------------------------------------
with tab1:
    st.subheader("Średnie wartości według Arm Slot Bucket")

    agg = {
        "arm_angle": "mean",
        "pitches": "sum" if "pitches" in df.columns else "count"
    }
    if has_release:
        agg["release_z"] = "mean"
        agg["release_x"] = "mean"

    bucket_stats = (
        df.groupby("arm_bucket", observed=True)
        .agg(agg)
        .reset_index()
        .sort_values("arm_bucket")
    )

    # Liczba miotaczy w buckecie
    pitcher_counts = df.groupby("arm_bucket", observed=True)["player_name"].nunique().reset_index(name="n_pitchers")
    bucket_stats = bucket_stats.merge(pitcher_counts, on="arm_bucket", how="left")

    format_dict = {
        "arm_angle": "{:.1f}",
        "pitches": "{:.0f}",
        "n_pitchers": "{:.0f}",
    }
    if has_release:
        format_dict["release_z"] = "{:.2f}"
        format_dict["release_x"] = "{:.2f}"

    st.dataframe(
        bucket_stats.style.format(format_dict),
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            bucket_stats,
            x="arm_bucket",
            y="n_pitchers",
            title="Liczba miotaczy w każdym Arm Slot Bucket",
            labels={"arm_bucket": "Arm Slot Bucket", "n_pitchers": "Liczba miotaczy"},
            color="n_pitchers",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if has_release:
            fig = px.bar(
                bucket_stats,
                x="arm_bucket",
                y="release_z",
                title="Średnia wysokość release (ft) vs Arm Slot",
                labels={"arm_bucket": "Arm Slot Bucket", "release_z": "Release Height (ft)"},
                color="release_z",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.bar(
                bucket_stats,
                x="arm_bucket",
                y="arm_angle",
                title="Średni Arm Angle w buckecie",
                labels={"arm_bucket": "Arm Slot Bucket", "arm_angle": "Arm Angle (°)"},
            )
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# TAB 2 – Distribution
# ----------------------------------------------------------
with tab2:
    st.subheader("Rozkład Arm Slotu")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x="arm_angle",
            nbins=40,
            title="Histogram Arm Angle",
            labels={"arm_angle": "Arm Angle (°)"},
            color_discrete_sequence=["#1f77b4"]
        )
        fig.add_vline(x=df["arm_angle"].mean(), line_dash="dash", annotation_text=f"Średnia: {df['arm_angle'].mean():.1f}°")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if has_hand:
            fig = px.box(
                df,
                x="pitch_hand",
                y="arm_angle",
                title="Arm Angle według ręki (L/R)",
                labels={"pitch_hand": "Ręka", "arm_angle": "Arm Angle (°)"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Brak kolumny pitch_hand")

    # Scatter: release height vs arm angle
    if has_release:
        st.subheader("Release Height vs Arm Angle")
        fig = px.scatter(
            df,
            x="arm_angle",
            y="release_z",
            color="pitch_hand" if has_hand else None,
            hover_data=["player_name", "year", "pitches"] if has_player else None,
            title="Wysokość release vs Arm Slot",
            labels={"arm_angle": "Arm Angle (°)", "release_z": "Release Height (ft)"},
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Release Side vs Arm Angle")
        fig = px.scatter(
            df,
            x="arm_angle",
            y="release_x",
            color="pitch_hand" if has_hand else None,
            hover_data=["player_name", "year"] if has_player else None,
            title="Boczna pozycja release vs Arm Slot",
            labels={"arm_angle": "Arm Angle (°)", "release_x": "Release Side (ft)"},
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# TAB 3 – Indywidualni miotacze
# ----------------------------------------------------------
with tab3:
    st.subheader("Analiza indywidualna miotacza")

    if not has_player:
        st.warning("Brak kolumny z nazwiskiem miotacza.")
    else:
        players = sorted(df["player_name"].dropna().unique())
        selected_player = st.selectbox("Wybierz miotacza", players)

        player_df = df[df["player_name"] == selected_player].sort_values("year")

        st.markdown(f"### {selected_player}")

        # Metryki
        cols = st.columns(4)
        cols[0].metric("Średni Arm Angle", f"{player_df['arm_angle'].mean():.1f}°")
        cols[1].metric("Min Arm Angle", f"{player_df['arm_angle'].min():.1f}°")
        cols[2].metric("Max Arm Angle", f"{player_df['arm_angle'].max():.1f}°")
        if "pitches" in player_df.columns:
            cols[3].metric("Łącznie pitchy", f"{int(player_df['pitches'].sum()):,}")

        # Tabela sezonowa
        st.subheader("Dane sezon po sezonie")
        display_cols = ["year", "arm_angle", "pitches", "pitch_hand"]
        if has_release:
            display_cols += ["release_z", "release_x"]
        display_cols = [c for c in display_cols if c in player_df.columns]

        st.dataframe(
            player_df[display_cols].style.format({
                "arm_angle": "{:.1f}",
                "pitches": "{:.0f}",
                "release_z": "{:.2f}",
                "release_x": "{:.2f}",
            }),
            use_container_width=True
        )

        # Wykres w czasie
        if has_year and player_df["year"].nunique() > 1:
            st.subheader("Zmiana Arm Angle w czasie")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=player_df["year"],
                y=player_df["arm_angle"],
                mode="lines+markers+text",
                text=[f"{v:.1f}°" for v in player_df["arm_angle"]],
                textposition="top center",
                name="Arm Angle",
                line=dict(width=3)
            ))
            fig.update_layout(
                title=f"{selected_player} – Arm Angle po sezonach",
                xaxis_title="Sezon",
                yaxis_title="Arm Angle (°)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            # Delta
            if len(player_df) >= 2:
                first = player_df.iloc[0]["arm_angle"]
                last = player_df.iloc[-1]["arm_angle"]
                delta = last - first
                st.metric(
                    f"Zmiana {int(player_df.iloc[0]['year'])} → {int(player_df.iloc[-1]['year'])}",
                    f"{delta:+.1f}°",
                    delta_color="inverse"
                )

# ----------------------------------------------------------
# TAB 4 – Największe zmiany
# ----------------------------------------------------------
with tab4:
    st.subheader("Miotacze z największymi zmianami Arm Slotu")

    if not (has_player and has_year):
        st.warning("Potrzebne kolumny player_name + year (z nazwy pliku).")
    else:
        yearly = (
            df.groupby(["player_name", "year"])
            .agg({
                "arm_angle": "mean",
                "pitches": "sum" if "pitches" in df.columns else "count",
                **({"release_z": "mean"} if has_release else {}),
            })
            .reset_index()
            .sort_values(["player_name", "year"])
        )

        yearly["prev_arm"] = yearly.groupby("player_name")["arm_angle"].shift(1)
        yearly["delta_arm"] = yearly["arm_angle"] - yearly["prev_arm"]
        yearly = yearly.dropna(subset=["delta_arm"])

        if yearly.empty:
            st.warning("Brak miotaczy z danymi w więcej niż jednym sezonie (po filtrze min. pitchy).")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### ⬇️ Top 25 największych **spadków** Arm Angle")
                top_drop = yearly.nsmallest(25, "delta_arm")[
                    ["player_name", "year", "prev_arm", "arm_angle", "delta_arm", "pitches"]
                ]
                st.dataframe(
                    top_drop.style.format({
                        "prev_arm": "{:.1f}",
                        "arm_angle": "{:.1f}",
                        "delta_arm": "{:+.1f}",
                        "pitches": "{:.0f}",
                    }),
                    use_container_width=True
                )

            with col2:
                st.markdown("#### ⬆️ Top 25 największych **wzrostów** Arm Angle")
                top_rise = yearly.nlargest(25, "delta_arm")[
                    ["player_name", "year", "prev_arm", "arm_angle", "delta_arm", "pitches"]
                ]
                st.dataframe(
                    top_rise.style.format({
                        "prev_arm": "{:.1f}",
                        "arm_angle": "{:.1f}",
                        "delta_arm": "{:+.1f}",
                        "pitches": "{:.0f}",
                    }),
                    use_container_width=True
                )

            # Histogram zmian
            st.subheader("Rozkład zmian Arm Angle (sezon do sezonu)")
            fig = px.histogram(
                yearly,
                x="delta_arm",
                nbins=40,
                title="Δ Arm Angle (bieżący sezon − poprzedni)",
                labels={"delta_arm": "Zmiana Arm Angle (°)"},
                color_discrete_sequence=["#ff7f0e"]
            )
            fig.add_vline(x=0, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)

            # Najwięksi "movers"
            st.subheader("Najwięksi movers (absolutna zmiana ≥ 5°)")
            big_movers = yearly[yearly["delta_arm"].abs() >= 5].sort_values("delta_arm")
            st.dataframe(
                big_movers[["player_name", "year", "prev_arm", "arm_angle", "delta_arm", "pitches"]]
                .style.format({
                    "prev_arm": "{:.1f}",
                    "arm_angle": "{:.1f}",
                    "delta_arm": "{:+.1f}",
                    "pitches": "{:.0f}",
                }),
                use_container_width=True
            )

# ----------------------------------------------------------
# TAB 5 – Insights
# ----------------------------------------------------------
with tab5:
    st.subheader("💡 Wskazówki interpretacyjne")

    st.markdown("""
    ### Skala Arm Angle (Baseball Savant)
    | Zakres | Typowy slot |
    |--------|-------------|
    | **< 10°** | Submarine / pure sidearm |
    | **10–25°** | Low three-quarters / sidearmish |
    | **25–40°** | Klasyczny three-quarters |
    | **40–55°** | High three-quarters |
    | **> 55°** | Over-the-top |

    ### Co widać w tych danych
    - **Wyższy arm slot** zwykle = wyższy `release_z` (wysokość wypuszczenia).
    - Lewacy i prawacy mają lekko inne rozkłady (lewa strona release_x jest odwrotna).
    - Duża zmiana arm angle (> 6–8°) to już poważna zmiana mechaniki.

    ### Co dalej możesz dodać
    1. Pobierz z Savant **Pitch Movement** lub **Custom Leaderboard** (IVB, HB, velo, whiff%).
    2. Wrzuć te pliki razem z obecnymi – dashboard automatycznie je wykryje po rozbudowie mapowania.
    3. Porównaj Δ arm angle z Δ IVB / Δ HB – to pokazuje, czy zmiana slotu faktycznie zmieniła tor lotu.
    """)

    # Szybkie statystyki
    st.subheader("Szybkie statystyki z obecnych danych")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Średni Arm Angle (liga)", f"{df['arm_angle'].mean():.1f}°")
    c2.metric("Mediana", f"{df['arm_angle'].median():.1f}°")
    c3.metric("Min", f"{df['arm_angle'].min():.1f}°")
    c4.metric("Max", f"{df['arm_angle'].max():.1f}°")

    if has_hand:
        st.markdown("#### Średni Arm Angle według ręki")
        hand_stats = df.groupby("pitch_hand")["arm_angle"].agg(["mean", "median", "count"]).round(1)
        st.dataframe(hand_stats, use_container_width=True)

st.caption("Dashboard dostosowany do plików Arm Angle Leaderboard (ball_angle → arm_angle). Dane: Baseball Savant / MLB.")