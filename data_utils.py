"""data_utils.py – wczytywanie, mapowanie kolumn, łączenie i agregacja danych o arm slocie."""
from __future__ import annotations

import io
import re
import unicodedata

import numpy as np
import pandas as pd

CORE_METRICS = ["ivb", "hb", "velo", "spin"]
OUTCOME_METRICS = ["whiff", "rv100", "xwoba"]  # opcjonalne (np. z Pitch Arsenal Stats)
LABELS = {
    "arm_angle": "Arm slot (°)", "ivb": "IVB (cale)", "hb": "HB (cale)",
    "velo": "Velo (mph)", "spin": "Spin (rpm)", "whiff": "Whiff %",
    "rv100": "RV/100", "xwoba": "xwOBA", "pitches": "Pitche", "pitchers": "Miotaczy",
}
UNITS = {"ivb": "cala", "hb": "cala", "velo": "mph", "spin": "rpm", "whiff": "p.p.", "rv100": "", "xwoba": ""}
PITCH_ORDER = ["FF", "SI", "FC", "SL", "ST", "CU", "CH", "FS"]
KIND_PL = {"arm": "surowy arm angle", "movement": "movement/arsenal", "enriched": "enriched (arm + movement)"}

# Nazwy kolumn (po zamianie na małe litery) -> nazwa standardowa
ALIASES = {
    "arm_angle": ["arm_angle", "ball_angle", "arm_slot", "armangle"],
    "pitcher_name": ["pitcher_name", "player_name", "name", "last_name, first_name", "last_name,first_name"],
    "pitcher_id": ["pitcher_id", "pitcher", "player_id", "entity_id", "mlbam_id", "mlb_id"],
    "pitch_hand": ["pitch_hand", "p_throws", "throws", "pitcher_hand", "hand"],
    "team": ["team_id", "team_name", "team_name_alt", "team"],
    "pitches": ["pitches", "pitches_thrown", "pitch_count", "n_pitches", "count"],
    "pitch_type": ["pitch_type", "pitch_code", "pitch_type_abbr"],
    "pitch_type_name": ["pitch_type_name", "pitch_name", "pitch_type_long"],
    "season": ["season", "year", "game_year"],
    "ivb": ["ivb", "induced_vertical_break", "pitcher_break_z_induced", "pfx_z_induced",
            "induced_vert_break", "vert_break_induced", "ivb_in"],
    "hb": ["hb", "horizontal_break", "pitcher_break_x", "pfx_x_in", "horz_break", "hb_in"],
    "velo": ["velo", "velocity", "avg_speed", "release_speed", "avg_velo", "avg_velocity"],
    "spin": ["spin", "spin_rate", "avg_spin", "release_spin_rate", "avg_spin_rate"],
    "whiff": ["whiff_percent", "whiff_pct", "whiff"],
    "rv100": ["run_value_per100", "rv_100", "rv100", "run_value_per_100"],
    "xwoba": ["est_woba", "xwoba", "estimated_woba"],
}
NUMERIC = ["arm_angle", "pitches", "ivb", "hb", "velo", "spin", "whiff", "rv100", "xwoba",
           "release_ball_z", "relative_release_ball_x", "pitcher_id"]

PITCH_MAP = {
    "4-seam fastball": "FF", "four-seam fastball": "FF", "4-seam": "FF", "fastball": "FF", "fa": "FF", "ff": "FF",
    "sinker": "SI", "si": "SI", "2-seam fastball": "SI", "two-seam fastball": "SI", "ft": "SI",
    "slider": "SL", "sl": "SL", "changeup": "CH", "ch": "CH", "curveball": "CU", "cu": "CU",
    "knuckle curve": "KC", "kc": "KC", "cutter": "FC", "fc": "FC", "sweeper": "ST", "st": "ST",
    "split-finger": "FS", "splitter": "FS", "fs": "FS", "fo": "FS", "slurve": "SV", "sv": "SV",
}


# ----------------------------------------------------------------------------- nazwy
def normalize_name(name) -> str:
    """'Skubal, Tarik' / 'Tarik Skubal Jr.' -> 'tarik skubal' (klucz do matchowania)."""
    if pd.isna(name):
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().strip()
    if "," in s:
        last, _, first = s.partition(",")
        s = f"{first.strip()} {last.strip()}"
    s = re.sub(r"[.'’`\-]", " ", s)
    tokens = [t for t in s.split() if t not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(tokens)


def display_name(name) -> str:
    """'Skubal, Tarik' -> 'Tarik Skubal' (ładna nazwa do wyświetlania)."""
    if pd.isna(name):
        return ""
    s = str(name).strip()
    if "," in s:
        last, _, first = s.partition(",")
        s = f"{first.strip()} {last.strip()}"
    return re.sub(r"\s+", " ", s)


def normalize_pitch_type(v) -> str:
    if pd.isna(v):
        return "ALL"
    s = str(v).strip()
    return PITCH_MAP.get(s.lower(), s.upper())


# ----------------------------------------------------------------------------- wczytywanie
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Nie udało się odczytać pliku (nieznane kodowanie).")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c).strip().lower().replace("\ufeff", "")) for c in df.columns]
    rename = {}
    for std, names in ALIASES.items():
        if std in df.columns:
            continue
        for n in names:
            if n in df.columns and n not in rename:
                rename[n] = std
                break
    return df.rename(columns=rename)


def detect_season(filename: str):
    m = re.search(r"(20\d{2})", filename)
    return int(m.group(1)) if m else None


def _to_inches(df: pd.DataFrame, col: str, msgs: list, fname: str):
    s = df[col].dropna()
    if len(s) and s.abs().quantile(0.95) <= 3.5:  # prawdopodobnie stopy (Statcast pfx)
        df[col] = df[col] * 12
        msgs.append(("info", f"`{fname}`: kolumna `{col}` wyglądała na stopy – przeliczono na cale (×12)."))


def prepare_frame(raw: pd.DataFrame, filename: str, msgs: list):
    """Standaryzuje ramkę i zwraca (df, rodzaj). Rodzaj: arm / movement / enriched."""
    df = standardize_columns(raw)
    if "pitcher_name" not in df.columns:
        if {"first_name", "last_name"} <= set(df.columns):
            df["pitcher_name"] = df["first_name"].astype(str).str.strip() + " " + df["last_name"].astype(str).str.strip()
        else:
            raise ValueError("brak kolumny z nazwiskiem miotacza (player_name / pitcher_name / 'last_name, first_name').")
    if "season" not in df.columns:
        s = detect_season(filename)
        if s is None:
            raise ValueError("brak kolumny sezonu – dodaj rok do nazwy pliku (np. `arm_2025.csv`).")
        df["season"] = s
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df = df.dropna(subset=["season"])
    df["season"] = df["season"].astype(int)
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["name_key"] = df["pitcher_name"].map(normalize_name)
    df["pitcher_name"] = df["pitcher_name"].map(display_name)

    if "pitch_type" in df.columns:
        df["pitch_type"] = df["pitch_type"].map(normalize_pitch_type)
    elif "pitch_type_name" in df.columns:
        df["pitch_type"] = df["pitch_type_name"].map(normalize_pitch_type)
    else:
        df["pitch_type"] = "ALL"

    has = lambda c: c in df.columns and df[c].notna().any()  # noqa: E731
    has_arm, has_mov = has("arm_angle"), any(has(m) for m in CORE_METRICS)
    if has_arm and has_mov:
        kind = "enriched"
    elif has_arm:
        kind = "arm"
    elif has_mov:
        kind = "movement"
    else:
        raise ValueError("nie znaleziono ani arm_angle/ball_angle, ani IVB/HB/velo/spin – sprawdź nagłówki kolumn.")

    if kind != "arm":
        for c in ("ivb", "hb"):
            if has(c):
                _to_inches(df, c, msgs, filename)
    if kind == "arm" and "pitches" in df.columns:
        df = df.rename(columns={"pitches": "pitches_total"})
    return df, kind


# ----------------------------------------------------------------------------- łączenie
def combine_movement(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Skleja pliki movement + arsenal (różne kolumny, ten sam klucz sezon/miotacz/pitch)."""
    m = pd.concat(frames, ignore_index=True)
    return m.groupby(["season", "name_key", "pitch_type"], as_index=False).first()


def merge_arm(arm: pd.DataFrame, mov: pd.DataFrame):
    sort_col = "pitches_total" if "pitches_total" in arm.columns else "season"
    arm_u = arm.sort_values(sort_col, ascending=False).drop_duplicates(["season", "name_key"])
    out = mov.copy().drop(columns=["arm_angle"], errors="ignore")

    out["arm_angle_id"] = np.nan
    if "pitcher_id" in arm_u.columns and "pitcher_id" in out.columns:
        right = (arm_u.dropna(subset=["pitcher_id"])[["season", "pitcher_id", "arm_angle"]]
                 .drop_duplicates(["season", "pitcher_id"]).rename(columns={"arm_angle": "arm_angle_id"}))
        out = out.drop(columns=["arm_angle_id"]).merge(right, on=["season", "pitcher_id"], how="left")

    extra = [c for c in ["pitch_hand", "team", "release_ball_z", "relative_release_ball_x"] if c in arm_u.columns]
    cols = ["season", "name_key", "arm_angle"] + extra
    right = arm_u[cols].rename(columns={c: f"{c}_arm" for c in ["arm_angle"] + extra})
    out = out.merge(right, on=["season", "name_key"], how="left")
    out["arm_angle"] = out["arm_angle_id"].fillna(out["arm_angle_arm"])
    for c in extra:
        out[c] = out[c].fillna(out[f"{c}_arm"]) if c in out.columns else out[f"{c}_arm"]
    out = out.drop(columns=["arm_angle_id", "arm_angle_arm"] + [f"{c}_arm" for c in extra])
    return out, float(out["arm_angle"].notna().mean())


def build_dataset(sources: list):
    """sources: lista (nazwa_pliku, bytes | DataFrame). Zwraca (df, komunikaty)."""
    msgs: list[tuple[str, str]] = []
    buckets = {"arm": [], "movement": [], "enriched": []}
    for name, data in sources:
        try:
            raw = read_csv_bytes(data) if isinstance(data, (bytes, bytearray)) else data
            df, kind = prepare_frame(raw, name, msgs)
        except Exception as e:  # noqa: BLE001
            msgs.append(("error", f"`{name}`: {e}"))
            continue
        buckets[kind].append(df)
        seasons = ", ".join(map(str, sorted(df.season.unique())))
        msgs.append(("info", f"`{name}` → **{KIND_PL[kind]}**, wierszy: {len(df):,}, sezon: {seasons}"))

    parts = list(buckets["enriched"])
    arm_df = pd.concat(buckets["arm"], ignore_index=True) if buckets["arm"] else None

    if buckets["movement"]:
        mov = combine_movement(buckets["movement"])
        if arm_df is not None:
            merged, rate = merge_arm(arm_df, mov)
            lvl = "success" if rate > 0.9 else "warning"
            msgs.append((lvl, f"Dopasowano arm slot do {rate:.0%} wierszy movement."))
            if rate < 1:
                miss = merged[merged.arm_angle.isna()].pitcher_name.drop_duplicates().head(8).tolist()
                msgs.append(("warning", "Bez arm slotu m.in.: " + ", ".join(miss)))
            parts.append(merged)
        else:
            msgs.append(("warning", "Wgrano movement, ale brak plików arm angle – arm slot będzie pusty."))
            parts.append(mov.assign(arm_angle=np.nan))

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if arm_df is not None:  # sezony, dla których mamy tylko arm angle
        covered = set(out.season.unique()) if len(out) else set()
        a = arm_df[~arm_df.season.isin(covered)].copy()
        if len(a):
            a["pitches"] = a.get("pitches_total")
            a["pitch_type"] = "ALL"
            msgs.append(("warning", "Sezony tylko z arm angle (bez movement): "
                         + ", ".join(map(str, sorted(a.season.unique()))) + "."))
            out = pd.concat([out, a], ignore_index=True)
    if out.empty:
        return out, msgs

    for c in ["pitches", "pitch_hand", "team", "pitcher_id"] + CORE_METRICS + ["arm_angle"]:
        if c not in out.columns:
            out[c] = np.nan
    out = out.drop_duplicates(["season", "name_key", "pitch_type"], keep="first")
    return out.reset_index(drop=True), msgs


def to_enriched(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["season", "pitcher_id", "pitcher_name", "pitch_hand", "team", "pitch_type", "pitches",
            "arm_angle", "ivb", "hb", "velo", "spin"] + [c for c in OUTCOME_METRICS + ["release_ball_z",
            "relative_release_ball_x"] if c in df.columns]
    return df[[c for c in cols if c in df.columns]].sort_values(["season", "pitcher_name", "pitch_type"])


# ----------------------------------------------------------------------------- agregacje
def wmean(values, weights) -> float:
    v = pd.to_numeric(pd.Series(values), errors="coerce").reset_index(drop=True)
    w = pd.Series(weights).reset_index(drop=True) if weights is not None else pd.Series(1.0, index=v.index)
    w = w.where(w > 0).fillna(1.0)
    m = v.notna()
    return float(np.average(v[m], weights=w[m])) if m.any() else np.nan


def weighted_agg(df: pd.DataFrame, by: list, value_cols: list) -> pd.DataFrame:
    """Średnie ważone liczbą pitchy + liczba pitchy i miotaczy w grupie."""
    d = df.copy()
    if "pitches" in d.columns:
        w = d["pitches"].where(d["pitches"] > 0).fillna(1.0)
    else:
        w = pd.Series(1.0, index=d.index)
    for c in value_cols:
        d[f"{c}__n"] = d[c] * w
        d[f"{c}__d"] = w.where(d[c].notna(), 0.0)
    g = d.groupby(by, observed=True)
    res = pd.DataFrame(index=g.size().index)
    for c in value_cols:
        den = g[f"{c}__d"].sum()
        res[c] = g[f"{c}__n"].sum() / den.where(den > 0)
    res["pitches"] = g["pitches"].sum(min_count=1) if "pitches" in d.columns else g.size()
    res["pitchers"] = g["name_key"].nunique()
    return res.reset_index()


def prepare_level(df: pd.DataFrame, level: str, metrics: list) -> pd.DataFrame:
    """level='type' -> wiersz = miotacz×sezon×pitch_type; 'overall' -> miotacz×sezon."""
    if level != "overall":
        return df.copy()
    cols = ["arm_angle"] + [m for m in metrics if m in df.columns]
    agg = weighted_agg(df, ["season", "name_key"], cols)
    meta_cols = [c for c in ["pitcher_name", "pitch_hand", "team"] if c in df.columns]
    meta = df.groupby(["season", "name_key"], as_index=False)[meta_cols].first()
    out = agg.merge(meta, on=["season", "name_key"], how="left").drop(columns=["pitchers"])
    out["pitch_type"] = "ALL"
    return out


def add_buckets(df: pd.DataFrame, width: int) -> pd.DataFrame:
    d = df[df["arm_angle"].notna()].copy()
    lo = (np.floor(d["arm_angle"] / width) * width).astype(int)
    d["bucket_low"] = lo
    d["bucket"] = lo.astype(str) + "–" + (lo + width).astype(str) + "°"
    return d


def league_by_bucket(df: pd.DataFrame, width: int, metrics: list, extra_by: list | None = None) -> pd.DataFrame:
    d = add_buckets(df, width)
    by = ["bucket_low", "bucket"] + list(extra_by or [])
    cols = [m for m in metrics if m in d.columns]
    return weighted_agg(d, by, cols).sort_values(by)


def linear_fit(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").reset_index(drop=True)
    y = pd.to_numeric(pd.Series(y), errors="coerce").reset_index(drop=True)
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope": float(slope), "intercept": float(intercept),
            "r": float(np.corrcoef(x, y)[0, 1]), "n": int(len(x))}


def yoy_changes(d: pd.DataFrame, s_from: int, s_to: int, metrics: list, min_pitches: float = 0) -> pd.DataFrame:
    """Zmiany między dwoma sezonami. `d` musi dotyczyć jednego pitch_type (lub 'ALL')."""
    metrics = [m for m in metrics if m in d.columns]
    cols = ["name_key", "pitcher_name", "pitch_hand", "arm_angle", "pitches"] + metrics
    a = d[d.season == s_from][cols]
    b = d[d.season == s_to][cols]
    j = a.merge(b, on="name_key", suffixes=("_from", "_to"))
    if j.empty:
        return j
    j["pitcher_name"] = j["pitcher_name_to"]
    j["pitch_hand"] = j["pitch_hand_to"]
    keep = (j["pitches_from"].fillna(min_pitches) >= min_pitches) & (j["pitches_to"].fillna(min_pitches) >= min_pitches)
    j = j[keep & j["arm_angle_from"].notna() & j["arm_angle_to"].notna()].copy()
    j["d_arm"] = j["arm_angle_to"] - j["arm_angle_from"]
    for m in metrics:
        j[f"d_{m}"] = j[f"{m}_to"] - j[f"{m}_from"]
    j["pair"] = f"{s_from}→{s_to}"
    return j


def yoy_all(d: pd.DataFrame, metrics: list, min_pitches: float = 0, seasons: list | None = None) -> pd.DataFrame:
    ss = sorted(seasons if seasons is not None else d.season.unique())
    frames = [yoy_changes(d, a, b, metrics, min_pitches) for a, b in zip(ss[:-1], ss[1:])]
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def league_reference(base: pd.DataFrame, season: int, pitch_type: str, arm: float, window: float,
                     metrics: list, exclude_key: str | None = None) -> dict:
    """Średnia ligowa dla miotaczy o arm slocie w przedziale arm ± window (ten sam sezon i pitch)."""
    d = base[(base.season == season) & (base.pitch_type == pitch_type)
             & ((base.arm_angle - arm).abs() <= window)]
    if exclude_key:
        d = d[d.name_key != exclude_key]
    out = {}
    for m in metrics:
        if m not in d.columns:
            continue
        s = d[m].notna()
        out[m] = (wmean(d.loc[s, m], d.loc[s, "pitches"]), int(d.loc[s, "name_key"].nunique()))
    return out
