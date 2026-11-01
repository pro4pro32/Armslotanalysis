"""insights.py – regułowe "AI Insights": statystyki -> wnioski tekstowe po polsku (bez LLM)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import data_utils as du

CAVEAT = ("⚠️ Korelacja ≠ przyczynowość: zmiana arm slotu często idzie w parze ze zmianą mechaniki, "
          "grip'u lub repertuaru.")


def num(v, nd: int = 1, sign: bool = False) -> str:
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace(".", ",")


def strength(r: float) -> str:
    a = abs(r)
    return "bardzo słaba" if a < 0.2 else "słaba" if a < 0.4 else "umiarkowana" if a < 0.6 else "silna"


def _lbl(m):
    return du.LABELS.get(m, m).split(" (")[0]


def league_insights(d: pd.DataFrame, pt: str, metrics: list, yoy: pd.DataFrame, min_n: int = 15) -> list[str]:
    """d: wiersze miotacz×sezon dla jednego pitch_type; yoy: wynik du.yoy_all dla tego typu."""
    out: list[str] = []
    d = d[d.arm_angle.notna()]
    metrics = [m for m in metrics if m in d.columns and d[m].notna().sum() >= min_n]
    if len(d) < min_n or not metrics:
        return [f"Za mało danych dla {pt} (min. {min_n} wierszy z arm slotem i movementem), aby wyciągać wnioski."]

    # 1) Regresja metryki względem arm slotu
    for m in metrics:
        fit = du.linear_fit(d.arm_angle, d[m])
        if fit:
            out.append(f"**{_lbl(m)} ({pt})**: każdy +1° arm slotu wiąże się ze zmianą o "
                       f"{num(fit['slope'], 2, True)} {du.UNITS.get(m, '')} "
                       f"(r = {num(fit['r'], 2)}, zależność {strength(fit['r'])}, n = {fit['n']}).")

    # 2) Górna vs dolna trzecia wg arm slotu
    if "ivb" in metrics:
        q1, q2 = d.arm_angle.quantile([1 / 3, 2 / 3])
        lo, hi = d[d.arm_angle <= q1], d[d.arm_angle >= q2]
        diff = du.wmean(hi.ivb, hi.pitches) - du.wmean(lo.ivb, lo.pitches)
        word = "wyższe" if diff > 0 else "niższe"
        out.append(f"Miotacze z arm slotem ≥ {num(q2)}° mają średnio o {num(abs(diff))} cala {word} IVB na {pt} "
                   f"niż miotacze z arm slotem ≤ {num(q1)}° (górna vs dolna trzecia stawki).")

        # 3) Najlepszy bucket 5°
        b = du.league_by_bucket(d, 5, ["ivb"])
        b = b[b.pitchers >= 5]
        if len(b):
            top = b.loc[b.ivb.idxmax()]
            avg = du.wmean(d.ivb, d.pitches)
            out.append(f"Najwyższe średnie IVB na {pt} mają miotacze z arm slotem {top.bucket} "
                       f"({num(top.ivb)} cala, {num(top.ivb - avg, 1, True)} względem średniej ligowej, "
                       f"{int(top.pitchers)} miotaczy).")

    # 4) Rekomendacje na bazie historycznych zmian arm slotu
    if len(yoy) and "d_ivb" in yoy.columns:
        rules = [("Podniesienie arm slotu o 2–5°", 1, 2, 5), ("Podniesienie arm slotu o ponad 5°", 1, 5, np.inf),
                 ("Obniżenie arm slotu o 2–5°", -1, 2, 5), ("Obniżenie arm slotu o ponad 5°", -1, 5, np.inf)]
        for label, sign, lo_, hi_ in rules:
            s = yoy[(sign * yoy.d_arm >= lo_) & (sign * yoy.d_arm < hi_)]
            if len(s) < 8:
                continue
            parts = [f"IVB {num(s.d_ivb.mean(), 1, True)}"]
            for m, nm in (("d_hb", "HB"), ("d_velo", "velo"), ("d_spin", "spin")):
                if m in s.columns and s[m].notna().sum() >= 5:
                    parts.append(f"{nm} {num(s[m].mean(), 1, True)}")
            out.append(f"{label} historycznie (n = {len(s)} zmian, {pt}) korelowało ze średnią zmianą: "
                       + ", ".join(parts) + f". IVB wzrosło u {(s.d_ivb > 0).mean():.0%} miotaczy.")
        fit = du.linear_fit(yoy.d_arm, yoy.d_ivb)
        if fit and fit["n"] >= 15:
            out.append(f"Model zmian rok do roku: +1° arm slotu → {num(fit['slope'], 2, True)} cala IVB "
                       f"(r = {num(fit['r'], 2)}, {strength(fit['r'])}, n = {fit['n']}).")
    else:
        out.append("Brak par sezonów do analizy zmian rok do roku – wgraj co najmniej 2 sezony.")
    out.append(CAVEAT)
    return out


def pitcher_insights(p: pd.DataFrame, league: pd.DataFrame, pt: str, metrics: list, window: float = 2.0) -> list[str]:
    """p: wiersze jednego miotacza (jeden pitch_type); league: baza ligowa tego samego poziomu/pitch_type."""
    p = p.sort_values("season").reset_index(drop=True)
    if p.empty:
        return ["Brak danych dla wybranego miotacza i typu pitcha."]
    out: list[str] = []
    name = p.pitcher_name.iloc[0]
    metrics = [m for m in metrics if m in p.columns]

    if len(p) == 1:
        out.append(f"**{name}**: w bazie jest tylko sezon {p.season.iloc[0]} (arm slot {num(p.arm_angle.iloc[0])}°).")
    else:
        first, last = p.iloc[0], p.iloc[-1]
        d = last.arm_angle - first.arm_angle
        out.append(f"**{name}**: arm slot zmienił się z {num(first.arm_angle)}° ({first.season}) na "
                   f"{num(last.arm_angle)}° ({last.season}), czyli {num(d, 1, True)}°.")

    last = p.iloc[-1]
    ref = du.league_reference(league, last.season, pt, last.arm_angle, window, metrics, exclude_key=last.name_key)
    for m in metrics:
        refv, n = ref.get(m, (np.nan, 0))
        if pd.notna(last[m]) and pd.notna(refv) and n >= 3:
            diff = last[m] - refv
            nd = 0 if m == "spin" else 1
            word = "powyżej" if diff > 0 else "poniżej"
            out.append(f"{_lbl(m)} w {last.season}: {num(last[m], nd)} – o {num(abs(diff), nd)} {du.UNITS.get(m, '')} "
                       f"{word} średniej dla miotaczy z arm slotem {num(last.arm_angle - window, 0)}–"
                       f"{num(last.arm_angle + window, 0)}° (n = {n}).")

    # Zmiany rok do roku vs oczekiwanie z ligowej regresji
    fits = {m: du.linear_fit(league.arm_angle, league[m]) for m in metrics if m in du.CORE_METRICS}
    for a, b in zip(range(len(p) - 1), range(1, len(p))):
        r0, r1 = p.iloc[a], p.iloc[b]
        d_arm = r1.arm_angle - r0.arm_angle
        if abs(d_arm) < 1.5:
            continue
        parts = []
        for m, fit in fits.items():
            if fit and pd.notna(r0[m]) and pd.notna(r1[m]):
                nd = 0 if m == "spin" else 1
                parts.append(f"{_lbl(m)} {num(r1[m] - r0[m], nd, True)} (ligowa zależność sugeruje ≈ "
                             f"{num(fit['slope'] * d_arm, nd, True)})")
        if parts:
            out.append(f"Zmiana arm slotu {num(d_arm, 1, True)}° ({r0.season}→{r1.season}) szła w parze ze zmianą: "
                       + "; ".join(parts) + ".")

    if "ivb" in p.columns and p.ivb.notna().sum() >= 2:
        best = p.loc[p.ivb.idxmax()]
        out.append(f"Najwyższe IVB ({pt}): sezon {best.season} – {num(best.ivb)} cala przy arm slocie {num(best.arm_angle)}°.")
    out.append(CAVEAT)
    return out
