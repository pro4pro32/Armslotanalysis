"""enrich_data.py – łączy surowe arm angle + movement/arsenal w pliki enriched_<rok>.csv.

Użycie:
    python enrich_data.py --input data/raw --output data
(w Jupyterze: z data_utils import build_dataset, to_enriched i wywołaj tak jak niżej w main())
"""
import argparse
from pathlib import Path

import data_utils as du


def main(input_dir: str, output_dir: str) -> None:
    files = sorted(Path(input_dir).glob("*.csv"))
    if not files:
        raise SystemExit(f"Brak plików CSV w {input_dir}")
    df, msgs = du.build_dataset([(p.name, p.read_bytes()) for p in files])
    for lvl, m in msgs:
        print(f"[{lvl.upper()}] {m}")
    if df.empty:
        raise SystemExit("Nie powstał żaden zbiór danych.")
    out = du.to_enriched(df)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for season, g in out.groupby("season"):
        path = Path(output_dir) / f"enriched_{season}.csv"
        g.to_csv(path, index=False)
        cov = g[["arm_angle", "ivb", "hb", "velo", "spin"]].notna().mean().round(2).to_dict()
        print(f"✔ {path}  wierszy: {len(g)}  pokrycie kolumn: {cov}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw", help="folder z surowymi CSV (arm angle + movement/arsenal)")
    ap.add_argument("--output", default="data", help="folder docelowy na enriched_<rok>.csv")
    a = ap.parse_args()
    main(a.input, a.output)
