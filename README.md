# ⚾ Arm Slot Lab – dashboard Streamlit

## Struktura projektu
```
arm_slot_dashboard/
├── app.py            # dashboard (6 zakładek, filtry w sidebarze)
├── data_utils.py     # mapowanie kolumn, normalizacja nazw, łączenie, agregacje
├── insights.py       # regułowe "AI Insights"
├── enrich_data.py    # opcjonalny skrypt: arm angle + movement -> enriched_<rok>.csv
├── requirements.txt
└── data/             # tu wrzucasz CSV (wczytywane automatycznie)
    └── raw/          # (opcjonalnie) surowe pliki dla enrich_data.py
```

## Krok 1 – instalacja
```bash
cd arm_slot_dashboard
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Krok 2 – dane
Masz 3 równoważne opcje (można je mieszać, także per sezon):

**A. Surowe arm angle + pliki movement z Savant** – wrzuć wszystko do `data/`.
Dashboard sam rozpozna rodzaj pliku i połączy je po `(sezon, pitcher_id)`, a gdy brak ID – po
znormalizowanej nazwie ("Skubal, Tarik" = "Tarik Skubal").

**B. Gotowe `enriched_2024.csv` itd.** (arm_angle + ivb + hb + velo + spin + pitch_type) – wrzuć do `data/`.

**C. Upload z poziomu aplikacji** – panel boczny → "Wgraj pliki CSV".

### Pobieranie brakujących danych z Baseball Savant
(Nazwy leaderboardów i URL-e mogą się zmienić – w razie czego znajdź je w menu *Leaderboards*.)

1. **Pitch Movement** – https://baseballsavant.mlb.com/leaderboard/pitch-movement
   Ustaw rok i minimalną liczbę pitchy, wybierz typ(y) pitcha i kliknij **Download CSV**.
   Daje m.in. `pitcher_break_z_induced` (= IVB), `pitcher_break_x` (= HB), `avg_speed`.
   Jeśli CSV obejmuje jeden typ pitcha, pobierz każdy osobno i wgraj wszystkie – kod je skleja.
2. **Spin (jeśli nie ma go w pliku movement)** – leaderboard **Pitch Arsenals**
   (https://baseballsavant.mlb.com/leaderboard/pitch-arsenals): `avg_speed`, `avg_spin` per typ pitcha.
3. **Wyniki (opcjonalnie)** – **Pitch Arsenal Stats** (`whiff_percent`, `run_value_per100`, `est_woba`).
   Dashboard je rozpozna i doda jako dodatkowe metryki.
4. **Nazwij pliki z rokiem**, np. `movement_2025.csv`, `arsenal_2025.csv`
   (rok jest brany z kolumny `year`/`season`, a gdy jej brak – z nazwy pliku).

Rozpoznawane nagłówki (wielkość liter bez znaczenia): `ball_angle/arm_angle/arm_slot`,
`player_name/pitcher_name/"last_name, first_name"`, `pitcher/pitcher_id/player_id`,
`pitcher_break_z_induced/ivb/induced_vertical_break`, `pitcher_break_x/hb/horizontal_break`,
`avg_speed/release_speed/velo`, `avg_spin/release_spin_rate/spin`, `pitches/pitches_thrown`.
Nowe aliasy dopiszesz w słowniku `ALIASES` w `data_utils.py`.
Dane w stopach (Statcast `pfx`) są automatycznie przeliczane na cale.

## Krok 3 – (opcjonalnie) enrichment do plików enriched
```bash
mkdir -p data/raw   # wrzuć tu surowe arm angle + movement + arsenal
python enrich_data.py --input data/raw --output data
```
Skrypt wypisze, ile wierszy dopasowano i jakie jest pokrycie kolumn, po czym zapisze `data/enriched_<rok>.csv`.
(Jupyter: `import data_utils as du; df, msgs = du.build_dataset([(nazwa, bytes), ...]); du.to_enriched(df)`)
Enrichment nie jest konieczny – dashboard robi to samo w locie.

## Krok 4 – start
```bash
streamlit run app.py
```
U góry dashboardu zobaczysz ✅/❌ dla `arm_angle, ivb, hb, velo, spin, pitch_type` oraz ostrzeżenia, gdy czegoś brakuje.

## Zakładki
1. **Liga wg arm slotu** – buckety (1–5°, domyślnie 2°), średnie ważone liczbą pitchy, liczba pitchy i miotaczy.
2. **Korelacje** – scatter arm slot vs IVB/HB/Velo/Spin, linia trendu (r, nachylenie), kolor wg typu/sezonu/ręki.
3. **Miotacz** – arm slot w czasie, metryki vs arm slot na tle ligi, porównanie z ligą w oknie ±N°.
4. **Zmiany YoY** – ranking największych zmian arm slotu + Δ IVB/HB/velo/spin, flaga "poprawa IVB".
5. **AI Insights** – wnioski dla ligi lub miotacza.
6. **Dane** – podgląd i eksport CSV.

Poziom analizy: *Wg typu pitcha* (miotacz×sezon×pitch) lub *Overall* (średnia ważona po repertuarze –
uwaga: IVB overall zależy od miksu pitchy, więc do wniosków o movemencie lepszy jest FF/SI).

## Logika AI Insights (reguły, bez LLM)
| Reguła | Statystyka | Przykładowy wynik |
|---|---|---|
| Nachylenie | regresja metryka ~ arm slot | "+1° arm slotu → +0,26 cala IVB (r = 0,79, silna, n = 360)" |
| Trzecie | średnia ważona górnej vs dolnej 1/3 arm slotu | "≥ 51,6° mają o 5,7 cala wyższe IVB niż ≤ 43,8°" |
| Najlepszy bucket | bucket 5° z max. średnim IVB (≥ 5 miotaczy) | "70–75°: 22,9 cala, +6,0 vs liga" |
| Rekomendacje YoY | Δ metryk w grupach zmian arm slotu (2–5°, >5°, w górę/w dół; n ≥ 8) | "Podniesienie o 2–5° korelowało z ΔIVB +1,2 (75% wzrosło)" |
| Miotacz vs liga | średnia ligowa dla arm slotu ± okno | "IVB o 1,5 cala poniżej średniej dla 46–50°" |
| Miotacz YoY | rzeczywista Δ vs oczekiwana z ligowej regresji | "ΔIVB +8,1 (oczekiwane ≈ +6,6)" |

Każdy wniosek podaje n i siłę korelacji; wszystkie kończą się zastrzeżeniem o korelacji vs przyczynowości.
Nowe reguły dopisujesz w `insights.py` (funkcje zwracają listę stringów).

## Uwagi
- Konwencja arm slotu (czy wyższy kąt = bardziej "over the top") wynika z plików Savant – kod jej nie zmienia.
- Znak HB zależy od ręki miotacza; przełącznik "|HB|" w sidebarze ułatwia porównania R vs L.
- Filtr "Min. pitchy" ucina małe próby; dla zakładki Miotacz i YoY działa na każdy sezon osobno.
- Dane o wynikach (whiff, RV/100, xwOBA) są opcjonalne – pojawią się w metrykach, jeśli je wgrasz.
