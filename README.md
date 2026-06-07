# Monte-Carlo-ROI-Skript

Begleitender Code zum Working Paper *"Stochastische ROI-Bewertung mittels
Monte-Carlo-Simulation"* (Scho & Finkenzeller, 2026).

Das Skript `src/monte_carlo_roi.py` implementiert das im Paper beschriebene
Vier-Variablen-Modell (CAPEX, OPEX, Fördervolumen, Ölpreis), führt eine
Monte-Carlo-Simulation durch und erzeugt die drei im Paper referenzierten
Abbildungen. Eine gleichwertige, schlanke Base-R-Implementierung befindet
sich in `mc_simulation.R` (RStudio-tauglich, keine Zusatzpakete).

## Voraussetzungen

- Python ≥ 3.10
- Standardweg: `numpy`, `scipy`, `matplotlib` (für Plots)
- Fallback: Falls weder `numpy` noch `matplotlib` installiert sind, läuft das
  Skript mit reiner Standardbibliothek — dann werden nur Kennzahlen in
  `results.json` geschrieben, **keine Plots und keine CSV**.

Empfohlene Installation:

```bash
python -m pip install numpy scipy matplotlib
```

## Aufruf

```bash
# Standardparameter (siehe Abschnitt 4 des Papers), 10 000 Iterationen
python -m src.monte_carlo_roi

# Höhere Stichprobe, eigene Output-Pfade
python -m src.monte_carlo_roi --iterations 50000 --output-dir figures

# Eigene Inputvariablen (überschreiben die Standardwerte)
python -m src.monte_carlo_roi \
    --capex-low 400 --capex-mode 700 --capex-high 1100 \
    --opex-low 70   --opex-mode 110  --opex-high 180  \
    --volume-low 40 --volume-mode 140 --volume-high 280 \
    --price-mean 65 --price-sigma 20

# Nur Kennzahlen, keine Stichproben-CSV
python -m src.monte_carlo_roi --no-csv

# Base-R-Implementierung (RStudio / Rscript, keine Zusatzpakete)
Rscript mc_simulation.R                       # Standardparameter
Rscript mc_simulation.R --iterations 50000    # größere Stichprobe
Rscript mc_simulation.R --kein-plot           # ohne Histogramm
```

## Ausgaben

Im Output-Verzeichnis (Standard `output/`):

| Datei | Inhalt |
| --- | --- |
| `roi_distribution.png` / `.pdf` | Histogramm + KDE der ROI-Stichprobe mit Mean, Median, VaR 5 % und P(ROI<0) |
| `inputs_distribution.png` / `.pdf` | Histogramme + theoretische Dichten der vier Inputvariablen |
| `sensitivity_tornado.png` / `.pdf` | Tornado-Diagramm der Pearson-basierten Sensitivitätsanalyse |
| `results.json` | Konfiguration, Kennzahlen, Bootstrap-Konfidenzintervall, Sensitivitäts-Tabelle |
| `samples.csv` | Stichprobe (CAPEX, OPEX, Volumen, Preis, ROI) — überspringbar mit `--no-csv` |

## Inputvariablen

| Variable | Verteilung | Default-Parameter | Quelle im Paper |
| --- | --- | --- | --- |
| CAPEX | Dreiecksverteilung | `Tri(500M, 750M, 1200M)` | §4.1 |
| OPEX  | Dreiecksverteilung | `Tri(80M, 120M, 200M)` | §4.1 |
| Fördervolumen | Dreiecksverteilung | `Tri(50M, 150M, 300M)` Barrel | §4.1 |
| Ölpreis | Lognormal | `LogN(μ≈4.20, σ≈0.35)` — E[X] ≈ 70 $/Barrel, σ_X ≈ 25 $/Barrel | §3.2.2 |

Die ROI-Formel folgt der im Paper in §3.3 angegebenen Definition:

    ROI = (Preis · Volumen − CAPEX − OPEX) / CAPEX

## Tests

Der Plausibilitätstest läuft ohne wissenschaftliche Abhängigkeiten:

```bash
python tests/test_monte_carlo_roi.py
```

Er prüft:

- Konfigurationsvalidierung (Reihenfolge, Vorzeichen)
- analytische Erwartungswerte und Varianzen der Dreiecks- und Lognormalverteilung
- Sampling-Grenzen (Triangular bleibt in [low, high], Lognormal > 0)
- strukturelle Eigenschaften der Stichprobenverteilung (Rechtsschiefe, finite Kennzahlen)
- optional Cross-Check mit numpy, falls verfügbar

## Code-Struktur

```
src/
  __init__.py
  monte_carlo_roi.py   # Hauptmodul (CLI + API)
mc_simulation.R        # Base-R-Implementierung (RStudio-tauglich)
tests/
  test_monte_carlo_roi.py  # Stdlib-Smoketest
output/                 # Wird beim Lauf angelegt
```
