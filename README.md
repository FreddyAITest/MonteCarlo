# Monte-Carlo-IRR/LCM-Skript

Begleitender Code zum Working Paper *"Stochastische ROI-Bewertung mittels
Monte-Carlo-Simulation"* (Scho & Finkenzeller, 2026).

Das Skript `src/monte_carlo_roi.py` implementiert das im Paper beschriebene
Modell (CAPEX, OPEX, Fördervolumen, Ölpreis, **Projekt-Lebensdauer**,
**Decline-Curve**), führt eine Monte-Carlo-Simulation durch und erzeugt die
Abbildungen sowie die annualisierte **Internal Rate of Return (IRR)** als
primäre Kennzahl. Die kumulative **Lifetime Capital Multiple (LCM)** wird
als sekundäre Kennzahl weitergeführt. Eine gleichwertige, schlanke
Base-R-Implementierung befindet sich in `mc_simulation.R`
(RStudio-tauglich, keine Zusatzpakete).

## Voraussetzungen

- Python ≥ 3.10
- Standardweg: `numpy`, `scipy`, `matplotlib` (für Plots)
- Optional: `numpy_financial` (schnellerer vektorisierter IRR-Root-Finder;
  das Modul fällt sonst auf eine Bisektions-Implementierung in reiner
  Standardbibliothek zurück)
- Fallback: Falls weder `numpy` noch `matplotlib` installiert sind, läuft das
  Skript mit reiner Standardbibliothek — dann werden nur Kennzahlen in
  `results.json` geschrieben, **keine Plots und keine CSV**.

Empfohlene Installation:

```bash
python -m pip install numpy scipy matplotlib numpy_financial
```

## Aufruf

```bash
# Standardparameter (Offshore-Deepwater, 10 000 Iterationen, IRR + LCM)
python -m src.monte_carlo_roi

# Höhere Stichprobe, eigene Output-Pfade
python -m src.monte_carlo_roi --iterations 50000 --output-dir figures

# Eigene Inputvariablen (überschreiben die Standardwerte)
python -m src.monte_carlo_roi \
    --capex-low 8000 --capex-mode 10000 --capex-high 12000 \
    --opex-low 1500 --opex-mode 2200  --opex-high 3000 \
    --volume-low 500 --volume-mode 1000 --volume-high 2000 \
    --price-mean 70 --price-sigma 25

# Decline-Curve & Projekt-Lebensdauer konfigurieren
python -m src.monte_carlo_roi --decline-curve hyperbolic \
    --decline-a 0.05 --decline-b 0.5 --project-life-years 10
python -m src.monte_carlo_roi --decline-curve exponential --decline-a 0.10
python -m src.monte_carlo_roi --decline-curve flat

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
| `lcm_distribution.png` / `.pdf` | Histogramm + KDE der **LCM**-Stichprobe (sekundär) |
| `irr_distribution.png` / `.pdf` | Histogramm + KDE der **IRR**-Stichprobe (primär) |
| `inputs_distribution.png` / `.pdf` | Histogramme + theoretische Dichten der vier Inputvariablen |
| `sensitivity_tornado.png` / `.pdf` | Tornado-Diagramm der Sensitivitätsanalyse (LCM) |
| `sensitivity_tornado_irr.png` / `.pdf` | Tornado-Diagramm der Sensitivitätsanalyse (IRR) |
| `results.json` | Konfiguration, IRR- + LCM-Kennzahlen, Bootstrap-KIs, Sensitivitäts-Tabellen |
| `samples.csv` | Stichprobe (CAPEX, OPEX, Volumen, Preis, LCM, IRR) — überspringbar mit `--no-csv` |

## Modell (vier Inputs + Projekt-Lebensdauer + Decline-Curve)

| Variable | Verteilung | Default-Parameter | Quelle im Paper |
| --- | --- | --- | --- |
| CAPEX | Dreiecksverteilung | `Tri(8000, 10000, 12000)` M USD | §4.1 (Offshore-Deepwater) |
| OPEX  | Dreiecksverteilung | `Tri(1500, 2200, 3000)` M USD | §4.1 |
| Fördervolumen | Dreiecksverteilung | `Tri(500, 1000, 2000)` M Barrel | §4.1 |
| Ölpreis | Lognormal | `LogN(μ≈4.20, σ≈0.35)` — E[X] ≈ 70 $/Barrel, σ_X ≈ 25 $/Barrel | §3.2.2 |
| Projekt-Lebensdauer (`T`) | deterministisch | `10` Jahre | neu in [ELI-28](/ELI/issues/ELI-28) |
| Decline-Curve | `flat | exponential | hyperbolic` | `hyperbolic` (a=0.05, b=0.5) | neu in [ELI-28](/ELI/issues/ELI-28) |

### Cashflow-Generierung

Pro Monte-Carlo-Draw ``i`` wird eine Cashflow-Reihe der Länge ``T+1``
konstruiert:

    cashflow[0]   = -CAPEX_i
    cashflow[t]   = Preis_i · (Volumen_i / T) · decline_factor(t)
                    - OPEX_i / T    für t = 1, …, T

Der **Decline-Faktor** modelliert die jährliche Förderabnahme:

- `flat`:        `decline_factor(t) = 1`
- `exponential`: `decline_factor(t) = exp(-a · (t - 1))`
- `hyperbolic`:  `decline_factor(t) = 1 / (1 + b · a · (t - 1)) ** (1 / b)` (Arps)

Bei `t = 1` ist der Faktor definitionsgemäß `1.0` (höchste Förderung im
ersten Produktionsjahr). Die OPEX wird gleichmäßig auf die
Produktionsjahre verteilt (Standardkonvention für die jährliche
Betriebskosten-Last).

### Kennzahlen

- **IRR (primär)** — Internal Rate of Return, annualisiert. Löst
  `Σ cashflow[t] / (1 + r)^t = 0` für `r > -1`. Draws ohne
  Vorzeichenwechsel liefern `NaN`. Optional `numpy_financial.irr`,
  sonst Bisektion in reiner Standardbibliothek.
- **LCM (sekundär)** — Lifetime Capital Multiple (kumulativ):
  `LCM = (Preis · Volumen − CAPEX − OPEX) / CAPEX`. Wird für
  Vergleichbarkeit mit früheren Studien weitergeführt.

Math-Details und Konventionen sind in den Docstrings von
`src/monte_carlo_roi.py` (Klassen `SimulationConfig`, Funktionen
`generate_cashflows`, `irr`, `decline_factor`) dokumentiert.

### Base-R-Implementierung (`mc_simulation.R`)

Die schlanke Base-R-Variante (`mc_simulation.R`, v3.0-Semantik gemäss
[ELI-25](/ELI/issues/ELI-25)) verwendet bewusst die **kleinen** Pilotprojekt-
Defaults aus §4.1 des Papers (1:1 kompatibel zur `run_simulation.py`-
Referenz), nicht die Offshore-Deepwater-Skala, die für die Python-Referenz
in [ELI-26](/ELI/issues/ELI-26) festgelegt wurde:

| Variable | Verteilung | R-Default (v3.0) | Quelle |
| --- | --- | --- | --- |
| CAPEX | Dreiecksverteilung | `Tri(500, 750, 1200)` M USD | §4.1 |
| OPEX  | Dreiecksverteilung (jährlich) | `Tri(80, 120, 200)` M USD/Jahr | §4.1 |
| Fördervolumen | Dreiecksverteilung | `Tri(50, 150, 300)` M Barrel | §4.1 |
| Ölpreis | Lognormal | `LogN(μ≈4.20, σ≈0.35)` | §3.2.2 |
| Projekt-Lebensdauer `T` | deterministisch | `10` Jahre (CLI: `--t-horizont`) | v3.0 |

Der OPEX wird im R-Skript über `T` Jahre annualisiert:
`OPEX_total = OPEX_annual · T`. Die Ausgabe-Datei `results_r.csv`
enthält dementsprechend zwei OPEX-Spalten (`opex_annual` und `opex`),
sowie die Spalte `lcm` (statt `roi`). Die Verlustwahrscheinlichkeit
ist `P(LCM < 1)` — Schwelle gemäss [ELI-25](/ELI/issues/ELI-25), nicht
`P(ROI < 0)`. Siehe `cross_check_r_vs_py.md` für den Vergleich mit
der Python-Referenz.

## Tests

Der Plausibilitätstest läuft ohne wissenschaftliche Abhängigkeiten:

```bash
python tests/test_monte_carlo_roi.py
python tests/test_irr.py
```

`test_monte_carlo_roi.py` prüft:

- Konfigurationsvalidierung (Reihenfolge, Vorzeichen, neue
  Decline-Parameter)
- analytische Erwartungswerte und Varianzen der Dreiecks- und Lognormalverteilung
- Sampling-Grenzen (Triangular bleibt in [low, high], Lognormal > 0)
- strukturelle Eigenschaften der Stichprobenverteilung (Rechtsschiefe, finite Kennzahlen)
- optional Cross-Check mit numpy, falls verfügbar

`test_irr.py` prüft:

- Decline-Faktor-Werte gegen analytisch geschlossene Ausdrücke
- IRR-Root-Finder-Genauigkeit (2-, 5-, 10-Jahre-Referenzprojekte)
- IRR-Sonderfälle: kein Vorzeichenwechsel ⇒ `NaN`
- Konsistenz der IRR-Stichprobe zwischen numpy- und stdlib-Pfad
- Akzeptanzkriterium #6 aus [ELI-28](/ELI/issues/ELI-28) (Flat-Decline,
  IRR-Vergleich gegen den geometrischen Mittelwert — siehe Hinweis im
  Test-Quelltext bezüglich der systematischen Abweichung zwischen
  Annuitäts-IRR und `(1+LCM)^(1/T)-1`)

## Code-Struktur

```
src/
  __init__.py
  monte_carlo_roi.py   # Hauptmodul (CLI + API; IRR + LCM)
mc_simulation.R        # Base-R-Implementierung (RStudio-tauglich)
tests/
  test_monte_carlo_roi.py  # Stdlib-Smoketest (Inputs + LCM)
  test_irr.py              # Stdlib-Smoketest (Cashflows + IRR)
output/                 # Wird beim Lauf angelegt
```
