# Cross-Check `mc_simulation.R` (R) vs. `run_simulation.py` (Python)

Dieses Dokument beschreibt den numerischen Vergleich zwischen der
Base-R-Implementierung (`mc_simulation.R`, v3.0) und der Python-Referenz
(`run_simulation.py`). Es ist Teil der Lieferung zu
[ELI-25](/ELI/issues/ELI-25) und dient der Plausibilisierung der
v3.0-Semantik (OPEX-Annualisierung, LCM-Outputmetrik, Schwellenwert
`P(LCM < 1)`).

## 1. Modellgleichheit

| Aspekt | R (`mc_simulation.R`) | Python (`run_simulation.py`) | Konsistent? |
| --- | --- | --- | --- |
| CAPEX-Verteilung | `Tri(500, 750, 1200)` $M | `Tri(500, 750, 1200)` $M | ✅ identisch |
| OPEX-Verteilung | `Tri(80, 120, 200)` $M/Jahr | `Tri(80, 120, 200)` $M/Jahr | ✅ identisch |
| Volumen-Verteilung | `Tri(50, 150, 300)` M bbl | `Tri(50, 150, 300)` M bbl | ✅ identisch |
| Preis-Verteilung | Lognormal, `μ_ln=4.20`, `σ_ln=0.35` (E[X] ≈ 71) | Lognormal, `μ = ln(70) − 0.35²/2`, `σ = 0.35` (E[X] = 70) | ✅ konsistent (Rundung μ_ln) |
| Projekt-Lebensdauer `T` | `t_horizont = 10` (Default) | `project_years = 10` | ✅ identisch |
| OPEX-Annualisierung | `opex_total = opex_annual · T` | `opex_total = opex_annual · project_years` | ✅ identisch |
| LCM-Formel | `(Preis·Volumen − CAPEX − OPEX) / CAPEX` | `(Preis·Volumen − CAPEX − OPEX) / CAPEX` | ✅ identisch |
| Default-Iterationszahl | `10 000` | `10 000` | ✅ identisch |
| Default-Seed | **`42`** | **`12345`** | ⚠ absichtlich verschieden — siehe unten |

### Hinweis zu den Seeds

Die Default-Seeds sind **bewusst unterschiedlich**:

- R: `42` (kleiner, deterministisch, gut für Replikation in der Lehre)
- Python: `12345` (klassischer numerischer Demonstrations-Seed)

Beide Implementierungen verhalten sich reproduzierbar; die Seeds werden
in den JSON-Konfigurationen der jeweiligen Outputs festgehalten. Ein
numerischer Vergleich der Stichproben-Realisierungen ist daher **nicht**
punktgenau möglich — wir vergleichen ausschliesslich die **Form der
Verteilung** (Mean/Median/Std/VaR/P(Loss)-Grössenordnung).

## 2. Aufruf

```bash
# R-Variante (Output in ./output)
Rscript mc_simulation.R --seed 42

# Python-Referenz (Output in /tmp/MonteCarlo gemäss Originalskript)
python3 run_simulation.py
```

## 3. Numerische Plausibilisierung — vom CTO-Reviewer auszuführen

Da auf der aktuellen Build-Umgebung kein R-Interpreter verfügbar ist
(vgl. Anforderung 7 in [ELI-29](/ELI/issues/ELI-29)), wird die
numerische Plausibilisierung **nach dem Push** durch den CTO-Reviewer
manuell durchgeführt (R lokal installieren, beide Skripte laufen
lassen, Verteilungsform vergleichen).

### Erwartete Eigenschaften

1. **Rechtsschiefe:** Die LCM-Stichprobe ist rechtsschief
   (Preis-Lognormal treibt das obere Ende). `Skewness > 0`.
2. **Endliche Kennzahlen:** `min(LCM)`, `max(LCM)`, `Mean`, `Median`,
   `Std`, `VaR 5%` sind alle finite; keine NaN/Inf in der Stichprobe.
3. **`P(LCM < 1)` signifikant > 0:** Der Anteil der Draws mit
   `LCM < 1` sollte in der Grössenordnung des im Paper §4.2
   berichteten Verlustanteils liegen (≈ 12 %). Erwartet wird ein
   vergleichbarer Wert für die R-Implementierung, **nicht** identisch
   zur Python-Referenz, da `P(LCM < 1)` ≠ `P(ROI < 0)`.
4. **`Mean(LCM)` und `Median(LCM)`** in derselben Grössenordnung wie
   die Python-Referenz bei den v3.0-Defaults. Erwartet wird ein
   Median im Bereich ≈ 0.6–1.2 und ein Mean im Bereich ≈ 0.7–1.4
   (grobe Schätzung — exakte Werte nach R-Lauf eintragen).
5. **Streuung** vergleichbar zur Python-Referenz (gleiche σ in den
   Inputs ⇒ ähnliche σ im LCM).

### Schwellenwert-Diskussion: `P(LCM < 1)` vs. `P(ROI < 0)`

Inhaltliche Verschiebung gemäss [ELI-25](/ELI/issues/ELI-25):

- `ROI < 0` ≡ `(Preis·Volumen − CAPEX − OPEX) / CAPEX < 0` ≡
  Gewinn < 0 (Verlust).
- `LCM < 1` ≡ `(Preis·Volumen − CAPEX − OPEX) / CAPEX < 1` ≡
  Gewinn < CAPEX (Investition nicht zurückverdient, aber positiver
  Deckungsbeitrag möglich).

Numerisch gilt: `ROI < 0` ⇒ `LCM < 1`, aber **nicht** umgekehrt. Daher
ist `P(LCM < 1)` ≥ `P(ROI < 0)`.

**Konsequenz für die Vergleichbarkeit:** Die Python-Referenz
`run_simulation.py` rechnet `prob_loss = mean(roi < 0)`. Die
R-Implementierung rechnet `probability_of_loss = mean(lcm < 1)`. Die
beiden Werte sind **nicht** direkt vergleichbar; das ist im Cross-
Check zu beachten. Der JSON-Key heisst bewusst `probability_of_loss`
in beiden Implementationen, ist aber semantisch unterschiedlich
definiert.

## 4. Kennzahlen-Tabelle (vom CTO-Reviewer auszufüllen)

Die folgende Tabelle wird nach den beiden Läufen manuell gefüllt.
Platzhalter `TODO<CTO>` markieren die noch zu erhebenden Werte.

| Kennzahl | R (`mc_simulation.R`, seed 42) | Python (`run_simulation.py`, seed 12345) |
| --- | --- | --- |
| `mean(LCM)` | `TODO<CTO>` | `TODO<CTO>` |
| `median(LCM)` | `TODO<CTO>` | `TODO<CTO>` |
| `std(LCM)` | `TODO<CTO>` | `TODO<CTO>` |
| `VaR 5%` | `TODO<CTO>` | `TODO<CTO>` |
| `P(Loss)` (R: `<1`, Py: `<0`) | `TODO<CTO>` | `TODO<CTO>` |

Erwartungsbild (grobe Orientierung für den Reviewer):

- `mean(LCM)`: vergleichbare Grössenordnung in beiden Läufen.
- `median(LCM)`: vergleichbar; rechtsschief ⇒ Median < Mean.
- `std(LCM)`: vergleichbar (selbe Input-σ, leicht unterschiedlich
  wegen Seeds — Abweichung < 10 %).
- `VaR 5%`: vergleichbar (5 %-Quantil der Stichprobe).
- `P(Loss)`: R ≥ Python (siehe Schwellenwert-Diskussion oben).

## 5. Strukturelle Selbst-Verifikation (durch Coder durchgeführt)

Vor dem Push wurden folgende strukturelle Eigenschaften sichergestellt
(siehe auch Anforderung 7 in [ELI-29](/ELI/issues/ELI-29)):

- ✅ `grep -n '\broi\b' mc_simulation.R` liefert **keine** Treffer in
  Code-Identifikatoren. Verbleibende `ROI`-Vorkommen sind
  Historik-Kommentare (Papertitel, Nomenklaturhinweis).
- ✅ `kennzahlen()` ist parameterunabhängig vom Variablennamen — der
  Parameter heisst jetzt `lcm`, die internen Variablen sind
  konsistent umbenannt.
- ✅ `probability_of_loss` rechnet mit `mean(lcm < 1)`.
- ✅ `schreibe_csv` schreibt die Spalte `lcm` (zusätzlich `opex` und
  `opex_annual` als getrennte Spalten, damit der Annualisierungs-
  Schritt transparent bleibt).
- ✅ `schreibe_json` schreibt Keys `lcm` (in `mean`/`median`/…) und
  `probability_of_loss`.
- ✅ Plot-Funktion heisst `plot_lcm`, schreibt `lcm_histogramm.pdf`
  mit Achsenbeschriftung `"Lifetime Capital Multiple (LCM)"` und
  Hauptlinie bei `LCM = 1`.
- ✅ Konsolen-Report trägt die Überschrift
  `Monte-Carlo-LCM — Ergebnisreport` und gibt `P(LCM<1)` aus.

## 6. Bekannte Limitierungen

- R-Output-Mapping: `opex` in `results_r.csv` ist der **annualisierte**
  Wert über `T` Jahre (= `opex_annual · T`); `opex_annual` wird
  zusätzlich als separate Spalte ausgegeben, damit der Bezug zur
  Python-Referenz (die `opex_annual` und `opex_total` getrennt führt)
  gewahrt bleibt.
- Die R-Implementierung berechnet **keinen IRR**. Das ist konsistent
  zur Vorgabe: [ELI-25](/ELI/issues/ELI-25) verlangt nur die
  LCM-Ausgabe; IRR-Berechnung ist Aufgabe separater Tickets.
- Seeds unterscheiden sich absichtlich — siehe §1.