"""
Monte-Carlo-Simulation des Internal Rate of Return (IRR, primär) und
Lifetime Capital Multiple (LCM, sekundär) für ölindustrielle
Investitionsprojekte.

Seit [ELI-28](/ELI/issues/ELI-28) ist das Modell um eine Projekt-Lebensdauer
(``project_life_years``) und eine Decline-Curve (``decline_curve``) erweitert.
Pro Monte-Carlo-Draw wird eine jahresweise Cashflow-Reihe erzeugt und
daraus der annualisierte IRR berechnet. Der LCM (kumulativ) bleibt als
sekundäre Kennzahl erhalten — Rückwärtskompatibilität siehe
``compute_roi``, ``summarize_roi``, ``plot_roi_distribution``, ``result.roi``.

Hinweis zur Nomenklatur: Die in [ELI-19](/ELI/issues/ELI-19) analysierte
Stichprobenkennzahl wurde in [ELI-24](/ELI/issues/ELI-24) von "ROI" auf
``lifetime_capital_multiple`` (LCM) umgestellt (kumulatives Multiple,
kein annualisierter Return). Mit [ELI-28](/ELI/issues/ELI-28) ist IRR
die primäre Kennzahl; LCM bleibt als sekundäre Vergleichsgröße.

Dieses Skript implementiert das im Working Paper "Stochastische
ROI-Bewertung mittels Monte-Carlo-Simulation" beschriebene Modell und
erzeugt die im Paper referenzierten Abbildungen
(Inputverteilungen, IRR-Histogramm, LCM-Histogramm, Tornado-Diagramme
der Sensitivitätsanalyse).

Aufruf:

    python -m src.monte_carlo_roi                       # Offshore-Deepwater-Defaults
    python -m src.monte_carlo_roi --iterations 50000    # mehr Iterationen
    python -m src.monte_carlo_roi --seed 7 --output-dir figures
    python -m src.monte_carlo_roi --decline-curve flat  # ohne Decline
    python -m src.monte_carlo_roi --decline-curve exponential --decline-a 0.10

Standard-Defaults (Offshore-Deepwater, siehe [ELI-26](/ELI/issues/ELI-26)):

    CAPEX  Tri(8000/10000/12000) M USD
    OPEX   Tri(1500/2200/3000)   M USD
    Vol    Tri(500/1000/2000)    M bbl
    Preis  LogN(mean=70, sigma=25) USD/bbl

Cashflow-Modell (siehe ``generate_cashflows``):

    cashflow[0]   = -CAPEX
    cashflow[t]   = Preis · (Volumen / T) · decline_factor(t)
                    - OPEX / T      für t = 1, …, T

Decline-Faktoren (siehe ``decline_factor``):

    flat:        1
    exponential: exp(-a · (t - 1))
    hyperbolic:  1 / (1 + b · a · (t - 1)) ** (1 / b)

Ausgaben (im Output-Verzeichnis, default ``./output``):

    irr_distribution.pdf / .png        Histogramm + Kennzahlen der IRR-Stichprobe
    lcm_distribution.pdf / .png        Histogramm + Kennzahlen der LCM-Stichprobe
    sensitivity_tornado_irr.pdf / .png Tornado-Diagramm der IRR-Sensitivität
    sensitivity_tornado.pdf / .png     Tornado-Diagramm der LCM-Sensitivität
    inputs_distribution.pdf / .png     Verteilungen der vier Inputvariablen
    results.json                       IRR + LCM Kennzahlen, Bootstrap-KIs,
                                       Sensitivitäts-Tabellen
    samples.csv                        Stichprobe (CAPEX, OPEX, Volumen, Preis,
                                       LCM, IRR) — überspringbar mit ``--no-csv``

Abhängigkeiten: numpy, matplotlib, scipy (für die Dichtefunktion der
Dreiecksverteilung, die in den Input-Plots als Referenzlinie dient).
Optional: ``numpy_financial`` (schnellerer vektorisierter IRR-Root-Finder;
fällt sonst auf Bisektion in reiner Standardbibliothek zurück).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

# Heavy scientific stack wird lazy geladen, damit der Plausibilitätstest
# auch in schlanken Umgebungen ohne numpy/matplotlib die Konfiguration
# importieren und validieren kann.
np: Any = None
plt: Any = None
_scipy_stats: Any = None
_HAS_SCIPY = False


def _ensure_numpy() -> None:
    """Lazy-Import von numpy; bricht mit klarer Meldung ab, falls fehlend."""
    global np
    if np is None:
        import numpy as _np  # noqa: WPS433
        np = _np


def _ensure_matplotlib() -> None:
    """Lazy-Import von matplotlib; setzt das Backend auf 'Agg'."""
    global plt
    if plt is None:
        import matplotlib as _mpl  # noqa: WPS433
        _mpl.use("Agg")
        import matplotlib.pyplot as _plt  # noqa: WPS433
        plt = _plt


def _ensure_scipy() -> bool:
    """Lazy-Import von scipy.stats; gibt zurück, ob scipy verfügbar ist."""
    global _scipy_stats, _HAS_SCIPY
    if _HAS_SCIPY:
        return True
    try:
        from scipy import stats as _stats  # noqa: WPS433
    except ImportError:
        _HAS_SCIPY = False
        return False
    _scipy_stats = _stats
    _HAS_SCIPY = True
    return True


def _has_matplotlib() -> bool:
    """True, wenn matplotlib importierbar ist (kein Import als Seiteneffekt)."""
    if plt is not None:
        return True
    try:
        import importlib.util  # noqa: WPS433
    except ImportError:
        return False
    return importlib.util.find_spec("matplotlib") is not None


def _has_numpy() -> bool:
    """True, wenn numpy importierbar ist (kein Import als Seiteneffekt)."""
    if np is not None:
        return True
    try:
        import importlib.util  # noqa: WPS433
    except ImportError:
        return False
    return importlib.util.find_spec("numpy") is not None


# ---------------------------------------------------------------------------
# Eingabeparameter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriangularParams:
    """Parameter einer Dreiecksverteilung (Min, Modus, Max)."""

    low: float
    mode: float
    high: float
    name: str

    def __post_init__(self) -> None:
        if not (self.low <= self.mode <= self.high):
            raise ValueError(
                f"{self.name}: erwartet low ≤ mode ≤ high, "
                f"erhielt ({self.low}, {self.mode}, {self.high})"
            )
        if self.low == self.high:
            raise ValueError(f"{self.name}: low und high dürfen nicht identisch sein")

    @property
    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3.0

    @property
    def variance(self) -> float:
        return (
            self.low**2 + self.mode**2 + self.high**2
            - self.low * self.mode - self.low * self.high - self.mode * self.high
        ) / 18.0


@dataclass(frozen=True)
class LogNormalParams:
    """Parameter einer Lognormalverteilung in Dollar/Barrel."""

    mean: float
    sigma: float
    name: str

    def __post_init__(self) -> None:
        if self.mean <= 0:
            raise ValueError(f"{self.name}: mean muss > 0 sein")
        if self.sigma <= 0:
            raise ValueError(f"{self.name}: sigma muss > 0 sein")


@dataclass(frozen=True)
class SimulationConfig:
    """Vollständige Konfiguration der Monte-Carlo-Simulation.

    Felder für die jahresweise Cashflow-Generierung (siehe
    [ELI-20](/ELI/issues/ELI-20)):

    - ``project_life_years``: Anzahl Produktionsjahre ``T`` (Cashflow
      bei ``t=0`` ist die Initialinvestition ``-CAPEX``; Cashflows für
      ``t = 1..T`` sind die annualisierten Umsätze minus annualisierter
      OPEX).
    - ``decline_curve``: Form der Produktions-Abnahme
      (``'flat' | 'exponential' | 'hyperbolic'``). Der
      Decline-Faktor bei ``t=1`` ist definitionsgemäß ``1.0``;
      die jährliche Förderung ist ``(Volumen / T) * decline_factor(t)``.
    - ``decline_a``: Annualisierte Anfangs-Abnahmerate ``a``. Nur für
      exponentiell / hyperbolisch verwendet. Beispiel-Defaults aus der
      Arps-DECLINE-Literatur (siehe Spec von [ELI-28](/ELI/issues/ELI-28)).
    - ``decline_b``: Arps-``b``-Faktor, nur für hyperbolisch verwendet
      (``q(t) = q_i / (1 + b * a * t) ** (1 / b)`` mit ``t`` in Jahren
      ab Erstförderung).
    """

    capex: TriangularParams
    opex: TriangularParams
    volume: TriangularParams
    price: LogNormalParams
    iterations: int = 10_000
    seed: int = 42
    bootstrap_samples: int = 1_000
    bootstrap_confidence: float = 0.95
    project_life_years: int = 10
    decline_curve: str = "hyperbolic"
    decline_a: float = 0.05
    decline_b: float = 0.5

    def __post_init__(self) -> None:
        # ``frozen=True`` ⇒ kein normales Attribut-Set; über ``object.__setattr__``
        # validieren wir readonly-Felder.
        if self.project_life_years < 1:
            raise ValueError(
                f"project_life_years muss >= 1 sein, ist {self.project_life_years}"
            )
        allowed = {"flat", "exponential", "hyperbolic"}
        if self.decline_curve not in allowed:
            raise ValueError(
                f"decline_curve muss eins von {sorted(allowed)} sein, "
                f"ist {self.decline_curve!r}"
            )
        if self.decline_a < 0:
            raise ValueError(f"decline_a muss >= 0 sein, ist {self.decline_a}")
        if self.decline_b <= 0 or self.decline_b > 1.0:
            raise ValueError(
                f"decline_b muss in (0, 1] liegen (Arps-Konvention), "
                f"ist {self.decline_b}"
            )

    def expected_metrics(self) -> dict[str, float]:
        """Analytische Erwartungswerte & Varianzen der Inputs (für Plausibilität)."""
        # Für die Lognormal-Verteilung interpretieren wir self.price.sigma als
        # Standardabweichung im Originalmaßstab ($/Barrel) — konsistent mit
        # der im Paper verwendeten Schreibweise "σ ≈ 25 $/Barrel". Die
        # Varianz der Lognormalvariablen wird über die zugehörigen
        # Log-Raum-Parameter (mu, sigma_ln) ausgedrückt, die den gleichen
        # Erwartungswert und die gleiche Streuung liefern.
        variance_dollar = self.price.sigma ** 2
        sigma_ln = math.sqrt(math.log(1.0 + variance_dollar
                                      / (self.price.mean ** 2)))
        return {
            "E_CAPEX": self.capex.mean,
            "E_OPEX": self.opex.mean,
            "E_VOLUME": self.volume.mean,
            "E_PRICE": self.price.mean,
            "Var_CAPEX": self.capex.variance,
            "Var_OPEX": self.opex.variance,
            "Var_VOLUME": self.volume.variance,
            "Var_PRICE": (math.exp(sigma_ln ** 2)
                          * (math.exp(sigma_ln ** 2) - 1.0)
                          * self.price.mean ** 2),
        }

    def cashflow_config(self) -> dict[str, Any]:
        """Cashflow-Konfiguration (für Report / Tests)."""
        return {
            "project_life_years": self.project_life_years,
            "decline_curve": self.decline_curve,
            "decline_a": self.decline_a,
            "decline_b": self.decline_b,
        }


def default_config() -> SimulationConfig:
    """Standardkonfiguration für Offshore-Deepwater-Projekte (Stand [ELI-26](/ELI/issues/ELI-26)).

    Die Defaults wurden im Rahmen der [ELI-19](/ELI/issues/ELI-19)-Investigation
    und der [ELI-26](/ELI/issues/ELI-26)-Board-Abstimmung von einem
    "kleinen" Pilotprojekt (CAPEX 0.5–1.2 Mrd USD) auf ein realistisches
    Offshore-Deepwater-Investitionsprojekt hochskaliert. Quellen:
    IEA *World Energy Investment 2024* (Deepwater-IRR 10–20 % p.a.),
    Wood Mackenzie / Rystad (CAPEX 8–12 Mrd USD pro Deepwater-Projekt).

    - CAPEX Tri(8000/10000/12000) M USD  — Deepwater-FPSO / Subsea-Bereich
    - OPEX  Tri(1500/2200/3000)   M USD  — inkl. Wartung, Logistik, Tanker
    - Vol   Tri(500/1000/2000)    M bbl  — recoverable reserves
    - Preis LogN(mean=70, sigma=25) USD/bbl — wie zuvor
    """
    return SimulationConfig(
        capex=TriangularParams(low=8_000.0, mode=10_000.0, high=12_000.0, name="CAPEX"),
        opex=TriangularParams(low=1_500.0, mode=2_200.0, high=3_000.0, name="OPEX"),
        volume=TriangularParams(low=500.0, mode=1_000.0, high=2_000.0, name="Volume"),
        price=LogNormalParams(mean=70.0, sigma=25.0, name="Price"),
        iterations=10_000,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_inputs(
    config: SimulationConfig, rng: "np.random.Generator"
) -> dict[str, "np.ndarray"]:
    """Zieht N Stichproben aus den vier Verteilungen."""
    _ensure_numpy()
    n = config.iterations
    return {
        "capex": rng.triangular(
            left=config.capex.low,
            mode=config.capex.mode,
            right=config.capex.high,
            size=n,
        ),
        "opex": rng.triangular(
            left=config.opex.low,
            mode=config.opex.mode,
            right=config.opex.high,
            size=n,
        ),
        "volume": rng.triangular(
            left=config.volume.low,
            mode=config.volume.mode,
            right=config.volume.high,
            size=n,
        ),
        "price": rng.lognormal(
            mean=math.log(config.price.mean**2
                         / math.sqrt(config.price.sigma**2 + config.price.mean**2)),
            sigma=math.sqrt(math.log(1.0 + (config.price.sigma / config.price.mean) ** 2)),
            size=n,
        ),
    }


def compute_lcm(samples: dict[str, "np.ndarray"]) -> "np.ndarray":
    """Berechnet den Lifetime Capital Multiple nach der im Paper definierten Formel.

    LCM = (Preis · Volumen − CAPEX − OPEX) / CAPEX

    Die Kennzahl wird im Working Paper informell als "ROI" bezeichnet, ist
    aber *kein* annualisierter Return, sondern ein kumulatives
    Lebensdauer-Kapitalvielfaches (siehe [ELI-24](/ELI/issues/ELI-24)). Die
    Preise sind in $/Barrel, das Volumen in M Barrel, CAPEX/OPEX in $M —
    damit ist (Preis · Volumen) bereits in $M und alle Terme sind konsistent.
    """
    _ensure_numpy()
    revenue = samples["price"] * samples["volume"]
    profit = revenue - samples["capex"] - samples["opex"]
    return profit / samples["capex"]


def compute_roi(samples: dict[str, "np.ndarray"]) -> "np.ndarray":
    """Deprecated alias für :func:`compute_lcm`.

    Wird für eine Release-Phase als Rückwärtskompatibilitäts-Shim
    vorgehalten, damit externe Konsumenten, die noch ``compute_roi``
    importieren, weiterhin funktionieren. Neue Aufrufe sollten
    ``compute_lcm`` verwenden.
    """
    return compute_lcm(samples)


# ---------------------------------------------------------------------------
# Cashflow-Generator (Projekt-Lebensdauer + Decline-Curve)
# ---------------------------------------------------------------------------
#
# Konvention (siehe Spec von [ELI-28](/ELI/issues/ELI-28) und CTO-Delegation):
#   * ``t = 0`` ist der Investitionszeitpunkt — Cashflow = ``-CAPEX``.
#   * ``t = 1..T`` (``T = project_life_years``) sind die Produktionsjahre.
#     Die jährliche Förderung folgt dem Decline-Faktor und summiert sich
#     über die Projektlebensdauer definitionsgemäß **nicht** zwingend auf
#     ``Volumen`` auf — ``Volumen`` ist eine stochastische Obergrenze für
#     die kumulierte Förderung (im Sinne von "recoverable reserves"), wird
#     aber nicht normalisiert. Der Operator kann das durch Anpassen der
#     Default-Decline-Parameter (``decline_a``) steuern.
#   * ``decline_factor(1) = 1`` in allen Kurvenformen — das erste volle
#     Produktionsjahr trägt den höchsten Anteil.
#   * OPEX wird **gleichmäßig** über die Produktionsjahre verteilt (gemäß
#     CTO-Delegation: "OPEX distributed evenly across years"). Das
#     modelliert den laufenden Betrieb ohne Reservoirspezifische
#     Kostenverläufe.
#
# Form (pro Draw):
#     cashflow[0] = -CAPEX
#     cashflow[t] = Preis · (Volumen / T) · decline_factor(t)
#                   - OPEX / T       für t = 1..T
#
# Decline-Faktoren:
#     flat:        decline_factor(t) = 1
#     exponential: decline_factor(t) = exp(-a · (t - 1))
#     hyperbolic:  decline_factor(t) = 1 / (1 + b · a · (t - 1)) ** (1 / b)
# ---------------------------------------------------------------------------


def decline_factor(curve: str, t: int, a: float, b: float) -> float:
    """Liefert den Decline-Faktor für Jahr ``t`` (1-indiziert)."""
    if t < 1:
        raise ValueError(f"decline_factor: t muss >= 1 sein, ist {t}")
    if curve == "flat":
        return 1.0
    if curve == "exponential":
        return math.exp(-a * (t - 1))
    if curve == "hyperbolic":
        # Arps-hyperbolisch: q(t) = q_i / (1 + b·a·t)^(1/b), wobei ``t``
        # in Jahren ab Erstförderung gezählt wird (bei uns also t-1).
        if b <= 0:
            raise ValueError(f"decline_b muss > 0 sein, ist {b}")
        return 1.0 / (1.0 + b * a * (t - 1)) ** (1.0 / b)
    raise ValueError(f"Unbekannte decline_curve: {curve!r}")


def generate_cashflows(
    capex: "np.ndarray",
    opex: "np.ndarray",
    volume: "np.ndarray",
    price: "np.ndarray",
    project_life_years: int,
    decline_curve: str,
    decline_a: float,
    decline_b: float,
) -> "np.ndarray":
    """Baut die Cashflow-Matrix ``(N, T+1)`` für jeden Monte-Carlo-Draw.

    Die Cashflows sind in derselben Einheit wie CAPEX/OPEX ($M); Umsatz
    ist ``Preis · Volumen`` ($/bbl · M bbl = $M). Die zeitliche Verteilung
    folgt der in der Modul-Docstring beschriebenen Konvention.
    """
    _ensure_numpy()
    if project_life_years < 1:
        raise ValueError(
            f"project_life_years muss >= 1 sein, ist {project_life_years}"
        )
    n = capex.shape[0]
    T = int(project_life_years)
    cashflows = np.empty((n, T + 1), dtype=float)
    cashflows[:, 0] = -capex
    annual_volume = volume / float(T)
    annual_opex = opex / float(T)
    for t in range(1, T + 1):
        df = decline_factor(decline_curve, t, decline_a, decline_b)
        cashflows[:, t] = price * annual_volume * df - annual_opex
    return cashflows


# ---------------------------------------------------------------------------
# IRR (Internal Rate of Return)
# ---------------------------------------------------------------------------
#
# Definition: ``r`` löst  ``Σ cashflow[t] / (1 + r)^t = 0``  über
# ``t = 0..T`` mit ``r > -1``.
#
# Implementierungs-Strategie:
#   1. Versuche ``numpy_financial.irr`` (vektorisiert, schnell, exakt).
#   2. Fallback: Bracket-and-Bisection (``[-0.99, +10]``), optional mit
#      Newton-Polish. Robust gegen pathologische Draws (kein Vorzeichen-
#      wechsel ⇒ ``nan``).
# ---------------------------------------------------------------------------


def _npv(cashflows: "np.ndarray", rate: float) -> float:
    """NPV für eine 1-D-Cashflowreihe."""
    return float(np.sum(cashflows / (1.0 + rate) ** np.arange(cashflows.size)))


def _irr_single_bisect(cashflows: "np.ndarray",
                       lo: float = -0.99,
                       hi: float = 10.0,
                       tol: float = 1e-9,
                       max_iter: int = 256) -> float:
    """Bracket-Bisektion + Newton-Polish für eine einzelne Cashflowreihe."""
    f_lo = _npv(cashflows, lo)
    f_hi = _npv(cashflows, hi)
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
        return float("nan")
    if f_lo * f_hi > 0:
        # kein Vorzeichenwechsel im Suchintervall — versuche ``hi`` zu
        # vergrößern, danach geben wir auf.
        for hi_try in (50.0, 200.0, 1000.0):
            f_hi = _npv(cashflows, hi_try)
            if math.isfinite(f_hi) and f_lo * f_hi <= 0:
                hi = hi_try
                break
        else:
            return float("nan")
    a, b = lo, hi
    fa = _npv(cashflows, a)
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fmid = _npv(cashflows, mid)
        if abs(fmid) < tol or (b - a) < tol:
            a, b, fa = mid, mid, fmid
            break
        if fa * fmid <= 0:
            b, fb = mid, fmid
        else:
            a, fa = mid, fmid
    r = 0.5 * (a + b)
    # Newton-Polish (1–2 Schritte) — verwendet die analytische Ableitung.
    for _ in range(4):
        f = _npv(cashflows, r)
        if abs(f) < tol:
            return r
        # df/dr = Σ -t · cashflow[t] / (1+r)^(t+1)
        t = np.arange(cashflows.size, dtype=float)
        denom = (1.0 + r) ** (t + 1.0)
        df = float(np.sum(-t * cashflows / denom))
        if df == 0.0 or not math.isfinite(df):
            break
        step = f / df
        # Dämpfe Sprünge über die Intervallgrenzen
        if r - step <= lo:
            step = max(0.5 * (r - lo), 0.0)
        elif r - step >= hi:
            step = min(0.5 * (hi - r), 0.0)
        r = r - step
        if r <= lo or r >= hi:
            break
    return r


def _has_numpy_financial() -> bool:
    """True, wenn ``numpy_financial`` importierbar ist."""
    try:
        import importlib.util  # noqa: WPS433
    except ImportError:
        return False
    return importlib.util.find_spec("numpy_financial") is not None


def _irr_vectorized_npf(cashflows: "np.ndarray") -> "np.ndarray":
    """Vektorisierter IRR über ``numpy_financial.irr`` (pro Zeile)."""
    import numpy_financial as npf  # noqa: WPS433
    n = cashflows.shape[0]
    out = np.full(n, float("nan"), dtype=float)
    for i in range(n):
        try:
            r = npf.irr(cashflows[i])
        except Exception:
            r = float("nan")
        if r is None or not math.isfinite(float(r)):
            out[i] = float("nan")
        else:
            out[i] = float(r)
    return out


def irr(cashflows: "np.ndarray") -> "np.ndarray":
    """Annualisierter Internal Rate of Return für jedes Draw.

    Eingabe ist eine 2-D-Matrix ``(N, T+1)`` (Cashflowreihen zeilenweise).
    Pro Draw ``i`` wird ``r_i`` so bestimmt, dass
    ``Σ cashflows[i, t] / (1 + r_i)^t = 0``.

    Draws ohne Vorzeichenwechsel oder mit nicht-konvergenter Suche
    liefern ``float('nan')``.

    Bevorzugt ``numpy_financial.irr``, fällt sonst auf eine
    Bisektions-Implementierung in reiner ``numpy``-Mathematik zurück.
    Beide Pfade sind ohne externe wissenschaftliche Bibliotheken
    lauffähig (scipy wird nicht benötigt).
    """
    _ensure_numpy()
    cashflows = np.asarray(cashflows, dtype=float)
    if cashflows.ndim == 1:
        cashflows = cashflows.reshape(1, -1)
    if cashflows.ndim != 2:
        raise ValueError(
            f"cashflows muss 1-D oder 2-D sein, ist {cashflows.ndim}-D"
        )
    if _has_numpy_financial():
        try:
            return _irr_vectorized_npf(cashflows)
        except Exception:
            # Fallback unten verwenden, falls numpy_financial scheitert
            pass

    n = cashflows.shape[0]
    out = np.empty(n, dtype=float)
    for i in range(n):
        out[i] = _irr_single_bisect(cashflows[i])
    return out


# ---------------------------------------------------------------------------
# Stdlib-Fallback (für Umgebungen ohne numpy)
# ---------------------------------------------------------------------------


def _triangular_stlib(low: float, mode: float, high: float, rng) -> float:
    u = rng.random()
    if u < (mode - low) / (high - low):
        return low + math.sqrt(u * (high - low) * (mode - low))
    return high - math.sqrt((1.0 - u) * (high - low) * (high - mode))


def _lognormal_stlib(mean: float, sigma: float, rng) -> float:
    variance = sigma * sigma
    mu = math.log(mean * mean / math.sqrt(variance + mean * mean))
    sigma_ln = math.sqrt(math.log(1.0 + variance / (mean * mean)))
    return math.exp(rng.gauss(mu, sigma_ln))


def _decline_factor_stlib(curve: str, t: int, a: float, b: float) -> float:
    if t < 1:
        raise ValueError(f"decline_factor: t muss >= 1 sein, ist {t}")
    if curve == "flat":
        return 1.0
    if curve == "exponential":
        return math.exp(-a * (t - 1))
    if curve == "hyperbolic":
        if b <= 0:
            raise ValueError(f"decline_b muss > 0 sein, ist {b}")
        return 1.0 / (1.0 + b * a * (t - 1)) ** (1.0 / b)
    raise ValueError(f"Unbekannte decline_curve: {curve!r}")


def _npv_stlib(cashflows, rate: float) -> float:
    return sum(c / (1.0 + rate) ** t for t, c in enumerate(cashflows))


def _irr_single_stlib(cashflows,
                      lo: float = -0.99,
                      hi: float = 10.0,
                      tol: float = 1e-9,
                      max_iter: int = 256) -> float:
    """Bracket-Bisektion + Newton-Polish — reine Standardbibliothek."""
    f_lo = _npv_stlib(cashflows, lo)
    f_hi = _npv_stlib(cashflows, hi)
    if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
        return float("nan")
    if f_lo * f_hi > 0:
        for hi_try in (50.0, 200.0, 1000.0):
            f_hi = _npv_stlib(cashflows, hi_try)
            if math.isfinite(f_hi) and f_lo * f_hi <= 0:
                hi = hi_try
                break
        else:
            return float("nan")
    a, b = lo, hi
    fa = f_lo
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fmid = _npv_stlib(cashflows, mid)
        if abs(fmid) < tol or (b - a) < tol:
            return mid
        if fa * fmid <= 0:
            b, fb = mid, fmid
        else:
            a, fa = mid, fmid
    r = 0.5 * (a + b)
    # Newton-Polish
    for _ in range(4):
        f = _npv_stlib(cashflows, r)
        if abs(f) < tol:
            return r
        df = sum(
            -t * c / (1.0 + r) ** (t + 1)
            for t, c in enumerate(cashflows)
        )
        if df == 0.0 or not math.isfinite(df):
            break
        step = f / df
        if r - step <= lo:
            step = max(0.5 * (r - lo), 0.0)
        elif r - step >= hi:
            step = min(0.5 * (hi - r), 0.0)
        r = r - step
        if r <= lo or r >= hi:
            break
    return r


def _percentile(sorted_values: list[float], q: float) -> float:
    """Lineares Perzentil (numpy-kompatibel) auf einer sortierten Liste."""
    if not sorted_values:
        return float("nan")
    pos = (len(sorted_values) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo))


def _run_stlib(
    config: SimulationConfig, output_dir: Path, write_csv: bool
) -> SimulationResult:
    """Langsamer Stdlib-Pfad — identische Mathematik, aber ohne Plots/CSV."""
    import random as _random
    import statistics as _stats

    rng = _random.Random(config.seed)
    samples: dict[str, list[float]] = {
        "capex": [], "opex": [], "volume": [], "price": [],
    }
    for _ in range(config.iterations):
        samples["capex"].append(_triangular_stlib(
            config.capex.low, config.capex.mode, config.capex.high, rng))
        samples["opex"].append(_triangular_stlib(
            config.opex.low, config.opex.mode, config.opex.high, rng))
        samples["volume"].append(_triangular_stlib(
            config.volume.low, config.volume.mode, config.volume.high, rng))
        samples["price"].append(_lognormal_stlib(config.price.mean,
                                                config.price.sigma, rng))

    T = int(config.project_life_years)
    annual_volume_template = 1.0 / float(T)
    annual_opex_template = 1.0 / float(T)
    precomputed_decline = [
        _decline_factor_stlib(config.decline_curve, t,
                              config.decline_a, config.decline_b)
        for t in range(1, T + 1)
    ]

    # Cashflow-Matrix (liste von Listen) + LCM + IRR
    cashflow_rows: list[list[float]] = []
    lcms: list[float] = []
    irrs: list[float] = []
    for c, o, v, p in zip(samples["capex"], samples["opex"],
                          samples["volume"], samples["price"]):
        lcms.append((p * v - c - o) / c)
        row = [-c]
        for t_idx, df in enumerate(precomputed_decline, start=1):
            row.append(p * (v * annual_volume_template) * df
                       - o * annual_opex_template)
        cashflow_rows.append(row)
        irrs.append(_irr_single_stlib(row))

    sorted_lcm = sorted(lcms)
    n = len(sorted_lcm)
    mean = _stats.fmean(lcms)
    std = _stats.stdev(lcms) if n > 1 else 0.0
    summary = {
        "n": n,
        "mean": mean,
        "std": std,
        "median": _stats.median(lcms),
        "p05": _percentile(sorted_lcm, 5),
        "p25": _percentile(sorted_lcm, 25),
        "p75": _percentile(sorted_lcm, 75),
        "p95": _percentile(sorted_lcm, 95),
        "min": sorted_lcm[0],
        "max": sorted_lcm[-1],
        "var_5pct": _percentile(sorted_lcm, 5),
        "probability_of_loss": sum(1 for r in lcms if r < 0) / n,
        "skewness": float("nan"),
        "kurtosis": float("nan"),
    }

    # IRR-Kennzahlen (NaNs ignorieren)
    valid_irrs = [r for r in irrs if math.isfinite(r)]
    n_irr_valid = len(valid_irrs)
    sorted_irr = sorted(valid_irrs)
    if n_irr_valid:
        irr_mean = _stats.fmean(valid_irrs)
        irr_std = _stats.stdev(valid_irrs) if n_irr_valid > 1 else 0.0
        irr_summary = {
            "n": n_irr_valid,
            "n_total": n,
            "n_nan": n - n_irr_valid,
            "mean": irr_mean,
            "std": irr_std,
            "median": _stats.median(valid_irrs),
            "p05": _percentile(sorted_irr, 5),
            "p25": _percentile(sorted_irr, 25),
            "p75": _percentile(sorted_irr, 75),
            "p95": _percentile(sorted_irr, 95),
            "min": sorted_irr[0],
            "max": sorted_irr[-1],
            "var_5pct": _percentile(sorted_irr, 5),
            "probability_of_loss": sum(1 for r in valid_irrs if r < 0) / n_irr_valid,
            "skewness": float("nan"),
            "kurtosis": float("nan"),
        }
    else:
        irr_summary = {
            "n": 0,
            "n_total": n,
            "n_nan": n,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p05": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "var_5pct": float("nan"),
            "probability_of_loss": float("nan"),
            "skewness": float("nan"),
            "kurtosis": float("nan"),
        }

    # Sensitivität: Pearson-Korrelation, in Python per Hilfsformel
    def _corr(xs: list[float], ys: list[float]) -> float:
        mx, my = _stats.fmean(xs), _stats.fmean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        deny = math.sqrt(sum((y - my) ** 2 for y in ys))
        if denx == 0 or deny == 0:
            return 0.0
        return num / (denx * deny)

    # Sensitivität bzgl. LCM (sekundär)
    total = 0.0
    sens_rows: list[dict[str, float]] = []
    for name, xs in samples.items():
        r = _corr(xs, lcms)
        r2 = r * r
        sens_rows.append({"variable": name, "pearson_r": r, "r_squared": r2})
        total += r2
    for row in sens_rows:
        row["relative_influence_pct"] = (
            100.0 * row["r_squared"] / total if total > 0 else 0.0
        )
    sens_rows.sort(key=lambda r: r["relative_influence_pct"], reverse=True)

    # Sensitivität bzgl. IRR (primär)
    sens_rows_irr: list[dict[str, float]] = []
    if n_irr_valid > 1:
        # nur finite IRR-Draws verwenden
        valid_idx = [i for i, r in enumerate(irrs) if math.isfinite(r)]
        total_irr = 0.0
        for name, xs in samples.items():
            paired_x = [xs[i] for i in valid_idx]
            paired_y = [irrs[i] for i in valid_idx]
            r = _corr(paired_x, paired_y)
            r2 = r * r
            sens_rows_irr.append(
                {"variable": name, "pearson_r": r, "r_squared": r2}
            )
            total_irr += r2
        for row in sens_rows_irr:
            row["relative_influence_pct"] = (
                100.0 * row["r_squared"] / total_irr if total_irr > 0 else 0.0
            )
        sens_rows_irr.sort(
            key=lambda r: r["relative_influence_pct"], reverse=True
        )

    # Bootstrap-Konfidenzintervall (echtes Resampling) für den Mittelwert
    boot_rng = _random.Random(config.seed + 1)
    boot_means: list[float] = []
    n_boot = min(config.bootstrap_samples, 1000)
    for _ in range(n_boot):
        sample = [lcms[boot_rng.randrange(n)] for _ in range(n)]
        boot_means.append(_stats.fmean(sample))
    boot = {
        "mean": _stats.fmean(boot_means),
        "ci_low": _percentile(sorted(boot_means), 2.5),
        "ci_high": _percentile(sorted(boot_means), 97.5),
        "std_error": _stats.stdev(boot_means) if n_boot > 1 else 0.0,
    }

    # Bootstrap-Konfidenzintervall für IRR-Mittelwert
    if n_irr_valid > 1:
        boot_means_irr: list[float] = []
        for _ in range(n_boot):
            sample = [valid_irrs[boot_rng.randrange(n_irr_valid)]
                      for _ in range(n_irr_valid)]
            boot_means_irr.append(_stats.fmean(sample))
        boot_irr = {
            "mean": _stats.fmean(boot_means_irr),
            "ci_low": _percentile(sorted(boot_means_irr), 2.5),
            "ci_high": _percentile(sorted(boot_means_irr), 97.5),
            "std_error": _stats.stdev(boot_means_irr) if n_boot > 1 else 0.0,
        }
    else:
        boot_irr = {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "std_error": float("nan"),
        }

    files: list[Path] = []
    result = SimulationResult(
        config=config,
        samples={k: v for k, v in samples.items()},  # type: ignore[dict-item]
        lcm=lcms,  # type: ignore[arg-type]
        irr=irrs,  # type: ignore[arg-type]
        cashflows=cashflow_rows,  # type: ignore[arg-type]
        summary=summary,
        irr_summary=irr_summary,
        bootstrap=boot,
        bootstrap_irr=boot_irr,
        sensitivity=sens_rows,
        sensitivity_irr=sens_rows_irr,
        files=files,
    )

    json_path = output_dir / "results.json"
    result.files.append(json_path)
    json_path.write_text(json.dumps(result.to_json(), indent=2,
                                    ensure_ascii=False))
    return result


# ---------------------------------------------------------------------------
# Statistik
# ---------------------------------------------------------------------------


def summarize_lcm(lcm: "np.ndarray", confidence: float) -> dict[str, float]:
    """Berechnet die im Paper berichteten Kennzahlen für die LCM-Stichprobe."""
    _ensure_numpy()
    n = lcm.size
    var_alpha = (1.0 - confidence) * 100.0
    var_value = float(np.percentile(lcm, var_alpha))
    return {
        "n": int(n),
        "mean": float(lcm.mean()),
        "std": float(lcm.std(ddof=1)),
        "median": float(np.median(lcm)),
        "p05": float(np.percentile(lcm, 5)),
        "p25": float(np.percentile(lcm, 25)),
        "p75": float(np.percentile(lcm, 75)),
        "p95": float(np.percentile(lcm, 95)),
        "min": float(lcm.min()),
        "max": float(lcm.max()),
        "var_5pct": var_value,
        "probability_of_loss": float((lcm < 0).mean()),
        "skewness": float(((lcm - lcm.mean()) ** 3).mean() / lcm.std(ddof=1) ** 3),
        "kurtosis": float(((lcm - lcm.mean()) ** 4).mean() / lcm.std(ddof=1) ** 4 - 3.0),
    }


def summarize_roi(roi: "np.ndarray", confidence: float) -> dict[str, float]:
    """Deprecated alias für :func:`summarize_lcm`.

    Wird für eine Release-Phase als Rückwärtskompatibilitäts-Shim
    vorgehalten. Neue Aufrufe sollten ``summarize_lcm`` verwenden.
    """
    return summarize_lcm(roi, confidence)


def bootstrap_mean(
    lcm: "np.ndarray", samples: int, confidence: float, rng: "np.random.Generator"
) -> dict[str, float]:
    """Bootstrap-Konfidenzintervall für den Erwartungswert."""
    _ensure_numpy()
    means = np.empty(samples, dtype=float)
    n = lcm.size
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        means[i] = lcm[idx].mean()
    alpha = 1.0 - confidence
    return {
        "mean": float(means.mean()),
        "ci_low": float(np.percentile(means, 100 * alpha / 2)),
        "ci_high": float(np.percentile(means, 100 * (1.0 - alpha / 2))),
        "std_error": float(means.std(ddof=1)),
    }


def sensitivity(
    samples: dict[str, "np.ndarray"], lcm: "np.ndarray"
) -> list[dict[str, float]]:
    """Pearson-Korrelation + normiertes R² für jede Inputvariable."""
    _ensure_numpy()
    total = 0.0
    rows: list[dict[str, float]] = []
    for name, values in samples.items():
        r = float(np.corrcoef(values, lcm)[0, 1])
        r2 = r * r
        rows.append({"variable": name, "pearson_r": r, "r_squared": r2})
        total += r2
    for row in rows:
        row["relative_influence_pct"] = (
            100.0 * row["r_squared"] / total if total > 0 else 0.0
        )
    rows.sort(key=lambda row: row["relative_influence_pct"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# IRR-Kennzahlen / Sensitivität
# ---------------------------------------------------------------------------


def summarize_irr(irr_arr: "np.ndarray", confidence: float) -> dict[str, float]:
    """Kennzahlen der IRR-Stichprobe (NaNs werden ignoriert)."""
    _ensure_numpy()
    finite = irr_arr[np.isfinite(irr_arr)]
    n_total = int(irr_arr.size)
    n_valid = int(finite.size)
    n_nan = n_total - n_valid
    if n_valid == 0:
        nan_keys = ("mean", "std", "median", "p05", "p25", "p75", "p95",
                    "min", "max", "var_5pct", "probability_of_loss",
                    "skewness", "kurtosis")
        base = {k: float("nan") for k in nan_keys}
        base.update({"n": 0, "n_total": n_total, "n_nan": n_nan,
                     "confidence": float(confidence)})
        return base
    var_alpha = (1.0 - confidence) * 100.0
    var_value = float(np.percentile(finite, var_alpha))
    std = float(finite.std(ddof=1)) if n_valid > 1 else 0.0
    mean = float(finite.mean())
    return {
        "n": n_valid,
        "n_total": n_total,
        "n_nan": n_nan,
        "mean": mean,
        "std": std,
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "var_5pct": var_value,
        "probability_of_loss": float((finite < 0).mean()),
        "skewness": (
            float(((finite - mean) ** 3).mean() / std ** 3)
            if std > 0 else float("nan")
        ),
        "kurtosis": (
            float(((finite - mean) ** 4).mean() / std ** 4 - 3.0)
            if std > 0 else float("nan")
        ),
        "confidence": float(confidence),
    }


def sensitivity_irr(
    samples: dict[str, "np.ndarray"], irr_arr: "np.ndarray"
) -> list[dict[str, float]] | None:
    """Pearson-Korrelation der Inputs mit dem IRR (nur finite Werte)."""
    _ensure_numpy()
    mask = np.isfinite(irr_arr)
    if int(mask.sum()) < 2:
        return None
    total = 0.0
    rows: list[dict[str, float]] = []
    for name, values in samples.items():
        r = float(np.corrcoef(values[mask], irr_arr[mask])[0, 1])
        if not math.isfinite(r):
            continue
        r2 = r * r
        rows.append({"variable": name, "pearson_r": r, "r_squared": r2})
        total += r2
    if total <= 0:
        return rows
    for row in rows:
        row["relative_influence_pct"] = (
            100.0 * row["r_squared"] / total
        )
    rows.sort(key=lambda row: row["relative_influence_pct"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_inputs(samples: dict[str, "np.ndarray"], output_dir: Path) -> Path:
    """Histogramme + theoretische Dichte der vier Inputvariablen."""
    _ensure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    titles = {
        "capex": "CAPEX ($M, Dreiecksverteilung)",
        "opex": "OPEX ($M, Dreiecksverteilung)",
        "volume": "Fördervolumen (M Barrel, Dreiecksverteilung)",
        "price": "Ölpreis ($/Barrel, Lognormalverteilung)",
    }
    for ax, (name, values) in zip(axes.ravel(), samples.items()):
        ax.hist(values, bins=60, density=True, color="#4c72b0", alpha=0.7,
                edgecolor="white")
        if _ensure_scipy() and name in {"capex", "opex", "volume"}:
            params = {
                "capex": (8_000.0, 10_000.0, 12_000.0),
                "opex": (1_500.0, 2_200.0, 3_000.0),
                "volume": (500.0, 1_000.0, 2_000.0),
            }[name]
            xs = np.linspace(params[0], params[2], 400)
            ax.plot(xs, _scipy_stats.triang.pdf(
                xs, c=(params[1] - params[0]) / (params[2] - params[0]),
                loc=params[0], scale=params[2] - params[0],
            ), color="black", lw=1.5, label="Theoretische Dichte")
            ax.legend(loc="upper right", fontsize=9)
        elif _ensure_scipy() and name == "price":
            xs = np.linspace(1.0, max(200.0, float(values.max()) * 1.05), 400)
            mu = math.log(70.0**2 / math.sqrt(25.0**2 + 70.0**2))
            sigma = math.sqrt(math.log(1.0 + (25.0 / 70.0) ** 2))
            ax.plot(xs, _scipy_stats.lognorm.pdf(xs, s=sigma, scale=math.exp(mu)),
                    color="black", lw=1.5, label="Theoretische Dichte")
            ax.legend(loc="upper right", fontsize=9)
        ax.set_title(titles[name])
        ax.set_xlabel(name.upper())
        ax.set_ylabel("Dichte")

    fig.suptitle("Empirische Verteilungen der Inputvariablen (N = {:,})"
                 .format(len(samples["capex"])), fontsize=13)
    target = output_dir / "inputs_distribution.png"
    _save(fig, target)
    return target


def plot_lcm_distribution(
    lcm: "np.ndarray", summary: dict[str, float], output_dir: Path
) -> Path:
    _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = 80
    counts, edges, patches = ax.hist(lcm, bins=bins, density=True,
                                     color="#4c72b0", alpha=0.75,
                                     edgecolor="white", label="Stichprobe")
    if _ensure_scipy():
        xs = np.linspace(lcm.min(), lcm.max(), 400)
        kde = _scipy_stats.gaussian_kde(lcm)
        ax.plot(xs, kde(xs), color="black", lw=1.5, label="KDE")

    ax.axvline(0, color="#c44e52", lw=1.2, ls="--",
               label=f"P(LCM<0) = {summary['probability_of_loss']:.1%}")
    ax.axvline(summary["mean"], color="#2ca02c", lw=1.2, ls="-",
               label=f"Mean = {summary['mean']:.1%}")
    ax.axvline(summary["median"], color="#dd8452", lw=1.2, ls="-",
               label=f"Median = {summary['median']:.1%}")
    ax.axvline(summary["var_5pct"], color="#9467bd", lw=1.2, ls=":",
               label=f"VaR 5% = {summary['var_5pct']:.1%}")

    text = (
        f"N = {summary['n']:,}\n"
        f"Mean = {summary['mean']:.2%}\n"
        f"Median = {summary['median']:.2%}\n"
        f"Std = {summary['std']:.2%}\n"
        f"VaR 5% = {summary['var_5pct']:.2%}\n"
        f"P(LCM<0) = {summary['probability_of_loss']:.2%}"
    )
    ax.text(0.98, 0.97, text, transform=ax.transAxes, ha="right", va="top",
            family="monospace", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="lightgray"))

    ax.set_title("Lifetime Capital Multiple — Verteilung der Monte-Carlo-Simulation")
    ax.set_xlabel("Lifetime Capital Multiple")
    ax.set_ylabel("Dichte")
    ax.legend(loc="upper left", fontsize=9)
    target = output_dir / "lcm_distribution.png"
    _save(fig, target)
    return target


def plot_roi_distribution(
    roi: "np.ndarray", summary: dict[str, float], output_dir: Path
) -> Path:
    """Deprecated alias für :func:`plot_lcm_distribution`.

    Wird für eine Release-Phase als Rückwärtskompatibilitäts-Shim
    vorgehalten. Neue Aufrufe sollten ``plot_lcm_distribution`` verwenden.
    """
    return plot_lcm_distribution(roi, summary, output_dir)


def plot_irr_distribution(
    irr_arr: "np.ndarray", summary: dict[str, float], output_dir: Path
) -> Path:
    """Histogramm + KDE der IRR-Stichprobe (annualisiert)."""
    _ensure_numpy()
    _ensure_matplotlib()
    finite = irr_arr[np.isfinite(irr_arr)]
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = 80
    ax.hist(finite, bins=bins, density=True, color="#55a868", alpha=0.75,
            edgecolor="white", label="Stichprobe (IRR)")
    if _ensure_scipy() and finite.size > 2 and float(finite.std()) > 0:
        xs = np.linspace(float(finite.min()), float(finite.max()), 400)
        kde = _scipy_stats.gaussian_kde(finite)
        ax.plot(xs, kde(xs), color="black", lw=1.5, label="KDE")

    ax.axvline(0, color="#c44e52", lw=1.2, ls="--",
               label=f"P(IRR<0) = {summary['probability_of_loss']:.1%}")
    ax.axvline(summary["mean"], color="#2ca02c", lw=1.2, ls="-",
               label=f"Mean = {summary['mean']:.1%}")
    ax.axvline(summary["median"], color="#dd8452", lw=1.2, ls="-",
               label=f"Median = {summary['median']:.1%}")
    ax.axvline(summary["var_5pct"], color="#9467bd", lw=1.2, ls=":",
               label=f"VaR 5% = {summary['var_5pct']:.1%}")

    text = (
        f"N (valid) = {summary['n']:,}\n"
        f"N (total) = {summary['n_total']:,}\n"
        f"Mean IRR  = {summary['mean']:.2%}\n"
        f"Median IRR= {summary['median']:.2%}\n"
        f"Std       = {summary['std']:.2%}\n"
        f"VaR 5%    = {summary['var_5pct']:.2%}\n"
        f"P(IRR<0)  = {summary['probability_of_loss']:.2%}"
    )
    ax.text(0.98, 0.97, text, transform=ax.transAxes, ha="right", va="top",
            family="monospace", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="lightgray"))

    ax.set_title("Internal Rate of Return — Verteilung der Monte-Carlo-Simulation")
    ax.set_xlabel("IRR (annualisiert)")
    ax.set_ylabel("Dichte")
    ax.legend(loc="upper left", fontsize=9)
    target = output_dir / "irr_distribution.png"
    _save(fig, target)
    return target


def plot_tornado(sensitivity_rows: Sequence[dict[str, float]], output_dir: Path) -> Path:
    _ensure_matplotlib()
    names = [row["variable"].upper() for row in sensitivity_rows]
    values = [row["relative_influence_pct"] for row in sensitivity_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names[::-1], values[::-1], color="#4c72b0", edgecolor="white")
    ax.set_xlabel("Anteil an erklärter LCM-Varianz (%)")
    ax.set_title("Tornado-Diagramm — Sensitivitätsanalyse (LCM)")
    ax.set_xlim(0, max(values) * 1.15 if values else 1.0)
    for bar, value in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%", va="center", fontsize=10)
    target = output_dir / "sensitivity_tornado.png"
    _save(fig, target)
    return target


def plot_tornado_irr(sensitivity_rows: Sequence[dict[str, float]],
                     output_dir: Path) -> Path:
    """Tornado-Diagramm der Sensitivitätsanalyse bzgl. IRR."""
    _ensure_matplotlib()
    names = [row["variable"].upper() for row in sensitivity_rows]
    values = [row["relative_influence_pct"] for row in sensitivity_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names[::-1], values[::-1], color="#55a868", edgecolor="white")
    ax.set_xlabel("Anteil an erklärter IRR-Varianz (%)")
    ax.set_title("Tornado-Diagramm — Sensitivitätsanalyse (IRR)")
    ax.set_xlim(0, max(values) * 1.15 if values else 1.0)
    for bar, value in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%", va="center", fontsize=10)
    target = output_dir / "sensitivity_tornado_irr.png"
    _save(fig, target)
    return target


# ---------------------------------------------------------------------------
# Orchestrierung
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    config: SimulationConfig
    samples: dict[str, Any]
    lcm: Any
    summary: dict[str, float]
    bootstrap: dict[str, float]
    sensitivity: list[dict[str, float]]
    files: list[Path] = field(default_factory=list)
    # Optional Felder für IRR / Cashflows (siehe [ELI-28](/ELI/issues/ELI-28)).
    # Vor dem IRR-Refactor waren diese nicht vorhanden — ``None`` bedeutet
    # "nicht berechnet" (z. B. wenn der Aufrufer ``compute_lcm`` direkt
    # verwendet, ohne ``run()``).
    irr: Any = None
    cashflows: Any = None
    irr_summary: dict[str, float] | None = None
    bootstrap_irr: dict[str, float] | None = None
    sensitivity_irr: list[dict[str, float]] | None = None

    @property
    def roi(self) -> Any:
        """Deprecated Alias für :attr:`lcm` (Rückwärtskompatibilität).

        Externe Konsumenten, die noch ``result.roi`` lesen, bekommen
        weiterhin die Stichprobe zurück. Neue Aufrufe sollten
        ``result.lcm`` verwenden.
        """
        return self.lcm

    def to_json(self) -> dict:
        def _round(value):
            if value is None:
                return None
            try:
                result = float(round(value, 6))
            except (TypeError, ValueError):
                return value
            if math.isnan(result) or math.isinf(result):
                return None
            return result

        rounded_summary = {k: _round(v) for k, v in self.summary.items()}
        rounded_bootstrap = {k: _round(v) for k, v in self.bootstrap.items()}
        irr_block = None
        if self.irr_summary is not None:
            irr_block = {
                "summary": {k: _round(v) for k, v in self.irr_summary.items()},
                "bootstrap_mean": (
                    {k: _round(v) for k, v in self.bootstrap_irr.items()}
                    if self.bootstrap_irr is not None else None
                ),
                "sensitivity": (
                    [{k: _round(v) for k, v in row.items()}
                     for row in self.sensitivity_irr]
                    if self.sensitivity_irr is not None else None
                ),
                "config": {
                    "project_life_years": self.config.project_life_years,
                    "decline_curve": self.config.decline_curve,
                    "decline_a": self.config.decline_a,
                    "decline_b": self.config.decline_b,
                },
            }
        return {
            "config": {
                "iterations": self.config.iterations,
                "seed": self.config.seed,
                "bootstrap_samples": self.config.bootstrap_samples,
                "bootstrap_confidence": self.config.bootstrap_confidence,
                "project_life_years": self.config.project_life_years,
                "decline_curve": self.config.decline_curve,
                "decline_a": self.config.decline_a,
                "decline_b": self.config.decline_b,
                "capex": asdict(self.config.capex),
                "opex": asdict(self.config.opex),
                "volume": asdict(self.config.volume),
                "price": asdict(self.config.price),
                "expected_inputs": {k: _round(v)
                                    for k, v in self.config.expected_metrics().items()},
            },
            "irr": irr_block,
            "lifetime_capital_multiple": {
                "summary": rounded_summary,
                "bootstrap_mean": rounded_bootstrap,
            },
            "sensitivity": [
                {k: _round(v) for k, v in row.items()}
                for row in self.sensitivity
            ],
            "artifacts": [str(p) for p in self.files],
            # Deprecated Top-Level-Aliase (eine Release-Phase): identische
            # Daten wie ``lifetime_capital_multiple``, damit ältere
            # Konsumenten ohne Schema-Migration weiter funktionieren.
            "summary": rounded_summary,
            "bootstrap_mean": rounded_bootstrap,
            "roi": {
                "summary": rounded_summary,
                "bootstrap_mean": rounded_bootstrap,
            },
        }


def run(
    config: SimulationConfig | None = None,
    output_dir: Path | str = Path("output"),
    write_csv: bool = True,
) -> SimulationResult:
    """Führt die komplette Monte-Carlo-Simulation aus und schreibt Artefakte.

    Bevorzugt den schnellen numpy/matplotlib-Pfad. Falls weder numpy noch
    matplotlib verfügbar sind, wird automatisch auf einen langsameren
    Stdlib-Pfad zurückgegriffen (kein CSV, keine Plots — nur JSON-Report).

    Ab [ELI-28](/ELI/issues/ELI-28) werden zusätzlich zum sekundären LCM
    primär ein **Internal Rate of Return** (IRR) auf Basis der
    jahresweisen Cashflowreihe (``generate_cashflows``) berechnet und im
    Report ausgewiesen.
    """
    if config is None:
        config = default_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not _has_numpy():
        print("[HINWEIS] numpy fehlt – nutze Stdlib-Fallback (langsamer, keine Plots).")
        return _run_stlib(config=config, output_dir=output_dir, write_csv=False)

    _ensure_numpy()
    rng = np.random.default_rng(config.seed)
    samples = sample_inputs(config, rng)
    lcm = compute_lcm(samples)
    summary = summarize_lcm(lcm, confidence=config.bootstrap_confidence)
    boot = bootstrap_mean(lcm, config.bootstrap_samples, config.bootstrap_confidence, rng)
    sens = sensitivity(samples, lcm)

    # IRR: Cashflow-Matrix + Root-Finding
    cashflows = generate_cashflows(
        capex=samples["capex"],
        opex=samples["opex"],
        volume=samples["volume"],
        price=samples["price"],
        project_life_years=config.project_life_years,
        decline_curve=config.decline_curve,
        decline_a=config.decline_a,
        decline_b=config.decline_b,
    )
    irr_arr = irr(cashflows)
    irr_summary = summarize_irr(irr_arr,
                                confidence=config.bootstrap_confidence)
    boot_irr = bootstrap_mean(
        irr_arr,
        config.bootstrap_samples,
        config.bootstrap_confidence,
        rng,
    )
    sens_irr = sensitivity_irr(samples, irr_arr)

    files: list[Path] = []
    if _has_matplotlib():
        _ensure_matplotlib()
        files.append(plot_inputs(samples, output_dir))
        files.append(plot_lcm_distribution(lcm, summary, output_dir))
        files.append(plot_irr_distribution(irr_arr, irr_summary, output_dir))
        files.append(plot_tornado(sens, output_dir))
        if sens_irr is not None:
            files.append(plot_tornado_irr(sens_irr, output_dir))
    else:
        print("[WARNUNG] matplotlib fehlt – überspringe Plots.")

    if write_csv:
        csv_path = output_dir / "samples.csv"
        header = ",".join([
            "capex", "opex", "volume", "price",
            "lifetime_capital_multiple", "irr",
        ])
        stack = np.column_stack([samples["capex"], samples["opex"],
                                 samples["volume"], samples["price"],
                                 lcm, irr_arr])
        np.savetxt(csv_path, stack, delimiter=",", header=header,
                   comments="", fmt="%.6f")
        files.append(csv_path)

    result = SimulationResult(
        config=config,
        samples=samples,
        lcm=lcm,
        irr=irr_arr,
        cashflows=cashflows,
        summary=summary,
        irr_summary=irr_summary,
        bootstrap=boot,
        bootstrap_irr=boot_irr,
        sensitivity=sens,
        sensitivity_irr=sens_irr,
        files=files,
    )

    json_path = output_dir / "results.json"
    result.files.append(json_path)
    json_path.write_text(json.dumps(result.to_json(), indent=2,
                                    ensure_ascii=False))
    return result


def _format_pct(value: float) -> str:
    return f"{value * 100:6.2f}%"


def print_report(result: SimulationResult) -> None:
    print("\n=== Monte-Carlo IRR / LCM — Ergebnisreport ===")
    print("Primäre Kennzahl: Internal Rate of Return (IRR, annualisiert).")
    print("Sekundäre Kennzahl: Lifetime Capital Multiple (LCM, kumulativ).")
    cfg = result.config
    print(f"\nIterationen : {cfg.iterations:,}  (Seed = {cfg.seed})")
    print(f"\nProjekt-Lebensdauer & Decline:")
    print(f"  project_life_years = {cfg.project_life_years}")
    print(f"  decline_curve      = {cfg.decline_curve}")
    if cfg.decline_curve != "flat":
        print(f"  decline_a          = {cfg.decline_a}")
    if cfg.decline_curve == "hyperbolic":
        print(f"  decline_b          = {cfg.decline_b}")
    print(f"\nInput-Erwartungswerte (analytisch):")
    for key, value in cfg.expected_metrics().items():
        if key.startswith("decline_") or key == "project_life_years":
            print(f"  {key:18s} = {value}")
        else:
            print(f"  {key:18s} = {value:10.2f} $M")
    if result.irr_summary is not None:
        print("\nInternal Rate of Return — Kennzahlen (Stichprobe, primär):")
        s = result.irr_summary
        if s["n"] > 0:
            print(f"  N (valid/total) = {s['n']:,} / {s['n_total']:,}  "
                  f"(NaN = {s['n_nan']})")
            print(f"  Mean     = {_format_pct(s['mean'])}")
            print(f"  Median   = {_format_pct(s['median'])}")
            print(f"  Std      = {_format_pct(s['std'])}")
            print(f"  P05      = {_format_pct(s['p05'])}")
            print(f"  P95      = {_format_pct(s['p95'])}")
            print(f"  VaR 5%   = {_format_pct(s['var_5pct'])}")
            print(f"  P(IRR<0) = {_format_pct(s['probability_of_loss'])}")
            print(f"  Min/Max  = {_format_pct(s['min'])} / {_format_pct(s['max'])}")
            if result.bootstrap_irr is not None:
                b = result.bootstrap_irr
                print(
                    f"\nBootstrap-{int(cfg.bootstrap_confidence*100)}%-KI "
                    f"für IRR-Mean: "
                    f"[{b['ci_low']:.2%}, {b['ci_high']:.2%}]  "
                    f"(SE = {b['std_error']:.2%})"
                )
        else:
            print("  Keine finite IRR-Stichprobe — alle Draws sind NaN.")
    print("\nLifetime Capital Multiple — Kennzahlen (Stichprobe, sekundär):")
    s = result.summary
    print(f"  Mean     = {_format_pct(s['mean'])}")
    print(f"  Median   = {_format_pct(s['median'])}")
    print(f"  Std      = {_format_pct(s['std'])}")
    print(f"  P05      = {_format_pct(s['p05'])}")
    print(f"  P95      = {_format_pct(s['p95'])}")
    print(f"  VaR 5%   = {_format_pct(s['var_5pct'])}")
    print(f"  P(LCM<0) = {_format_pct(s['probability_of_loss'])}")
    print(f"  Min/Max  = {_format_pct(s['min'])} / {_format_pct(s['max'])}")
    b = result.bootstrap
    print(f"\nBootstrap-{int(cfg.bootstrap_confidence*100)}%-KI für LCM-Mean: "
          f"[{b['ci_low']:.2%}, {b['ci_high']:.2%}]  (SE = {b['std_error']:.2%})")
    if result.sensitivity_irr is not None and len(result.sensitivity_irr) > 0:
        print("\nSensitivitätsanalyse (relativer Varianzanteil am IRR, primär):")
        for row in result.sensitivity_irr:
            print(f"  {row['variable']:7s} r = {row['pearson_r']:+.3f}  "
                  f"R² = {row['r_squared']:.3f}  Anteil = {row['relative_influence_pct']:.1f}%")
    print("\nSensitivitätsanalyse (relativer Varianzanteil am LCM, sekundär):")
    for row in result.sensitivity:
        print(f"  {row['variable']:7s} r = {row['pearson_r']:+.3f}  "
              f"R² = {row['r_squared']:.3f}  Anteil = {row['relative_influence_pct']:.1f}%")
    print("\nArtefakte:")
    for path in result.files:
        print(f"  - {path}")
    print()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    # Offshore-Deepwater-Defaults (siehe [ELI-26](/ELI/issues/ELI-26)).
    # CAPEX/OPEX in $M, Volumen in M Barrel, Preis in $/Barrel.
    parser.add_argument("--capex-low", type=float, default=8_000.0,
                        help="CAPEX Untergrenze (Tri-Distribution, $M). Default 8000.")
    parser.add_argument("--capex-mode", type=float, default=10_000.0,
                        help="CAPEX Modus (Tri-Distribution, $M). Default 10000.")
    parser.add_argument("--capex-high", type=float, default=12_000.0,
                        help="CAPEX Obergrenze (Tri-Distribution, $M). Default 12000.")
    parser.add_argument("--opex-low", type=float, default=1_500.0,
                        help="OPEX Untergrenze (Tri-Distribution, $M). Default 1500.")
    parser.add_argument("--opex-mode", type=float, default=2_200.0,
                        help="OPEX Modus (Tri-Distribution, $M). Default 2200.")
    parser.add_argument("--opex-high", type=float, default=3_000.0,
                        help="OPEX Obergrenze (Tri-Distribution, $M). Default 3000.")
    parser.add_argument("--volume-low", type=float, default=500.0,
                        help="Fördervolumen Untergrenze (Tri-Distribution, M bbl). Default 500.")
    parser.add_argument("--volume-mode", type=float, default=1_000.0,
                        help="Fördervolumen Modus (Tri-Distribution, M bbl). Default 1000.")
    parser.add_argument("--volume-high", type=float, default=2_000.0,
                        help="Fördervolumen Obergrenze (Tri-Distribution, M bbl). Default 2000.")
    parser.add_argument("--price-mean", type=float, default=70.0,
                        help="Ölpreis-Erwartungswert (LogN, $/Barrel). Default 70.")
    parser.add_argument("--price-sigma", type=float, default=25.0,
                        help="Ölpreis-Streuung im Dollar-Raum (LogN, $/Barrel). Default 25.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--no-csv", action="store_true",
                        help="Schreibt keine samples.csv-Datei")
    parser.add_argument("--project-life-years", type=int, default=10,
                        help="Projekt-Lebensdauer in Jahren (T). Default 10.")
    parser.add_argument("--decline-curve", type=str, default="hyperbolic",
                        choices=["flat", "exponential", "hyperbolic"],
                        help="Form der Decline-Kurve. Default 'hyperbolic'.")
    parser.add_argument("--decline-a", type=float, default=0.05,
                        help="Annualisierte Anfangs-Abnahmerate (Arps 'a'). "
                             "Default 0.05.")
    parser.add_argument("--decline-b", type=float, default=0.5,
                        help="Arps 'b'-Faktor (hyperbolic). Default 0.5.")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = SimulationConfig(
        capex=TriangularParams(args.capex_low, args.capex_mode,
                               args.capex_high, "CAPEX"),
        opex=TriangularParams(args.opex_low, args.opex_mode,
                              args.opex_high, "OPEX"),
        volume=TriangularParams(args.volume_low, args.volume_mode,
                                args.volume_high, "Volume"),
        price=LogNormalParams(args.price_mean, args.price_sigma, "Price"),
        iterations=args.iterations,
        seed=args.seed,
        project_life_years=args.project_life_years,
        decline_curve=args.decline_curve,
        decline_a=args.decline_a,
        decline_b=args.decline_b,
    )
    result = run(config=config, output_dir=args.output_dir, write_csv=not args.no_csv)
    print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
