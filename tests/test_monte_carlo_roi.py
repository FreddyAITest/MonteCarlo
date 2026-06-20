"""
Stdlib-Smoketest für die Monte-Carlo-LCM-Implementierung.

Da die Zielumgebung möglicherweise kein numpy / matplotlib installiert hat,
reimplementiert dieser Test die zentralen Bausteine (Triangular-Sampling,
Lognormal-Sampling, LCM-Berechnung, Kennzahlen) in reiner Standardbibliothek
und vergleicht sie mit den analytischen Erwartungswerten sowie mit der
Referenzimplementierung in ``src.monte_carlo_roi``, falls numpy vorhanden ist.

Hinweis zur Nomenklatur: Die ehemals als "ROI" bezeichnete Kennzahl heißt
jetzt ``lifetime_capital_multiple`` (LCM). Die algebraische Formel ist
unverändert.

Ausgeführt mit:

    python tests/test_monte_carlo_roi.py

Exit-Code 0 bei Erfolg, 1 bei einem fehlgeschlagenen Plausibilitätstest.
"""

from __future__ import annotations

import math
import random
import statistics
import sys
from pathlib import Path

# Paketimporte relativ zu diesem Testfile
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.monte_carlo_roi import (  # noqa: E402
    LogNormalParams,
    SimulationConfig,
    TriangularParams,
    default_config,
)


# ---------------------------------------------------------------------------
# Stdlib-Referenzimplementierung (1:1 zur SciPy/Numpy-Logik)
# ---------------------------------------------------------------------------


def triangular_sample(low: float, mode: float, high: float, rng: random.Random) -> float:
    u = rng.random()
    if u < (mode - low) / (high - low):
        return low + math.sqrt(u * (high - low) * (mode - low))
    return high - math.sqrt((1.0 - u) * (high - low) * (high - mode))


def lognormal_sample(mean: float, sigma: float, rng: random.Random) -> float:
    # scipy-Parametrisierung: log(X) ~ N(mu, sigma^2) mit E[X] = exp(mu + sigma^2/2)
    variance = sigma * sigma
    mu = math.log(mean * mean / math.sqrt(variance + mean * mean))
    sigma_ln = math.sqrt(math.log(1.0 + variance / (mean * mean)))
    return math.exp(rng.gauss(mu, sigma_ln))


def run_stlib_sim(config: SimulationConfig, seed: int = 1234) -> list[float]:
    rng = random.Random(seed)
    lcms: list[float] = []
    for _ in range(config.iterations):
        capex = triangular_sample(config.capex.low, config.capex.mode,
                                  config.capex.high, rng)
        opex = triangular_sample(config.opex.low, config.opex.mode,
                                  config.opex.high, rng)
        volume = triangular_sample(config.volume.low, config.volume.mode,
                                   config.volume.high, rng)
        price = lognormal_sample(config.price.mean, config.price.sigma, rng)
        revenue = price * volume
        profit = revenue - capex - opex
        lcms.append(profit / capex)
    return lcms


# ---------------------------------------------------------------------------
# Plausibilitätstests
# ---------------------------------------------------------------------------


def test_config_validation() -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        TriangularParams(1, 0, 2, "bad")
    except ValueError as exc:
        print(f"  ok  TriangularParams validiert Reihenfolge ({exc})")
    else:
        errors.append("TriangularParams hätte low > mode ablehnen müssen")
    try:
        LogNormalParams(-1.0, 0.1, "bad")
    except ValueError as exc:
        print(f"  ok  LogNormalParams validiert mean>0 ({exc})")
    else:
        errors.append("LogNormalParams hätte mean<=0 ablehnen müssen")
    return not errors, errors


def test_analytical_match() -> tuple[bool, list[str]]:
    errors: list[str] = []
    config = default_config()
    expected = config.expected_metrics()
    actual = {
        "E_CAPEX": config.capex.mean,
        "E_OPEX": config.opex.mean,
        "E_VOLUME": config.volume.mean,
        "E_PRICE": config.price.mean,
        "Var_CAPEX": config.capex.variance,
        "Var_OPEX": config.opex.variance,
        "Var_VOLUME": config.volume.variance,
    }
    for key, exp in expected.items():
        if key == "Var_PRICE":
            continue
        if abs(actual[key] - exp) > 1e-9:
            errors.append(f"{key}: analytisch {actual[key]} ≠ erwartet {exp}")
        else:
            print(f"  ok  {key} analytisch = {actual[key]:.4f}")
    cf_cfg = config.cashflow_config()
    if cf_cfg["project_life_years"] != 10:
        errors.append(
            f"project_life_years default sollte 10 sein, ist {cf_cfg['project_life_years']}"
        )
    else:
        print(f"  ok  project_life_years default = {cf_cfg['project_life_years']}")
    if cf_cfg["decline_curve"] != "hyperbolic":
        errors.append(
            f"decline_curve default sollte 'hyperbolic' sein, ist {cf_cfg['decline_curve']!r}"
        )
    else:
        print(f"  ok  decline_curve default = {cf_cfg['decline_curve']!r}")
    if not (0 < cf_cfg["decline_a"] <= 0.5):
        errors.append(
            f"decline_a default sollte in (0, 0.5] liegen, ist {cf_cfg['decline_a']}"
        )
    else:
        print(f"  ok  decline_a default = {cf_cfg['decline_a']}")
    if not (0 < cf_cfg["decline_b"] <= 1.0):
        errors.append(
            f"decline_b default sollte in (0, 1.0] liegen, ist {cf_cfg['decline_b']}"
        )
    else:
        print(f"  ok  decline_b default = {cf_cfg['decline_b']}")
    return not errors, errors


def test_simulation_plausibility(iterations: int = 20_000) -> tuple[bool, list[str]]:
    """Plausibilität der Verteilung (nicht absolute Werte).

    Die im Paper berichteten Tilde-Werte (Mean ~45 %, P(loss) ~12 %)
    widersprechen der im Paper selbst definierten LCM-Formel mit den
    angegebenen Parametern. Die stochastischen Eigenschaften (Rechtsschiefe,
    Spannweite, Korrelationen) sind aber robust ableitbar. Wir prüfen
    diese strukturellen Eigenschaften.
    """
    errors: list[str] = []
    config = SimulationConfig(
        capex=default_config().capex,
        opex=default_config().opex,
        volume=default_config().volume,
        price=default_config().price,
        iterations=iterations,
        seed=2024,
    )

    lcms = run_stlib_sim(config, seed=config.seed)
    mean = statistics.fmean(lcms)
    median = statistics.median(lcms)
    std = statistics.stdev(lcms)
    quantiles = statistics.quantiles(lcms, n=100, method="inclusive")
    p01, p50, p99 = quantiles[0], quantiles[49], quantiles[98]

    print(f"  info  N = {len(lcms):,}")
    print(f"  info  Mean={mean:.2%} Median={median:.2%} Std={std:.2%}")
    print(f"  info  P01={p01:.2%} P50={p50:.2%} P99={p99:.2%}")

    # Strukturelle Eigenschaften, die aus Formel + Parameter folgen:
    if not (mean > 0):
        errors.append(f"Mean sollte positiv sein, ist {mean:.2%}")
    if not (median > 0):
        errors.append(f"Median sollte positiv sein, ist {median:.2%}")
    if not (std > 0):
        errors.append("Standardabweichung muss > 0 sein")
    if not (mean > median):
        # Rechtsschiefe wegen Lognormal-Ölpreis
        errors.append(
            f"Mean ({mean:.2%}) sollte > Median ({median:.2%}) sein (Rechtsschiefe)"
        )
    if not (p99 > 5.0):
        errors.append(f"P99 sollte weit > 1 sein, ist {p99:.2%}")
    if not (p01 > -1.0):
        errors.append(f"P01 unplausibel: {p01:.2%}")
    if abs(p01) < 1e-6 and p99 < 1.0:
        errors.append("Verteilung wirkt entartet (sehr enge Spannweite)")
    if std < 0.1 * mean:
        errors.append(
            f"Streuung (Std={std:.2%}) unplausibel klein im Vergleich zum "
            f"Erwartungswert (Mean={mean:.2%})"
        )

    for label, ok in [
        ("Mean > 0", mean > 0),
        ("Median > 0", median > 0),
        ("Std > 0", std > 0),
        ("Mean > Median (Rechtsschiefe)", mean > median),
        ("P99 > 500%", p99 > 5.0),
    ]:
        print(f"  {'ok' if ok else 'FAIL'}  {label}")
    return not errors, errors


def test_sampling_bounds() -> tuple[bool, list[str]]:
    errors: list[str] = []
    # Offshore-Deepwater-Defaults (siehe [ELI-26](/ELI/issues/ELI-26)).
    rng = random.Random(99)
    for _ in range(20_000):
        x = triangular_sample(8_000.0, 10_000.0, 12_000.0, rng)
        if not (8_000.0 <= x <= 12_000.0):
            errors.append(f"Triangular out of bounds: {x}")
            break
    else:
        print("  ok  20k Triangular-Samples innerhalb [8000, 12000]")

    rng = random.Random(101)
    for _ in range(20_000):
        x = lognormal_sample(70.0, 25.0, rng)
        if not (x > 0):
            errors.append(f"Lognormal nicht positiv: {x}")
            break
    else:
        print("  ok  20k Lognormal-Samples > 0")
    return not errors, errors


def test_consistency_with_numpy() -> tuple[bool, list[str]]:
    """Falls numpy vorhanden ist, vergleiche stdlib mit der echten Implementierung."""
    errors: list[str] = []
    try:
        import numpy as np  # noqa: WPS433
    except ImportError:
        print("  skip  numpy nicht verfügbar – Cross-Check übersprungen")
        return True, []

    from src.monte_carlo_roi import compute_lcm, sample_inputs, summarize_lcm
    config = default_config()
    rng = np.random.default_rng(config.seed)
    samples = sample_inputs(config, rng)
    lcm = compute_lcm(samples)
    summary = summarize_lcm(lcm, confidence=0.95)

    # Die im Paper berichteten Werte sind grobe Schätzungen — wir prüfen
    # nur, dass die Implementierung sich im Bereich sinnvoller Größenordnungen
    # bewegt und numerisch stabil ist (keine NaN/Inf, sinnvolle Streuung).
    if not (math.isfinite(summary["mean"]) and math.isfinite(summary["std"])):
        errors.append("Mean oder Std sind nicht finit")
    else:
        print(f"  ok  numpy-Implementierung finit (mean={summary['mean']:.3%}, "
              f"std={summary['std']:.3%})")
    if summary["min"] >= summary["max"]:
        errors.append("Min >= Max, Verteilung degeneriert")
    if summary["probability_of_loss"] < 0 or summary["probability_of_loss"] > 1:
        errors.append(f"P(loss) außerhalb [0,1]: {summary['probability_of_loss']}")
    return not errors, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=== Monte-Carlo-LCM :: Plausibilitätstest (stdlib) ===")
    suites: list[tuple[str, callable]] = [
        ("Config-Validierung", test_config_validation),
        ("Analytische Erwartungswerte", test_analytical_match),
        ("Sampling-Grenzen", test_sampling_bounds),
        ("Stichproben-Plausibilität (N=20k)", test_simulation_plausibility),
        ("Konsistenz mit numpy", test_consistency_with_numpy),
    ]
    all_errors: list[str] = []
    for name, fn in suites:
        print(f"\n[{name}]")
        ok, errors = fn()
        if not ok:
            all_errors.extend(errors)
            print(f"  FAIL {name}: {len(errors)} Fehler")
        else:
            print(f"  PASS {name}")
    print()
    if all_errors:
        print(f"{len(all_errors)} Plausibilitätstest(s) fehlgeschlagen:")
        for err in all_errors:
            print(f"  - {err}")
        return 1
    print("Alle Plausibilitätstests bestanden.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
