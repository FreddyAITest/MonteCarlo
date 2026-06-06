"""
Monte-Carlo-ROI-Simulation für ölindustrielle Investitionsprojekte.

Dieses Skript implementiert das im Working Paper "Stochastische ROI-Bewertung
mittels Monte-Carlo-Simulation" beschriebene Vier-Variablen-Modell und erzeugt
die im Paper referenzierten Abbildungen (Inputverteilungen, ROI-Histogramm,
Tornado-Diagramm der Sensitivitätsanalyse).

Aufruf:

    python -m src.monte_carlo_roi                       # Standardparameter
    python -m src.monte_carlo_roi --iterations 50000    # mehr Iterationen
    python -m src.monte_carlo_roi --seed 7 --output-dir figures

Ausgaben (im Output-Verzeichnis, default ``./output``):

    roi_distribution.pdf / .png   Histogramm + Kennzahlen der ROI-Stichprobe
    sensitivity_tornado.pdf / .png Tornado-Diagramm der Sensitivitätsanalyse
    inputs_distribution.pdf / .png Verteilungen der vier Inputvariablen
    results.json                  Kennzahlen + Stichproben-Konfidenz
    samples.csv                   Stichprobe (CAPEX, OPEX, Volumen, Preis, ROI)

Abhängigkeiten: numpy, matplotlib, scipy (für die Dichtefunktion der
Dreiecksverteilung, die in den Input-Plots als Referenzlinie dient).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    """Vollständige Konfiguration der Monte-Carlo-Simulation."""

    capex: TriangularParams
    opex: TriangularParams
    volume: TriangularParams
    price: LogNormalParams
    iterations: int = 10_000
    seed: int = 42
    bootstrap_samples: int = 1_000
    bootstrap_confidence: float = 0.95

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


def default_config() -> SimulationConfig:
    """Standardkonfiguration passend zu Abschnitt 4 des Papers."""
    return SimulationConfig(
        capex=TriangularParams(low=500.0, mode=750.0, high=1_200.0, name="CAPEX"),
        opex=TriangularParams(low=80.0, mode=120.0, high=200.0, name="OPEX"),
        volume=TriangularParams(low=50.0, mode=150.0, high=300.0, name="Volume"),
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


def compute_roi(samples: dict[str, "np.ndarray"]) -> "np.ndarray":
    """Berechnet den ROI nach der im Paper definierten Formel (alle Werte in $M).

    ROI = (Preis · Volumen − CAPEX − OPEX) / CAPEX

    Preise in $/Barrel, Volumen in M Barrel, CAPEX/OPEX in $M.
    Damit ist (Preis · Volumen) bereits in $M und alle Terme sind konsistent.
    """
    _ensure_numpy()
    revenue = samples["price"] * samples["volume"]
    profit = revenue - samples["capex"] - samples["opex"]
    return profit / samples["capex"]


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

    rois: list[float] = []
    for c, o, v, p in zip(samples["capex"], samples["opex"],
                          samples["volume"], samples["price"]):
        rois.append((p * v - c - o) / c)

    sorted_roi = sorted(rois)
    n = len(sorted_roi)
    mean = _stats.fmean(rois)
    std = _stats.stdev(rois) if n > 1 else 0.0
    summary = {
        "n": n,
        "mean": mean,
        "std": std,
        "median": _stats.median(rois),
        "p05": _percentile(sorted_roi, 5),
        "p25": _percentile(sorted_roi, 25),
        "p75": _percentile(sorted_roi, 75),
        "p95": _percentile(sorted_roi, 95),
        "min": sorted_roi[0],
        "max": sorted_roi[-1],
        "var_5pct": _percentile(sorted_roi, 5),
        "probability_of_loss": sum(1 for r in rois if r < 0) / n,
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

    total = 0.0
    sens_rows: list[dict[str, float]] = []
    for name, xs in samples.items():
        r = _corr(xs, rois)
        r2 = r * r
        sens_rows.append({"variable": name, "pearson_r": r, "r_squared": r2})
        total += r2
    for row in sens_rows:
        row["relative_influence_pct"] = (
            100.0 * row["r_squared"] / total if total > 0 else 0.0
        )
    sens_rows.sort(key=lambda r: r["relative_influence_pct"], reverse=True)

    # Bootstrap-Konfidenzintervall (echtes Resampling) für den Mittelwert
    boot_rng = _random.Random(config.seed + 1)
    boot_means: list[float] = []
    n_boot = min(config.bootstrap_samples, 1000)
    for _ in range(n_boot):
        sample = [rois[boot_rng.randrange(n)] for _ in range(n)]
        boot_means.append(_stats.fmean(sample))
    boot = {
        "mean": _stats.fmean(boot_means),
        "ci_low": _percentile(sorted(boot_means), 2.5),
        "ci_high": _percentile(sorted(boot_means), 97.5),
        "std_error": _stats.stdev(boot_means) if n_boot > 1 else 0.0,
    }

    files: list[Path] = []
    result = SimulationResult(
        config=config,
        samples={k: v for k, v in samples.items()},  # type: ignore[dict-item]
        roi=rois,  # type: ignore[arg-type]
        summary=summary,
        bootstrap=boot,
        sensitivity=sens_rows,
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


def summarize_roi(roi: "np.ndarray", confidence: float) -> dict[str, float]:
    """Berechnet die im Paper berichteten Kennzahlen."""
    _ensure_numpy()
    n = roi.size
    var_alpha = (1.0 - confidence) * 100.0
    var_value = float(np.percentile(roi, var_alpha))
    return {
        "n": int(n),
        "mean": float(roi.mean()),
        "std": float(roi.std(ddof=1)),
        "median": float(np.median(roi)),
        "p05": float(np.percentile(roi, 5)),
        "p25": float(np.percentile(roi, 25)),
        "p75": float(np.percentile(roi, 75)),
        "p95": float(np.percentile(roi, 95)),
        "min": float(roi.min()),
        "max": float(roi.max()),
        "var_5pct": var_value,
        "probability_of_loss": float((roi < 0).mean()),
        "skewness": float(((roi - roi.mean()) ** 3).mean() / roi.std(ddof=1) ** 3),
        "kurtosis": float(((roi - roi.mean()) ** 4).mean() / roi.std(ddof=1) ** 4 - 3.0),
    }


def bootstrap_mean(
    roi: "np.ndarray", samples: int, confidence: float, rng: "np.random.Generator"
) -> dict[str, float]:
    """Bootstrap-Konfidenzintervall für den Erwartungswert."""
    _ensure_numpy()
    means = np.empty(samples, dtype=float)
    n = roi.size
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        means[i] = roi[idx].mean()
    alpha = 1.0 - confidence
    return {
        "mean": float(means.mean()),
        "ci_low": float(np.percentile(means, 100 * alpha / 2)),
        "ci_high": float(np.percentile(means, 100 * (1.0 - alpha / 2))),
        "std_error": float(means.std(ddof=1)),
    }


def sensitivity(
    samples: dict[str, "np.ndarray"], roi: "np.ndarray"
) -> list[dict[str, float]]:
    """Pearson-Korrelation + normiertes R² für jede Inputvariable."""
    _ensure_numpy()
    total = 0.0
    rows: list[dict[str, float]] = []
    for name, values in samples.items():
        r = float(np.corrcoef(values, roi)[0, 1])
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
                "capex": (500.0, 750.0, 1_200.0),
                "opex": (80.0, 120.0, 200.0),
                "volume": (50.0, 150.0, 300.0),
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


def plot_roi_distribution(
    roi: "np.ndarray", summary: dict[str, float], output_dir: Path
) -> Path:
    _ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = 80
    counts, edges, patches = ax.hist(roi, bins=bins, density=True,
                                     color="#4c72b0", alpha=0.75,
                                     edgecolor="white", label="Stichprobe")
    if _ensure_scipy():
        xs = np.linspace(roi.min(), roi.max(), 400)
        kde = _scipy_stats.gaussian_kde(roi)
        ax.plot(xs, kde(xs), color="black", lw=1.5, label="KDE")

    ax.axvline(0, color="#c44e52", lw=1.2, ls="--",
               label=f"P(ROI<0) = {summary['probability_of_loss']:.1%}")
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
        f"P(ROI<0) = {summary['probability_of_loss']:.2%}"
    )
    ax.text(0.98, 0.97, text, transform=ax.transAxes, ha="right", va="top",
            family="monospace", fontsize=10,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="lightgray"))

    ax.set_title("ROI-Histogramm der Monte-Carlo-Simulation")
    ax.set_xlabel("ROI")
    ax.set_ylabel("Dichte")
    ax.legend(loc="upper left", fontsize=9)
    target = output_dir / "roi_distribution.png"
    _save(fig, target)
    return target


def plot_tornado(sensitivity_rows: Sequence[dict[str, float]], output_dir: Path) -> Path:
    _ensure_matplotlib()
    names = [row["variable"].upper() for row in sensitivity_rows]
    values = [row["relative_influence_pct"] for row in sensitivity_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names[::-1], values[::-1], color="#4c72b0", edgecolor="white")
    ax.set_xlabel("Anteil an erklärter ROI-Varianz (%)")
    ax.set_title("Tornado-Diagramm — Sensitivitätsanalyse")
    ax.set_xlim(0, max(values) * 1.15 if values else 1.0)
    for bar, value in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%", va="center", fontsize=10)
    target = output_dir / "sensitivity_tornado.png"
    _save(fig, target)
    return target


# ---------------------------------------------------------------------------
# Orchestrierung
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    config: SimulationConfig
    samples: dict[str, Any]
    roi: Any
    summary: dict[str, float]
    bootstrap: dict[str, float]
    sensitivity: list[dict[str, float]]
    files: list[Path] = field(default_factory=list)

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

        return {
            "config": {
                "iterations": self.config.iterations,
                "seed": self.config.seed,
                "bootstrap_samples": self.config.bootstrap_samples,
                "bootstrap_confidence": self.config.bootstrap_confidence,
                "capex": asdict(self.config.capex),
                "opex": asdict(self.config.opex),
                "volume": asdict(self.config.volume),
                "price": asdict(self.config.price),
                "expected_inputs": {k: _round(v)
                                    for k, v in self.config.expected_metrics().items()},
            },
            "summary": {k: _round(v) for k, v in self.summary.items()},
            "bootstrap_mean": {k: _round(v) for k, v in self.bootstrap.items()},
            "sensitivity": [
                {k: _round(v) for k, v in row.items()}
                for row in self.sensitivity
            ],
            "artifacts": [str(p) for p in self.files],
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
    roi = compute_roi(samples)
    summary = summarize_roi(roi, confidence=config.bootstrap_confidence)
    boot = bootstrap_mean(roi, config.bootstrap_samples, config.bootstrap_confidence, rng)
    sens = sensitivity(samples, roi)

    files: list[Path] = []
    if _has_matplotlib():
        _ensure_matplotlib()
        files.append(plot_inputs(samples, output_dir))
        files.append(plot_roi_distribution(roi, summary, output_dir))
        files.append(plot_tornado(sens, output_dir))
    else:
        print("[WARNUNG] matplotlib fehlt – überspringe Plots.")

    if write_csv:
        csv_path = output_dir / "samples.csv"
        header = ",".join(["capex", "opex", "volume", "price", "roi"])
        stack = np.column_stack([samples["capex"], samples["opex"],
                                 samples["volume"], samples["price"], roi])
        np.savetxt(csv_path, stack, delimiter=",", header=header,
                   comments="", fmt="%.6f")
        files.append(csv_path)

    result = SimulationResult(
        config=config,
        samples=samples,
        roi=roi,
        summary=summary,
        bootstrap=boot,
        sensitivity=sens,
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
    print("\n=== Monte-Carlo-ROI — Ergebnisreport ===")
    cfg = result.config
    print(f"Iterationen : {cfg.iterations:,}  (Seed = {cfg.seed})")
    print(f"\nInput-Erwartungswerte (analytisch):")
    for key, value in cfg.expected_metrics().items():
        print(f"  {key:10s} = {value:10.2f} $M")
    print("\nROI-Kennzahlen (Stichprobe):")
    s = result.summary
    print(f"  Mean     = {_format_pct(s['mean'])}")
    print(f"  Median   = {_format_pct(s['median'])}")
    print(f"  Std      = {_format_pct(s['std'])}")
    print(f"  P05      = {_format_pct(s['p05'])}")
    print(f"  P95      = {_format_pct(s['p95'])}")
    print(f"  VaR 5%   = {_format_pct(s['var_5pct'])}")
    print(f"  P(ROI<0) = {_format_pct(s['probability_of_loss'])}")
    print(f"  Min/Max  = {_format_pct(s['min'])} / {_format_pct(s['max'])}")
    b = result.bootstrap
    print(f"\nBootstrap-{int(cfg.bootstrap_confidence*100)}%-KI für Mean: "
          f"[{b['ci_low']:.2%}, {b['ci_high']:.2%}]  (SE = {b['std_error']:.2%})")
    print("\nSensitivitätsanalyse (relativer Varianzanteil):")
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
    parser.add_argument("--capex-low", type=float, default=500.0)
    parser.add_argument("--capex-mode", type=float, default=750.0)
    parser.add_argument("--capex-high", type=float, default=1_200.0)
    parser.add_argument("--opex-low", type=float, default=80.0)
    parser.add_argument("--opex-mode", type=float, default=120.0)
    parser.add_argument("--opex-high", type=float, default=200.0)
    parser.add_argument("--volume-low", type=float, default=50.0)
    parser.add_argument("--volume-mode", type=float, default=150.0)
    parser.add_argument("--volume-high", type=float, default=300.0)
    parser.add_argument("--price-mean", type=float, default=70.0)
    parser.add_argument("--price-sigma", type=float, default=25.0)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--no-csv", action="store_true",
                        help="Schreibt keine samples.csv-Datei")
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
    )
    result = run(config=config, output_dir=args.output_dir, write_csv=not args.no_csv)
    print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
