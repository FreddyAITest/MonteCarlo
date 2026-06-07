#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-
# =============================================================================
# Monte-Carlo-ROI-Simulation für ölindustrielle Investitionsprojekte
# -----------------------------------------------------------------------------
# Einfache RStudio-Implementierung des im Working Paper
# "Stochastische ROI-Bewertung mittels Monte-Carlo-Simulation" (Scho &
# Finkenzeller, 2026) beschriebenen Vier-Variablen-Modells.
#
# Verwendet ausschließlich Base-R (stats, graphics) — keine zusätzlichen
# Pakete erforderlich. Die Verteilungen und Kennzahlen sind 1:1 kompatibel
# zur Python-Referenzimplementierung in src/monte_carlo_roi.py.
#
# Aufruf:
#     Rscript mc_simulation.R                       # Standardparameter
#     Rscript mc_simulation.R --iterations 50000    # mehr Iterationen
#     Rscript mc_simulation.R --seed 7              # anderer Zufallssamen
#     Rscript mc_simulation.R --kein-plot           # ohne Histogramm
#
# Ausgaben (im Output-Verzeichnis, default ./output):
#     results_r.csv      Stichprobe (CAPEX, OPEX, Volumen, Preis, ROI)
#     results_r.json     Kennzahlen als JSON (für Vergleich mit Python)
#     roi_histogramm.pdf ROI-Histogramm mit VaR-Linie (Base-R-Graphics)
# =============================================================================

# ---------------------------------------------------------------------------
# 1. Argumente parsen (sehr einfach, nur Long-Optionen)
# ---------------------------------------------------------------------------

parse_args <- function(args) {
  defaults <- list(
    iter    = 10000L,
    seed    = 42L,
    ausgabe = "output",
    plot    = TRUE
  )
  i <- 1L
  while (i <= length(args)) {
    a <- args[[i]]
    if (a == "--iterations" || a == "--n") {
      defaults$iter <- as.integer(args[[i + 1L]]); i <- i + 2L
    } else if (a == "--seed") {
      defaults$seed <- as.integer(args[[i + 1L]]); i <- i + 2L
    } else if (a == "--output-dir" || a == "--ausgabe") {
      defaults$ausgabe <- args[[i + 1L]]; i <- i + 2L
    } else if (a == "--kein-plot" || a == "--no-plot") {
      defaults$plot <- FALSE; i <- i + 1L
    } else {
      i <- i + 1L
    }
  }
  defaults
}

# ---------------------------------------------------------------------------
# 2. Eingabeparameter (Dreiecksverteilung & Lognormalverteilung)
# ---------------------------------------------------------------------------
# Werte in $M (CAPEX, OPEX), M Barrel (Volumen) bzw. $/Barrel (Preis).
# Preise in $/Barrel, Volumen in M Barrel, CAPEX/OPEX in $M.
# Damit ist (Preis * Volumen) bereits in $M und alle Terme in der ROI-Formel
# sind dimensionskonsistent.

# Standardkonfiguration passend zu Abschnitt 4 des Papers.
capex_min   <- 500.0;  capex_modus  <- 750.0;  capex_max  <- 1200.0
opex_min    <- 80.0;   opex_modus   <- 120.0;  opex_max   <- 200.0
volumen_min <- 50.0;   volumen_modus <- 150.0; volumen_max <- 300.0

# Lognormal-Parameter im Log-Raum (siehe Paper §3.2.2 und Python-Referenz).
# Erwartungswert ≈ 70 $/Barrel, Streuung ≈ 25 $/Barrel.
# Hinweis: In der Python-Referenz werden mean/sigma im Dollar-Raum angegeben
# und intern in (mu_ln, sigma_ln) umgerechnet. Hier geben wir die Log-Raum-
# Parameter direkt an — die Werte 4.20 / 0.35 entsprechen der im Paper
# dokumentierten Parametrisierung und liefern E[X] ≈ 71, σ_X ≈ 25.
preis_mu    <- 4.20
preis_sigma <- 0.35

# ---------------------------------------------------------------------------
# 3. Verteilungsfunktionen (Base-R, keine Zusatzpakete)
# ---------------------------------------------------------------------------

# Stichprobe einer Dreiecksverteilung Tri(min, modus, max) der Länge n.
# Base-R bietet keine Dreiecksverteilung — wir leiten sie aus der
# allgemeinen inversen Verteilungsfunktion ab (siehe Wikipedia / Press et al.).
rtriang <- function(n, minimum, modus, maximum) {
  u <- runif(n)
  fc <- (modus - minimum) / (maximum - minimum)  # = c in der F-Verteilung
  # Inversionsmethode: Fallunterscheidung an der Mode-Position.
  links  <- u < fc
  rechts <- !links
  out <- numeric(n)
  out[links]  <- minimum + sqrt(u[links]  * (maximum - minimum) * (modus - minimum))
  out[rechts] <- maximum - sqrt((1.0 - u[rechts]) *
                                (maximum - minimum) * (maximum - modus))
  out
}

# Stichprobe einer Lognormalverteilung im Log-Raum.
# rlnorm(n, meanlog = mu, sdlog = sigma) liefert X = exp(Y) mit
# Y ~ N(mu, sigma^2).  Damit ist E[X] = exp(mu + sigma^2/2) und
# Var[X] = (exp(sigma^2) - 1) * exp(2 mu + sigma^2).
rlogn_raum <- function(n, mu_ln, sigma_ln) {
  rlnorm(n, meanlog = mu_ln, sdlog = sigma_ln)
}

# Analytische Momente der Lognormalvariablen (für Plausibilitätsreport).
lognormal_momente <- function(mu_ln, sigma_ln) {
  mittelwert <- exp(mu_ln + sigma_ln^2 / 2)
  varianz    <- (exp(sigma_ln^2) - 1) * exp(2 * mu_ln + sigma_ln^2)
  list(mittelwert = mittelwert, varianz = varianz)
}

# ---------------------------------------------------------------------------
# 4. Simulation
# ---------------------------------------------------------------------------

run_simulation <- function(n_iter, startwert) {
  set.seed(startwert)

  cat(sprintf("[INFO] Ziehe %d Stichproben aus den vier Verteilungen ...\n", n_iter))

  capex   <- rtriang(n_iter, capex_min,   capex_modus,   capex_max)
  opex    <- rtriang(n_iter, opex_min,    opex_modus,    opex_max)
  volumen <- rtriang(n_iter, volumen_min, volumen_modus, volumen_max)

  preis <- rlogn_raum(n_iter, preis_mu, preis_sigma)

  # ROI = (Preis * Volumen - CAPEX - OPEX) / CAPEX  (alle Terme in $M)
  umsatz   <- preis * volumen
  gewinn   <- umsatz - capex - opex
  roi      <- gewinn / capex

  list(capex = capex, opex = opex, volumen = volumen, preis = preis, roi = roi)
}

# ---------------------------------------------------------------------------
# 5. Kennzahlen
# ---------------------------------------------------------------------------

kennzahlen <- function(roi) {
  n   <- length(roi)
  sor <- sort(roi)
  perzentil <- function(q) {
    pos <- (n - 1) * (q / 100)
    lo  <- floor(pos); hi <- ceiling(pos)
    if (lo == hi) sor[lo + 1L] else sor[lo + 1L] + (sor[hi + 1L] - sor[lo + 1L]) *
                                                       (pos - lo)
  }
  m  <- mean(roi)
  sd_ <- sd(roi)                         # Stichprobenstandardabweichung
  schiefe <- mean(((roi - m) / sd_)^3)
  kurt    <- mean(((roi - m) / sd_)^4) - 3

  list(
    n                    = n,
    mean                 = m,
    median               = median(roi),
    std                  = sd_,
    p05                  = perzentil(5),
    p25                  = perzentil(25),
    p75                  = perzentil(75),
    p95                  = perzentil(95),
    min                  = min(roi),
    max                  = max(roi),
    var_5pct             = perzentil(5),   # Value-at-Risk auf 5 %-Niveau
    probability_of_loss  = mean(roi < 0),
    skewness             = schiefe,
    kurtosis             = kurt
  )
}

# ---------------------------------------------------------------------------
# 6. Plot (Base-R-Graphics — keine Pakete erforderlich)
# ---------------------------------------------------------------------------

plot_roi <- function(roi, kz, ausgabe) {
  pdf_path <- file.path(ausgabe, "roi_histogramm.pdf")
  png_path <- file.path(ausgabe, "roi_histogramm.png")

  # PDF-Hauptdatei
  pdf(pdf_path, width = 10, height = 6)

  # Haupt-Histogramm
  hist(roi, breaks = 80, freq = FALSE, col = "#4c72b0", border = "white",
       main = sprintf("ROI-Histogramm (N = %d)", length(roi)),
       xlab = "ROI", ylab = "Dichte")

  # Vertikale Markierungslinien
  abline(v = 0, col = "#c44e52", lwd = 1.5, lty = 2)
  abline(v = kz$mean,   col = "#2ca02c", lwd = 1.5, lty = 1)
  abline(v = kz$median, col = "#dd8452", lwd = 1.5, lty = 1)
  abline(v = kz$var_5pct, col = "#9467bd", lwd = 1.5, lty = 3)

  # Legende
  legende <- c(
    sprintf("Mean = %.2f%%",   100 * kz$mean),
    sprintf("Median = %.2f%%", 100 * kz$median),
    sprintf("VaR 5%% = %.2f%%", 100 * kz$var_5pct),
    sprintf("P(ROI<0) = %.2f%%", 100 * kz$probability_of_loss)
  )
  legend("topright", legend = legende,
         col    = c("#2ca02c", "#dd8452", "#9467bd", "#c44e52"),
         lwd    = 1.5, lty = c(1, 1, 3, 2), bty = "n", cex = 0.85)

  # Kennzahlen-Box
  text_str <- sprintf(
    paste0("N = %d\nMean = %.2f%%\nMedian = %.2f%%\nStd = %.2f%%\n",
           "VaR 5%% = %.2f%%\nP(ROI<0) = %.2f%%"),
    kz$n, 100 * kz$mean, 100 * kz$median, 100 * kz$std,
    100 * kz$var_5pct, 100 * kz$probability_of_loss
  )
  mtext(text_str, side = 3, line = -2, adj = 0.02, cex = 0.8, font = 1)

  dev.off()

  # PNG-Kopie (best-effort — schlägt fehl, wenn kein PNG-Device verfügbar)
  png_ok <- tryCatch({
    png(png_path, width = 1200, height = 800, res = 150)
    hist(roi, breaks = 80, freq = FALSE, col = "#4c72b0", border = "white",
         main = sprintf("ROI-Histogramm (N = %d)", length(roi)),
         xlab = "ROI", ylab = "Dichte")
    abline(v = 0, col = "#c44e52", lwd = 1.5, lty = 2)
    abline(v = kz$mean,   col = "#2ca02c", lwd = 1.5, lty = 1)
    abline(v = kz$median, col = "#dd8452", lwd = 1.5, lty = 1)
    abline(v = kz$var_5pct, col = "#9467bd", lwd = 1.5, lty = 3)
    dev.off()
    TRUE
  }, error = function(e) {
    if (length(dev.list()) > 0) try(dev.off(), silent = TRUE)
    FALSE
  })

  if (png_ok) c(pdf_path, png_path) else pdf_path
}

# ---------------------------------------------------------------------------
# 7. Ausgabe (CSV + JSON)
# ---------------------------------------------------------------------------

schreibe_csv <- function(stichproben, ausgabe) {
  pfad <- file.path(ausgabe, "results_r.csv")
  df <- data.frame(
    capex   = stichproben$capex,
    opex    = stichproben$opex,
    volumen = stichproben$volumen,
    preis   = stichproben$preis,
    roi     = stichproben$roi
  )
  write.csv(df, pfad, row.names = FALSE)
  pfad
}

# Sehr einfache JSON-Serialisierung per sprintf, damit kein Zusatzpaket
# (jsonlite) nötig ist. Zahlen werden mit 6 Nachkommastellen formatiert.
schreibe_json <- function(kz, cfg, ausgabe) {
  pfad <- file.path(ausgabe, "results_r.json")
  num <- function(x) if (is.na(x)) "null" else sprintf("%.6f", x)
  int <- function(x) as.character(as.integer(x))
  bool <- function(x) if (isTRUE(x)) "true" else "false"

  json <- sprintf(
    paste(
      '{',
      '"config": {',
        '"iterations": %d, "seed": %d,',
        '"capex":  {"low": %.1f, "mode": %.1f, "high": %.1f},',
        '"opex":   {"low": %.1f, "mode": %.1f, "high": %.1f},',
        '"volume": {"low": %.1f, "mode": %.1f, "high": %.1f},',
        '"price":  {"mu_ln": %.4f, "sigma_ln": %.4f}',
      '},',
      '"summary": {',
        '"n": %d, "mean": %s, "median": %s, "std": %s,',
        '"p05": %s, "p25": %s, "p75": %s, "p95": %s,',
        '"min": %s, "max": %s,',
        '"var_5pct": %s, "probability_of_loss": %s,',
        '"skewness": %s, "kurtosis": %s',
      '}}',
      sep = ""
    ),
    cfg$iter, cfg$seed,
    capex_min,   capex_modus,   capex_max,
    opex_min,    opex_modus,    opex_max,
    volumen_min, volumen_modus, volumen_max,
    preis_mu,    preis_sigma,
    kz$n, num(kz$mean), num(kz$median), num(kz$std),
    num(kz$p05), num(kz$p25), num(kz$p75), num(kz$p95),
    num(kz$min), num(kz$max),
    num(kz$var_5pct), num(kz$probability_of_loss),
    num(kz$skewness), num(kz$kurtosis)
  )
  writeLines(json, pfad, useBytes = TRUE)
  pfad
}

# ---------------------------------------------------------------------------
# 8. Konsolen-Report (deutsch)
# ---------------------------------------------------------------------------

drucke_report <- function(kz, cfg, dateien) {
  cat("\n=== Monte-Carlo-ROI — Ergebnisreport (R-Implementierung) ===\n")
  cat(sprintf("Iterationen : %d  (Seed = %d)\n", cfg$iter, cfg$seed))

  cat("\nInput-Erwartungswerte (analytisch):\n")
  cat(sprintf("  E[CAPEX]   = %10.2f $M\n", (capex_min + capex_modus + capex_max) / 3))
  cat(sprintf("  E[OPEX]    = %10.2f $M\n", (opex_min  + opex_modus  + opex_max)  / 3))
  cat(sprintf("  E[Volumen] = %10.2f M Barrel\n",
              (volumen_min + volumen_modus + volumen_max) / 3))
  lp <- lognormal_momente(preis_mu, preis_sigma)
  cat(sprintf("  E[Preis]   = %10.2f $/Barrel  (Var = %.2f)\n",
              lp$mittelwert, lp$varianz))

  cat("\nROI-Kennzahlen (Stichprobe):\n")
  cat(sprintf("  Mean     = %6.2f%%\n", 100 * kz$mean))
  cat(sprintf("  Median   = %6.2f%%\n", 100 * kz$median))
  cat(sprintf("  Std      = %6.2f%%\n", 100 * kz$std))
  cat(sprintf("  P05      = %6.2f%%\n", 100 * kz$p05))
  cat(sprintf("  P95      = %6.2f%%\n", 100 * kz$p95))
  cat(sprintf("  VaR 5%%   = %6.2f%%\n", 100 * kz$var_5pct))
  cat(sprintf("  P(ROI<0) = %6.2f%%\n", 100 * kz$probability_of_loss))
  cat(sprintf("  Min/Max  = %6.2f%% / %6.2f%%\n",
              100 * kz$min, 100 * kz$max))

  cat("\nArtefakte:\n")
  for (d in dateien) cat(sprintf("  - %s\n", d))
  cat("\n")
}

# ---------------------------------------------------------------------------
# 9. Hauptprogramm
# ---------------------------------------------------------------------------

main <- function() {
  args    <- commandArgs(trailingOnly = TRUE)
  cfg     <- parse_args(args)
  ausgabe <- cfg$ausgabe
  dir.create(ausgabe, showWarnings = FALSE, recursive = TRUE)

  cat(sprintf("Monte-Carlo-ROI-Simulation (R) — %d Iterationen, Seed = %d\n",
              cfg$iter, cfg$seed))
  cat(sprintf("Ausgabe-Verzeichnis: %s\n\n", ausgabe))

  stichproben <- run_simulation(cfg$iter, cfg$seed)
  kz          <- kennzahlen(stichproben$roi)

  dateien <- c()
  dateien <- c(dateien, schreibe_csv(stichproben, ausgabe))
  dateien <- c(dateien, schreibe_json(kz, cfg, ausgabe))
  if (isTRUE(cfg$plot)) {
    dateien <- c(dateien, plot_roi(stichproben$roi, kz, ausgabe))
  }

  drucke_report(kz, cfg, dateien)
  invisible(0)
}

if (!exists("PAPERCLIP_IMPORT_TEST")) main()
