#!/usr/bin/env python3
"""
Monte Carlo Simulation for Oil Industry ROI Analysis
Corrected version with proper OPEX time-dimension scaling.
"""
import random, math, statistics, json

random.seed(12345)
N = 10_000

# Parameters
capex_a, capex_b, capex_c = 500, 750, 1200
opex_a, opex_b, opex_c = 80, 120, 200
project_years = 10
vol_a, vol_b, vol_c = 50, 150, 300
oil_mu = math.log(70) - 0.35**2 / 2
oil_sigma = 0.35

def triangular(a, b, c):
    u = random.random()
    if u <= (b - a) / (c - a):
        return a + math.sqrt(u * (c - a) * (b - a))
    else:
        return c - math.sqrt((1 - u) * (c - a) * (c - b))

def lognormal(mu, sigma):
    return math.exp(mu + sigma * random.gauss(0, 1))

results = []
for i in range(N):
    capex = triangular(capex_a, capex_b, capex_c)
    opex_annual = triangular(opex_a, opex_b, opex_c)
    opex_total = opex_annual * project_years
    volume = triangular(vol_a, vol_b, vol_c)
    price = lognormal(oil_mu, oil_sigma)

    revenue = price * volume
    total_cost = capex + opex_total
    profit = revenue - total_cost
    roi = profit / capex
    results.append({
        'capex': capex, 'opex_annual': opex_annual, 'opex_total': opex_total,
        'volume': volume, 'price': price, 'revenue': revenue,
        'total_cost': total_cost, 'profit': profit, 'roi': roi
    })

rois = [r['roi'] for r in results]
mean_roi = statistics.mean(rois)
median_roi = statistics.median(rois)
std_roi = statistics.stdev(rois)
sorted_roi = sorted(rois)
var_95 = sorted_roi[int(N * 0.05)]
prob_loss = sum(1 for r in rois if r < 0) / N
max_roi = max(rois)
min_roi = min(rois)

print("=== Corrected Monte Carlo Simulation Results ===")
print(f"N = {N}, Project life = {project_years} years")
print(f"Mean ROI          = {mean_roi*100:.1f}%")
print(f"Median (P50)      = {median_roi*100:.1f}%")
print(f"Std Dev           = {std_roi*100:.1f}%")
print(f"VaR (5%)          = {var_95*100:.1f}%")
print(f"P(Loss)           = {prob_loss*100:.1f}%")
print(f"Max ROI           = {max_roi*100:.1f}%")
print(f"Min ROI           = {min_roi*100:.1f}%")

# Sensitivity analysis
def pearson_corr(xs, ys):
    n = len(xs)
    mx = statistics.mean(xs); my = statistics.mean(ys)
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
    return num/den if den else 0

r_price = pearson_corr([r['price'] for r in results], rois)
r_volume = pearson_corr([r['volume'] for r in results], rois)
r_capex = pearson_corr([r['capex'] for r in results], rois)
r_opex = pearson_corr([r['opex_total'] for r in results], rois)

r2_price = r_price**2; r2_volume = r_volume**2
r2_capex = r_capex**2; r2_opex = r_opex**2
r2_total = r2_price + r2_volume + r2_capex + r2_opex

print(f"\n=== Sensitivity Analysis ===")
print(f"Price:   influence={r2_price/r2_total*100:.1f}%, r={r_price:.4f}")
print(f"Volume:  influence={r2_volume/r2_total*100:.1f}%, r={r_volume:.4f}")
print(f"CAPEX:   influence={r2_capex/r2_total*100:.1f}%, r={r_capex:.4f}")
print(f"OPEX:    influence={r2_opex/r2_total*100:.1f}%, r={r_opex:.4f}")

with open('/tmp/MonteCarlo/simulation_results.json', 'w') as f:
    json.dump({
        'N': N, 'project_years': project_years,
        'mean_roi': mean_roi, 'median_roi': median_roi,
        'std_roi': std_roi, 'var_95': var_95,
        'prob_loss': prob_loss, 'max_roi': max_roi, 'min_roi': min_roi,
        'influence': {'price': r2_price/r2_total, 'volume': r2_volume/r2_total,
                      'capex': r2_capex/r2_total, 'opex': r2_opex/r2_total}
    }, f, indent=2)
print("\nResults saved.")
