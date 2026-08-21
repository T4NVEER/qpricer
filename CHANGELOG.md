# Changelog

## 0.1.0 — 2026-07-12

First complete release.

### Pricing
- Black–Scholes closed form: prices, delta/gamma/vega/theta/rho, and the
  deterministic T=0 / vol=0 limits shared by all methods.
- Implied volatility: Newton with Brenner–Subrahmanyam seed, bisection
  fallback, no-arbitrage bounds validation.
- Cox–Ross–Rubinstein binomial tree: explicit-loop reference and vectorized
  backward induction (verified equal to 1e-12).
- Monte Carlo engine: exact lognormal terminal step, seeded RNG, standard
  error + confidence intervals, memory-bounded batching, backend selection.

### Variance reduction
- Antithetic variates, terminal-spot control variate, and their combination;
  `variance_ratios` comparison utility.

### Performance
- Optional numba kernels (serial + parallel prange) fusing payoff
  accumulation; ~2–2.5× over NumPy at 10M paths.
- Profiling-driven NumPy optimization: in-place ufuncs and BLAS dot removed
  temporary allocations (10M paths: 386 → 315 ms).

### Analysis
- Convergence study (RMSE vs N, O(N^-1/2) verified) and
  `paths_for_relative_error` (~20k paths for 1% on the ATM call).
- Committed reports: convergence, variance reduction, backend benchmarks,
  profiling notes.
