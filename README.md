# qpricer

A Python library for pricing European vanilla options with multiple numerical
methods, quantifying their accuracy against the Black–Scholes closed form, and
benchmarking their performance.

## Goals

The library answers four questions with reproducible numbers:

1. How close is Monte Carlo to Black–Scholes?
2. How many simulations are needed for 1% pricing error?
3. How much faster is Numba than plain NumPy?
4. How effective are variance reduction techniques?

## Results

### How close is Monte Carlo to Black–Scholes?

For an ATM 1y call (S=100, K=100, r=5%, σ=20%, BS price 10.4506), plain MC with
a fixed seed lands within its reported confidence interval of the closed form.
The observed RMSE across 30 independent runs matches the estimator's own
reported standard error and decays as O(N^-1/2):

| paths | RMSE | mean reported SE |
|---|---|---|
| 1,000 | 0.4449 | 0.4617 |
| 16,000 | 0.1223 | 0.1165 |
| 256,000 | 0.0298 | 0.0291 |
| 1,024,000 | 0.0126 | 0.0146 |

![MC convergence](reports/convergence.png)

### How many simulations for 1% error?

**≈ 20,000 paths** bring the standard error of the ATM call under 1% of its
price (pilot-run estimate: 19,751; see `qpricer.mc.convergence.paths_for_relative_error`).
The quadratic cost of accuracy: 0.5% error needs 4× that, 0.1% needs 100×.

Reproduce with `python scripts/run_convergence.py`.

### How much faster is Numba than plain NumPy?

The numba kernels fuse exp, payoff and accumulation into one allocation-free
pass over pre-drawn normals (median of 7 runs, JIT warm-up excluded, WSL2):

| paths | NumPy | numba | numba speedup | numba_parallel speedup |
|---|---|---|---|---|
| 100,000 | 2.2 ms | 1.3 ms | 1.7× | 1.5× |
| 1,000,000 | 28.8 ms | 13.3 ms | 2.2× | 2.5× |
| 10,000,000 | 386.5 ms | 164.0 ms | 2.4× | 2.8× |

All backends share the NumPy random draws, which cost roughly a third of the
NumPy runtime — an Amdahl's-law ceiling on the achievable speedup. Parallelism
only pays above ~1M paths; below that, thread startup dominates.

Reproduce with `python benchmarks/bench_mc.py`.

### How effective are variance reduction techniques?

Variance reduction factor vs plain MC at equal path count (400k paths, 1y
calls; a factor of R means plain MC needs R× more paths for the same error):

| strike | antithetic | control variate | antithetic + CV |
|---|---|---|---|
| 70 (deep ITM) | 17.3× | 406× | 942× |
| 100 (ATM) | 2.0× | 6.8× | 28× |
| 130 (deep OTM) | 1.1× | 1.7× | 13× |

![Variance reduction](reports/variance_reduction.png)

Both techniques exploit the payoff's near-linearity in S_T, so they shine ITM
(the terminal-spot control absorbs almost all variance) and fade OTM, where
the payoff is dominated by its nonlinear kink. Note the factors are themselves
noisy estimates — single-seed values for the combined technique wobble in the
OTM tail.

Reproduce with `python scripts/run_variance_reduction.py`.

## Roadmap

- [x] Core instrument and market data types
- [x] Black–Scholes analytic prices, Greeks and implied volatility
- [x] Cox–Ross–Rubinstein binomial tree
- [x] Monte Carlo engine with error estimates
- [x] Variance reduction: antithetic and control variates
- [x] Convergence study and error analysis
- [ ] Numba-accelerated kernels and benchmarks
- [ ] Profiling-driven optimization

## Development

```bash
pip install -e ".[dev,perf,analysis]"
pytest
```
