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
