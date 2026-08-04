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

## Roadmap

- [ ] Core instrument and market data types
- [ ] Black–Scholes analytic prices, Greeks and implied volatility
- [ ] Cox–Ross–Rubinstein binomial tree
- [ ] Monte Carlo engine with error estimates
- [ ] Variance reduction: antithetic and control variates
- [ ] Convergence study and error analysis
- [ ] Numba-accelerated kernels and benchmarks
- [ ] Profiling-driven optimization

## Development

```bash
pip install -e ".[dev,perf,analysis]"
pytest
```
