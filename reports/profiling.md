# Profiling notes: NumPy MC path (10M paths)

`python benchmarks/profile_mc.py`, WSL2, Python 3.12.

## Baseline breakdown

| stage | time | share |
|---|---|---|
| RNG draw (`standard_normal`) | 110 ms | 24% |
| exp/drift (`spot * exp(drift + diff*z)`) | 127 ms | 28% |
| payoff + reductions | 218 ms | 48% |

cProfile attributes 334 ms of 463 ms to `sample_terminal_spots`. The stage
timing shows only ~110 ms of that is the irreducible RNG draw; the rest is
elementwise work that allocates three 80 MB temporaries (`diff*z`,
`drift + ...`, `exp(...)`) before the final multiply.

## Actions

1. Rewrite `sample_terminal_spots` with in-place ufuncs on the `z` buffer
   (zero extra allocations).
2. Replace `(discounted * discounted).sum()` with `np.dot(discounted,
   discounted)` in the engine (drops one temporary, uses BLAS).
3. Apply the discount factor in place on the payoff array.

Measured effect: see "after" row in the README performance section; the
`bench_mc.py` numpy column is the before/after reference.

## Ceiling

The RNG draw is shared by all backends and is not vectorizable further from
Python; at 10M paths it bounds any payoff-side optimization to ~4x overall
(Amdahl). The numba kernels attack the same elementwise stages by fusion
instead of in-place NumPy, which is why their advantage shrinks after this
optimization.
