"""Price one option every way qpricer knows how."""

from qpricer import (
    EuropeanOption,
    MarketData,
    OptionType,
    bs_price,
    crr_price,
    implied_vol,
    mc_price,
    mc_price_antithetic_cv,
)

market = MarketData(spot=100.0, rate=0.05, vol=0.2, dividend_yield=0.01)
call = EuropeanOption(strike=105.0, maturity=1.0, option_type=OptionType.CALL)

exact = bs_price(call, market)
tree = crr_price(call, market, steps=1000)
mc = mc_price(call, market, n_paths=1_000_000, seed=42)
mc_vr = mc_price_antithetic_cv(call, market, n_paths=1_000_000, seed=42)

print(f"Black-Scholes:      {exact:.4f}")
print(f"CRR (1000 steps):   {tree:.4f}  (error {tree - exact:+.2e})")
print(f"MC plain:           {mc.price:.4f}  +/- {mc.std_error:.4f} (1 SE)")
print(f"MC antithetic+CV:   {mc_vr.price:.4f}  +/- {mc_vr.std_error:.4f} (1 SE)")

iv = implied_vol(exact, 100.0, 105.0, 1.0, 0.05, OptionType.CALL, dividend_yield=0.01)
print(f"implied vol round-trip: {iv:.6f} (true 0.200000)")
