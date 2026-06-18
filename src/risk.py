import pandas as pd

from src.pricing import model_price


def scenario_analysis(
    S_or_F: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    asset_type: str,
    q: float,
) -> pd.DataFrame:
    scenarios = [
        ["Low Volatility", 1.00, 0.70, 0.0000],
        ["High Volatility", 1.00, 1.50, 0.0000],
        ["Bull Market", 1.10, 1.00, 0.0000],
        ["Bear Market", 0.90, 1.20, 0.0000],
        ["Market Crash", 0.75, 2.00, -0.0025],
    ]

    rows = []

    for name, spot_mult, vol_mult, rate_shift in scenarios:
        scenario_spot = S_or_F * spot_mult
        scenario_sigma = max(sigma * vol_mult, 0.0001)
        scenario_r = max(r + rate_shift, 0.0)

        rows.append(
            {
                "scenario": name,
                "spot_or_future": scenario_spot,
                "volatility": scenario_sigma,
                "risk_free_rate": scenario_r,
                "option_price": model_price(
                    scenario_spot,
                    K,
                    T,
                    scenario_r,
                    scenario_sigma,
                    option_type,
                    asset_type,
                    q,
                ),
            }
        )

    return pd.DataFrame(rows)
