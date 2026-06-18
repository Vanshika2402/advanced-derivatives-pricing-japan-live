import numpy as np


def generate_signal(
    market_price: float,
    fair_value: float,
    delta: float,
    threshold: float,
) -> dict:
    if market_price is None or np.isnan(market_price) or market_price <= 0:
        return {
            "edge_pct": np.nan,
            "signal": "NO_MARKET_PRICE",
            "action": "NONE",
            "hedge_action": "NONE",
            "reason": "Real option market price unavailable.",
        }

    edge_pct = (fair_value - market_price) / market_price

    if edge_pct > threshold:
        return {
            "edge_pct": float(edge_pct),
            "signal": "BUY_OPTION_DELTA_HEDGE",
            "action": "BUY",
            "hedge_action": "SELL",
            "reason": "Option is cheaper than model fair value.",
        }

    if edge_pct < -threshold:
        return {
            "edge_pct": float(edge_pct),
            "signal": "SELL_OPTION_DELTA_HEDGE",
            "action": "SELL",
            "hedge_action": "BUY",
            "reason": "Option is expensive compared with model fair value.",
        }

    return {
        "edge_pct": float(edge_pct),
        "signal": "HOLD_NO_EDGE",
        "action": "NONE",
        "hedge_action": "NONE",
        "reason": "No strong mispricing edge.",
    }
