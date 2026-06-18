import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from src.utils import normalize_option_type


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float = 0.0,
) -> float:
    option_type = normalize_option_type(option_type)

    if T <= 0:
        return max(S - K, 0.0) if option_type == "CALL" else max(K - S, 0.0)

    d1 = (
        np.log(S / K)
        + (r - q + 0.5 * sigma ** 2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "CALL":
        return float(
            S * np.exp(-q * T) * norm.cdf(d1)
            - K * np.exp(-r * T) * norm.cdf(d2)
        )

    return float(
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S * np.exp(-q * T) * norm.cdf(-d1)
    )


def black76_price(
    F: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> float:
    option_type = normalize_option_type(option_type)

    if T <= 0:
        return max(F - K, 0.0) if option_type == "CALL" else max(K - F, 0.0)

    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    discount = np.exp(-r * T)

    if option_type == "CALL":
        return float(discount * (F * norm.cdf(d1) - K * norm.cdf(d2)))

    return float(discount * (K * norm.cdf(-d2) - F * norm.cdf(-d1)))


def model_price(
    S_or_F: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    asset_type: str,
    q: float = 0.0,
) -> float:
    if asset_type.lower() == "futures":
        return black76_price(S_or_F, K, T, r, sigma, option_type)

    return black_scholes_price(S_or_F, K, T, r, sigma, option_type, q)


def implied_volatility(
    market_price: float,
    S_or_F: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    asset_type: str,
    q: float = 0.0,
) -> float:
    if market_price <= 0 or T <= 0 or S_or_F <= 0 or K <= 0:
        return np.nan

    try:
        def objective(sigma):
            return model_price(
                S_or_F,
                K,
                T,
                r,
                sigma,
                option_type,
                asset_type,
                q,
            ) - market_price

        return float(brentq(objective, 0.0001, 5.0, maxiter=200))

    except Exception:
        return np.nan


def binomial_tree_price(
    S_or_F: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    steps: int,
    option_type: str,
    asset_type: str,
    q: float = 0.0,
) -> float:
    option_type = normalize_option_type(option_type)

    if T <= 0:
        return max(S_or_F - K, 0.0) if option_type == "CALL" else max(K - S_or_F, 0.0)

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u

    carry = 0.0 if asset_type.lower() == "futures" else r - q
    p = (np.exp(carry * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)

    prices = np.array(
        [S_or_F * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)]
    )

    if option_type == "CALL":
        values = np.maximum(prices - K, 0.0)
    else:
        values = np.maximum(K - prices, 0.0)

    discount = np.exp(-r * dt)

    for i in range(steps - 1, -1, -1):
        values = discount * (
            p * values[1: i + 2]
            + (1.0 - p) * values[0: i + 1]
        )

    return float(values[0])


def monte_carlo_price(
    S_or_F: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    asset_type: str,
    q: float,
    paths: int,
    steps: int,
    seed: int,
):
    option_type = normalize_option_type(option_type)

    if T <= 0:
        value = max(S_or_F - K, 0.0) if option_type == "CALL" else max(K - S_or_F, 0.0)
        return float(value), 0.0

    rng = np.random.default_rng(seed)
    dt = T / steps
    z = rng.standard_normal((paths, steps))

    if asset_type.lower() == "futures":
        drift = -0.5 * sigma ** 2
    else:
        drift = r - q - 0.5 * sigma ** 2

    log_returns = drift * dt + sigma * np.sqrt(dt) * z
    terminal = S_or_F * np.exp(np.cumsum(log_returns, axis=1)[:, -1])

    if option_type == "CALL":
        payoff = np.maximum(terminal - K, 0.0)
    else:
        payoff = np.maximum(K - terminal, 0.0)

    discounted = np.exp(-r * T) * payoff

    return float(np.mean(discounted)), float(np.std(discounted) / np.sqrt(paths))


def finite_difference_greeks(
    S_or_F: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    asset_type: str,
    q: float = 0.0,
):
    hS = max(S_or_F * 0.001, 0.01)
    hsig = 0.01
    hr = 0.0001
    hT = 1 / 365

    p0 = model_price(S_or_F, K, T, r, sigma, option_type, asset_type, q)

    p_up = model_price(S_or_F + hS, K, T, r, sigma, option_type, asset_type, q)
    p_dn = model_price(max(S_or_F - hS, 0.01), K, T, r, sigma, option_type, asset_type, q)

    delta = (p_up - p_dn) / (2 * hS)
    gamma = (p_up - 2 * p0 + p_dn) / (hS ** 2)

    vega = model_price(S_or_F, K, T, r, sigma + hsig, option_type, asset_type, q) - p0

    theta = (
        model_price(
            S_or_F,
            K,
            max(T - hT, 1 / 365),
            r,
            sigma,
            option_type,
            asset_type,
            q,
        )
        - p0
    )

    rho = (
        model_price(S_or_F, K, T, r + hr, sigma, option_type, asset_type, q)
        - p0
    ) / 0.01

    return float(delta), float(gamma), float(vega), float(theta), float(rho)
