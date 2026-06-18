from datetime import datetime
import argparse
import os

import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.data_loader import (
    calculate_return_stats,
    fetch_latest_yahoo_quote,
    fetch_yahoo_history,
    load_option_chain,
    select_atm_option,
)
from src.instruments import instrument_from_config
from src.pricing import (
    binomial_tree_price,
    finite_difference_greeks,
    implied_volatility,
    model_price,
    monte_carlo_price,
)
from src.reporting import save_outputs
from src.risk import scenario_analysis
from src.strategy import generate_signal
from src.utils import ensure_dirs, safe_name


def run_framework(config_path: str):
    config = load_config(config_path)

    project_cfg = config["project"]
    market_cfg = config["market"]
    mc_cfg = config["monte_carlo"]
    data_cfg = config["data"]

    data_dir = project_cfg.get("data_dir", "data")
    output_dir = project_cfg.get("output_dir", "outputs")
    plot_dir = project_cfg.get("plot_dir", "outputs/plots")

    ensure_dirs(data_dir, output_dir, plot_dir)

    start_date = market_cfg.get("start_date", "2024-01-01")
    trading_days = int(market_cfg.get("trading_days", 252))
    risk_free_rate = float(market_cfg.get("risk_free_rate", 0.01))
    threshold = float(market_cfg.get("mispricing_threshold", 0.05))

    mc_paths = int(mc_cfg.get("paths", 20000))
    mc_steps = int(mc_cfg.get("steps", 252))
    mc_seed = int(mc_cfg.get("seed", 42))

    option_chain_csv = data_cfg.get("option_chain_csv", "data/option_chain_japan.csv")
    option_chain = load_option_chain(option_chain_csv)

    summary_rows = []
    signal_rows = []
    scenario_tables = {}

    trade_date = pd.Timestamp.now().normalize()

    for instrument_raw in config["instruments"]:
        instrument = instrument_from_config(instrument_raw)

        print(f"\nProcessing: {instrument.name}")

        price_df = fetch_yahoo_history(
            yahoo_symbol=instrument.yahoo,
            start_date=start_date,
        )

        history_path = os.path.join(
            data_dir,
            f"{safe_name(instrument.name)}_historical_prices.csv",
        )

        price_df.to_csv(history_path, index=False)

        stats = calculate_return_stats(
            price_df=price_df,
            trading_days=trading_days,
        )

        try:
            spot_or_future, quote_timestamp = fetch_latest_yahoo_quote(instrument.yahoo)
        except Exception:
            spot_or_future = float(price_df["close"].iloc[-1])
            quote_timestamp = price_df["date"].iloc[-1]

        option_row = select_atm_option(
            option_chain=option_chain,
            instrument=instrument,
            spot_or_future=spot_or_future,
            trade_date=trade_date,
        )

        if option_row is None:
            print(f"No option data found for {instrument.name}. Skipping.")
            continue

        strike = float(option_row["strike"])
        option_type = str(option_row["option_type"])
        market_price = float(option_row["market_price"])

        expiry_dt = pd.to_datetime(
            option_row["expiry"],
            format="%Y%m%d",
            errors="coerce",
        )

        if pd.isna(expiry_dt):
            print(f"Invalid expiry for {instrument.name}. Skipping.")
            continue

        time_to_expiry = max((expiry_dt - trade_date).days / 365.0, 1 / 365.0)

        iv = implied_volatility(
            market_price=market_price,
            S_or_F=spot_or_future,
            K=strike,
            T=time_to_expiry,
            r=risk_free_rate,
            option_type=option_type,
            asset_type=instrument.asset_type,
            q=instrument.dividend_yield,
        )

        if not np.isnan(iv) and iv > 0:
            sigma = iv
        else:
            sigma = stats["historical_vol"]

        fair_value = model_price(
            S_or_F=spot_or_future,
            K=strike,
            T=time_to_expiry,
            r=risk_free_rate,
            sigma=sigma,
            option_type=option_type,
            asset_type=instrument.asset_type,
            q=instrument.dividend_yield,
        )

        binomial = binomial_tree_price(
            S_or_F=spot_or_future,
            K=strike,
            T=time_to_expiry,
            r=risk_free_rate,
            sigma=sigma,
            steps=500,
            option_type=option_type,
            asset_type=instrument.asset_type,
            q=instrument.dividend_yield,
        )

        mc_price, mc_error = monte_carlo_price(
            S_or_F=spot_or_future,
            K=strike,
            T=time_to_expiry,
            r=risk_free_rate,
            sigma=sigma,
            option_type=option_type,
            asset_type=instrument.asset_type,
            q=instrument.dividend_yield,
            paths=mc_paths,
            steps=mc_steps,
            seed=mc_seed,
        )

        delta, gamma, vega, theta, rho = finite_difference_greeks(
            S_or_F=spot_or_future,
            K=strike,
            T=time_to_expiry,
            r=risk_free_rate,
            sigma=sigma,
            option_type=option_type,
            asset_type=instrument.asset_type,
            q=instrument.dividend_yield,
        )

        signal = generate_signal(
            market_price=market_price,
            fair_value=fair_value,
            delta=delta,
            threshold=threshold,
        )

        scenario_df = scenario_analysis(
            S_or_F=spot_or_future,
            K=strike,
            T=time_to_expiry,
            r=risk_free_rate,
            sigma=sigma,
            option_type=option_type,
            asset_type=instrument.asset_type,
            q=instrument.dividend_yield,
        )

        scenario_tables[instrument.name] = scenario_df

        scenario_path = os.path.join(
            data_dir,
            f"{safe_name(instrument.name)}_scenario_analysis.csv",
        )

        scenario_df.to_csv(scenario_path, index=False)

        summary_rows.append(
            {
                "instrument": instrument.name,
                "symbol": instrument.symbol,
                "asset_type": instrument.asset_type,
                "yahoo": instrument.yahoo,
                "tradingview": instrument.tradingview,
                "quote_timestamp": str(quote_timestamp),
                "spot_or_future": round(spot_or_future, 4),
                "expiry": expiry_dt.date(),
                "strike": round(strike, 4),
                "option_type": option_type,
                "market_price": round(market_price, 4),
                "historical_vol": round(stats["historical_vol"], 6),
                "implied_vol": round(iv, 6) if not np.isnan(iv) else np.nan,
                "vol_used": round(sigma, 6),
                "fair_value": round(fair_value, 4),
                "binomial": round(binomial, 4),
                "monte_carlo": round(mc_price, 4),
                "mc_error": round(mc_error, 6),
                "delta": round(delta, 6),
                "gamma": round(gamma, 8),
                "vega": round(vega, 6),
                "theta": round(theta, 6),
                "rho": round(rho, 6),
                "annualized_return": round(stats["annualized_return"], 6),
                "skewness": round(stats["skewness"], 6),
                "kurtosis": round(stats["kurtosis"], 6),
            }
        )

        signal_rows.append(
            {
                "instrument": instrument.name,
                "signal": signal["signal"],
                "action": signal["action"],
                "hedge_action": signal["hedge_action"],
                "edge_pct": round(signal["edge_pct"], 6)
                if not np.isnan(signal["edge_pct"])
                else np.nan,
                "reason": signal["reason"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    signal_df = pd.DataFrame(signal_rows)

    save_outputs(
        output_dir=output_dir,
        plot_dir=plot_dir,
        summary_df=summary_df,
        signal_df=signal_df,
        scenario_tables=scenario_tables,
    )

    print("\nProject completed successfully.")
    print(f"Run time: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Pricing summary saved to: {output_dir}/pricing_summary.csv")
    print(f"Trading signals saved to: {output_dir}/trading_signals.csv")
    print(f"Report saved to: {output_dir}/research_report.md")

    print("\nTrading Signals:")
    if not signal_df.empty:
        print(signal_df.to_string(index=False))
    else:
        print("No signals generated.")


def main():
    parser = argparse.ArgumentParser(
        description="Japan Advanced Derivatives Pricing and Risk Framework"
    )

    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML file.",
    )

    args = parser.parse_args()
    run_framework(args.config)


if __name__ == "__main__":
    main()
