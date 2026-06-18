import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_outputs(
    output_dir: str,
    plot_dir: str,
    summary_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    scenario_tables: dict,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "pricing_summary.csv")
    signals_path = os.path.join(output_dir, "trading_signals.csv")
    report_path = os.path.join(output_dir, "research_report.md")

    summary_df.to_csv(summary_path, index=False)
    signal_df.to_csv(signals_path, index=False)

    plot_pricing_comparison(plot_dir, summary_df)
    generate_report(report_path, summary_df, signal_df, scenario_tables)


def plot_pricing_comparison(plot_dir: str, summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return

    x = np.arange(len(summary_df))
    width = 0.25

    plt.figure(figsize=(12, 6))

    plt.bar(x - width, summary_df["fair_value"], width, label="Fair Value")
    plt.bar(x, summary_df["binomial"], width, label="Binomial")
    plt.bar(x + width, summary_df["monte_carlo"], width, label="Monte Carlo")

    plt.xticks(x, summary_df["instrument"], rotation=25, ha="right")
    plt.ylabel("Option Price")
    plt.title("Japan Derivatives Pricing Comparison")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "pricing_comparison.png"))
    plt.close()


def generate_report(
    report_path: str,
    summary_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    scenario_tables: dict,
) -> None:
    lines = []

    lines.append("# Japan Advanced Derivatives Pricing and Risk Modeling Framework")
    lines.append("")
    lines.append("## Strategy Used")
    lines.append("")
    lines.append(
        "The strategy compares real option market price with model fair value. "
        "If the option is cheaper than fair value, the model gives BUY_OPTION_DELTA_HEDGE. "
        "If the option is expensive compared with fair value, the model gives SELL_OPTION_DELTA_HEDGE. "
        "Otherwise, the model gives HOLD_NO_EDGE."
    )

    lines.append("")
    lines.append("## Pricing Summary")
    lines.append("")
    lines.append(summary_df.to_markdown(index=False))

    lines.append("")
    lines.append("## Trading Signals")
    lines.append("")
    lines.append(signal_df.to_markdown(index=False))

    lines.append("")
    lines.append("## Scenario Analysis")
    lines.append("")

    for name, df in scenario_tables.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(df.to_markdown(index=False))
        lines.append("")

    lines.append("## Important Notes")
    lines.append("")
    lines.append("- This Colab version generates live-data signals using latest available Yahoo Finance data.")
    lines.append("- Real broker order execution should be tested separately with IBKR paper trading.")
    lines.append("- Option-chain prices in CSV must be replaced with real market data before final use.")
    lines.append("- This model does not guarantee profit.")

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
