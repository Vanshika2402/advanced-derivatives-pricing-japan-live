from dataclasses import dataclass


@dataclass
class Instrument:
    id: str
    name: str
    symbol: str
    asset_type: str
    yahoo: str
    tradingview: str
    option_type: str
    dividend_yield: float


def instrument_from_config(row: dict) -> Instrument:
    return Instrument(
        id=str(row["id"]),
        name=str(row["name"]),
        symbol=str(row["symbol"]),
        asset_type=str(row["asset_type"]).lower(),
        yahoo=str(row["yahoo"]),
        tradingview=str(row.get("tradingview", "")),
        option_type=str(row.get("option_type", "CALL")),
        dividend_yield=float(row.get("dividend_yield", 0.0)),
    )
