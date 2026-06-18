"""
Optional IBKR broker module.

Use this only on your own machine/server where IBKR TWS or IB Gateway is running.
Google Colab usually cannot connect to your local IBKR TWS/Gateway.

This file is included for future live execution upgrade.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class IBKRSettings:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 24
    account: str = ""
    trade_mode: str = "PAPER"


class IBKRBroker:
    def __init__(self, settings: IBKRSettings):
        try:
            from ib_async import IB
        except ImportError as exc:
            raise ImportError("Install ib_async first: pip install ib_async") from exc

        self.settings = settings
        self.ib = IB()

    def connect(self):
        self.ib.connect(
            self.settings.host,
            self.settings.port,
            clientId=self.settings.client_id,
        )

        if not self.ib.isConnected():
            raise ConnectionError("Could not connect to IBKR.")

        print("Connected to IBKR.")

    def disconnect(self):
        if self.ib.isConnected():
            self.ib.disconnect()
            print("Disconnected from IBKR.")

    def confirm_live_permission(self):
        if self.settings.trade_mode.upper() == "LIVE":
            if os.getenv("CONFIRM_LIVE_TRADING") != "YES":
                raise PermissionError(
                    "LIVE trading blocked. Set CONFIRM_LIVE_TRADING=YES only after approval."
                )
