# Japan Live Derivatives Pricing and Risk Framework

This project is a quantitative derivatives pricing, risk, and strategy-signal framework for Japanese instruments.

## Instruments

- SoftBank Group Corp. — `9984.T`
- Tokyo Electron Ltd. — `8035.T`
- Toyota Motor Corp. — `7203.T`
- Nikkei 225 USD Futures proxy — `NKD=F`

## Models Used

- Black-Scholes Model
- Black-76 Model for futures options
- Binomial Tree Model
- Monte Carlo Pricing Model
- Implied Volatility
- Greeks
- Scenario Analysis
- Fair-value mispricing signal

## Strategy

The strategy compares real option market price with model fair value.

If model fair value is higher than market price:

```text
BUY_OPTION_DELTA_HEDGE
