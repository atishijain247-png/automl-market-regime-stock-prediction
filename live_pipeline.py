from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf


# --------------------------------------------------
# 1. LOAD EXISTING PROJECT ARTIFACTS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "automl_stock_model.pkl"
REGIME_PATH = BASE_DIR / "market_regime_artifacts.pkl"

model = joblib.load(MODEL_PATH)
regime_artifacts = joblib.load(REGIME_PATH)

regime_scaler = regime_artifacts["regime_scaler"]
kmeans = regime_artifacts["kmeans"]
regime_names = regime_artifacts["regime_names"]


# Use the exact feature order expected by the trained model.
MODEL_FEATURES = list(model.feature_names_in_)


# --------------------------------------------------
# 2. FETCH APPLE MARKET DATA
# --------------------------------------------------

def fetch_stock_data(ticker="AAPL", period="1y"):

    stock = yf.Ticker(ticker)

    data = stock.history(
        period=period,
        interval="1d",
        auto_adjust=True
    )

    if data.empty:
        raise ValueError(
            f"No market data returned for {ticker}."
        )

    data.index = pd.to_datetime(data.index, utc=True)
    data = data.sort_index()

    # Remove timezone from the index for easier display.
    data.index = data.index.tz_localize(None)

    required_columns = ["Close", "Volume"]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(
                f"Required column missing: {column}"
            )

    data = data[["Close", "Volume"]].copy()

    data["Close"] = pd.to_numeric(
        data["Close"], errors="coerce"
    )

    data["Volume"] = pd.to_numeric(
        data["Volume"], errors="coerce"
    )

    data = data.dropna(subset=["Close", "Volume"])

    if len(data) < 60:
        raise ValueError(
            "Not enough historical data to calculate "
            "the required indicators."
        )

    return data


# --------------------------------------------------
# 3. CALCULATE FINANCIAL FEATURES
# --------------------------------------------------

def calculate_features(data):

    data = data.copy()

    close = data["Close"]
    volume = data["Volume"]

    data["daily_return"] = close.pct_change()

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()

    data["ma20_ratio"] = close / ma20 - 1
    data["ma50_ratio"] = close / ma50 - 1

    data["volatility"] = (
        data["daily_return"].rolling(20).std()
    )

    data["momentum"] = close.pct_change(5)

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    data["rsi"] = 100 - (100 / (1 + rs))

    data["volume_change"] = volume.pct_change()

    volume_ma20 = volume.rolling(20).mean()

    data["volume_ma20_ratio"] = (
        volume / volume_ma20 - 1
    )

    # A zero-loss period can make RSI undefined.
    # Handle common edge cases without changing model inputs.
    data["rsi"] = data["rsi"].fillna(50)

    data = data.replace(
        [np.inf, -np.inf], np.nan
    )

    return data


# --------------------------------------------------
# 4. DETECT THE CURRENT MARKET REGIME
# --------------------------------------------------

def detect_market_regime(data):

    data = data.copy()

    regime_inputs = data[
        ["daily_return", "volatility"]
    ]

    valid_rows = regime_inputs.notna().all(axis=1)

    if not valid_rows.any():
        raise ValueError(
            "Not enough valid data for regime detection."
        )

    scaled_inputs = regime_scaler.transform(
        regime_inputs.loc[valid_rows]
    )

    cluster_ids = kmeans.predict(scaled_inputs)

    data["regime_cluster"] = np.nan

    data.loc[
        valid_rows, "regime_cluster"
    ] = cluster_ids

    data["market_regime"] = (
        data["regime_cluster"]
        .map(regime_names)
    )

    return data


# --------------------------------------------------
# 5. PREPARE THE MODEL INPUTS
# --------------------------------------------------

def prepare_model_features(data):

    data = data.copy()

    for regime in ["Bearish", "Bullish", "Sideways"]:

        column = f"regime_{regime}"

        data[column] = (
            data["market_regime"] == regime
        ).astype(int)

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    X = data[MODEL_FEATURES].copy()

    X = X.replace(
        [np.inf, -np.inf], np.nan
    )

    return X


# --------------------------------------------------
# 6. GENERATE THE LATEST PREDICTION
# --------------------------------------------------

def analyze_stock(ticker="AAPL"):

    data = fetch_stock_data(ticker)

    data = calculate_features(data)

    data = detect_market_regime(data)

    X = prepare_model_features(data)

    # Select the latest row with all required features.
    valid_rows = X.notna().all(axis=1)

    if not valid_rows.any():
        raise ValueError(
            "No complete feature row is available."
        )

    latest_date = X.index[valid_rows][-1]

    latest_features = X.loc[[latest_date]]

    prediction = int(
        model.predict(latest_features)[0]
    )

    probabilities = model.predict_proba(
        latest_features
    )[0]

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "The model does not expose class 1 "
            "for upward movement."
        )

    probability_up = float(
        probabilities[classes.index(1)]
    )

    latest = data.loc[latest_date]

    volatility_limit = regime_artifacts[
        "volatility_limit"
    ]

    volatility = float(latest["volatility"])

    # Simple, transparent signal rules.
    # These are project rules, not investment advice.
    if prediction == 1 and probability_up >= 0.60:
        signal = "BUY"
    elif prediction == 0 and probability_up <= 0.40:
        signal = "AVOID"
    else:
        signal = "HOLD"

    if volatility > volatility_limit:
        risk_level = "High"
    else:
        risk_level = "Normal"

    return {
        "ticker": ticker.upper(),
        "data_date": latest_date.strftime("%Y-%m-%d"),
        "latest_close": float(latest["Close"]),
        "predicted_direction": (
            "Up" if prediction == 1 else "Down"
        ),
        "probability_up": probability_up,
        "market_regime": latest["market_regime"],
        "daily_return": float(latest["daily_return"]),
        "volatility": volatility,
        "risk_level": risk_level,
        "signal": signal,
        "features": latest_features.iloc[0].to_dict()
    }


# --------------------------------------------------
# 7. RUN A TEST WHEN EXECUTED DIRECTLY
# --------------------------------------------------

if __name__ == "__main__":

    result = analyze_stock("AAPL")

    print("\nAPPLE STOCK ANALYSIS")
    print("=" * 40)

    for key, value in result.items():

        if key == "features":
            continue

        print(f"{key}: {value}")