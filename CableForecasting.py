#!/usr/bin/env python3
"""
Models: N-BEATS and NHiTS, with Covariates: DXY (inverted), EUR/USD
Install: pip install numpy==2.4.2 pandas matplotlib yfinance darts==0.41.0 pytorch-lightning properscoring torch
"""

###############################################
############ PACKAGES INSTALLATION ############
###############################################

AUTO_INSTALL = False
# Set to True to auto-install any missing packages on the first run.

import importlib.util, subprocess, sys

_REQUIREMENTS = [
    ("numpy==2.4.2",      "numpy"),             # pinned numpy version required by darts
    ("pandas",            "pandas"),            # dataframe handling
    ("matplotlib",        "matplotlib"),        # plotting
    ("yfinance",          "yfinance"),          # market data download
    ("darts==0.41.0",     "darts"),             # time series forecasting library
    ("pytorch-lightning", "pytorch_lightning"), # training backend for darts torch models
    ("properscoring",     "properscoring"),     # CRPS scoring metric
    ("torch",             "torch"),             # deep learning framework
]
# Check the required list of packages (formatted as pip install name, importable module name) are available.

import importlib
import importlib.util
import subprocess 
import sys
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
try:
    import torch
    from darts import TimeSeries
    from darts.utils.missing_values import fill_missing_values
    from darts.dataprocessing.transformers import Scaler
    from darts.models import NHiTSModel, NBEATSModel
    from darts.utils.likelihood_models import QuantileRegression
    from darts.metrics import rmse
    import properscoring as ps 
    DARTS_AVAILABLE = True 
except Exception:
    DARTS_AVAILABLE = False 

_missing = [pkg for pkg, mod in _REQUIREMENTS if not importlib.util.find_spec(mod)]
if _missing:
    if AUTO_INSTALL:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade"] + _missing)
        # Install all missing packages in one pip call
    else:
        # Hygiene Check - If any packages are missing, produce a clear message listing what needs to be run manually.
        sys.exit(
            "Missing packages: " + ", ".join(_missing) +
            "\nRun:  pip install " + " ".join(_missing) +
            "\n  or set AUTO_INSTALL = True at the top of this script."
        )
# Build a list of packages whose import names cannot be found in the environment.

###############################################
######### LOAD CABLE DAILY CLOSE DATA #########
###############################################

print("Downloading GBP/USD data from yfinance...")
raw = yf.download("GBPUSD=X", start="2000-01-01", auto_adjust=True, progress=False)
# Download full available history for GBP/USD with dividend/split corrections.

df = raw[["Close"]].copy()
df.index = pd.to_datetime(df.index)
df.index.name = "Date" 
df.columns = ["Close"] 
# Clean the data by only using daily close data (the more stable price choice which I will use for the modelling).


full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
df = df.reindex(full_index)
df["Close"] = df["Close"].interpolate(method="linear")
df.index.name = "Date"
# Replace missing calendar days with NaN which will then be filled by linear interpolation. 

print(f"  Date range: {df.index.min().date()} → {df.index.max().date()}")
print(f"  Total days (after reindex): {len(df)}")
print(f"  Null values remaining: {df['Close'].isna().sum()}")
# Print a summary of downloaded data for transparency.

###############################################
########### COVARIATES EXPLORATION  ###########
###############################################

print("\nDownloading covariate data for exploratory plots...")
raw_dxy = yf.download("DX-Y.NYB", start="2000-01-01", auto_adjust=True, progress=False)
dxy = raw_dxy[["Close"]].copy()
dxy.index = pd.to_datetime(dxy.index)
dxy = dxy.reindex(full_index).interpolate(method="linear") 
# Download, clean and align DXY data to the continuous daily calendar.

raw_eur = yf.download("EURUSD=X", start="2000-01-01", auto_adjust=True, progress=False)
eur = raw_eur[["Close"]].copy()
eur.index = pd.to_datetime(eur.index)
eur = eur.reindex(full_index).interpolate(method="linear")
# Download, clean and align EUR/USD data to the continuous daily calendar.

fig, (ax_dxy, ax_eur) = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("Covariates", fontsize=14, fontweight="bold")
# Set up a stacked 2-row figure. A shared overall title of "Covariates".


ax_dxy.plot(df.index, df["Close"], color="black", linewidth=0.8, label="GBP/USD Close")  # GBP/USD in black
ax_dxy.set_ylabel("GBP/USD", color="black")
ax_dxy.tick_params(axis="y", labelcolor="black")
ax2_dxy = ax_dxy.twinx()                                       # right axis for DXY
ax2_dxy.plot(dxy.index, dxy["Close"], color="steelblue", linewidth=0.8, alpha=0.8, label="DXY (USD Index)")
ax2_dxy.set_ylabel("DXY (USD Index)", color="steelblue")
ax2_dxy.tick_params(axis="y", labelcolor="steelblue")
lines1, labels1 = ax_dxy.get_legend_handles_labels()           # merge both axes into one legend
lines2, labels2 = ax2_dxy.get_legend_handles_labels()
ax_dxy.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
ax_dxy.set_title("GBP/USD vs DXY (Inverse Correlation)", fontweight="bold")
ax_dxy.grid(True, alpha=0.3)
# Top Panel - Dual axes plot of GBP/USD and DXY.

ax_eur.plot(df.index, df["Close"], color="black", linewidth=0.8, label="GBP/USD Close")  # GBP/USD in black
ax_eur.set_ylabel("GBP/USD", color="black")
ax_eur.tick_params(axis="y", labelcolor="black")
ax2_eur = ax_eur.twinx()                                       # right axis for EUR/USD
ax2_eur.plot(eur.index, eur["Close"], color="darkorange", linewidth=0.8, alpha=0.8, label="EUR/USD")
ax2_eur.set_ylabel("EUR/USD", color="darkorange")
ax2_eur.tick_params(axis="y", labelcolor="darkorange")
lines3, labels3 = ax_eur.get_legend_handles_labels()           # merge both axes into one legend
lines4, labels4 = ax2_eur.get_legend_handles_labels()
ax_eur.legend(lines3 + lines4, labels3 + labels4, loc="upper left")
ax_eur.set_title("GBP/USD vs EUR/USD", fontweight="bold")
ax_eur.set_xlabel("Date")
ax_eur.grid(True, alpha=0.3)
# Bottom Panel - Dual axes plot of GBP/USD and EUR/USD.

plt.tight_layout()
plt.show()                                                     
# Plot and present the two covariate panels.
df_fullhistory = df.copy()  # snapshot of the full price series for the top panel

###############################################
######### MODELLING CABLE (6 MONTHS)  #########
###############################################

if not DARTS_AVAILABLE:
    print("\nNote: Darts or required ML packages are not installed.")
    print(f"  Run:  {sys.executable} -m pip install numpy==2.4.2 pandas matplotlib yfinance darts==0.41.0 pytorch-lightning properscoring torch")
    # Hygiene Check - Clearly state if the required packages are not available.
else:
    HORIZON             = 183   
    # Number of days to forecast, around 6 months.
    OUTPUT_CHUNK_LENGTH = 183   
    # Output window (= horizon).
    INPUT_CHUNK_LENGTH  = 365   
    # Input window (for model training).
    NUM_SAMPLES         = 500   
    # Number of stochastic samples drawn for the probabilistic forecast.

    ts = TimeSeries.from_dataframe(df.reset_index(), time_col="Date", value_cols="Close", freq="D")
    ts = fill_missing_values(ts)
    # Convert the pandas DataFrame to a Darts TimeSeries object at daily frequency

    train_start  = pd.Timestamp("2020-01-01")   
    # Training period start date.
    split_date   = pd.Timestamp("2024-01-01")   
    # Split Date - Training/Testing partition.
    train_target = ts.drop_before(train_start).drop_after(split_date)  
    # Training Slice - 2020 – 2023, drop the data after.
    test_target  = ts.drop_before(split_date)                          
    # Testing Slice: 2024 onwards, drop the data before.

    print("Downloading covariates: DXY and EUR/USD...")
    cov_tickers = {"DX-Y.NYB": "DXY", "EURUSD=X": "EURUSD"}
    df_covs = pd.DataFrame(index=full_index)
    # Load and prepare covariates: DXY and EUR/USD.
  
    for ticker, col in cov_tickers.items():
        raw_c = yf.download(ticker, start="2000-01-01", auto_adjust=True, progress=False)
        s = raw_c["Close"].copy()
        s.index = pd.to_datetime(s.index)
        s = s.reindex(full_index).interpolate(method="linear")
        df_covs[col] = s.values

    df_covs["DXY"] = -df_covs["DXY"]
    # Invert DXY such that DXY change aligns with GBP/USD change.

    ts_covs = TimeSeries.from_dataframe(
        df_covs.reset_index().rename(columns={"index": "Date"}),
        time_col="Date", value_cols=["DXY", "EURUSD"], freq="D"
    )
    ts_covs = fill_missing_values(ts_covs)
    # Build a Darts TimeSeries for all the covariates and slice them to the training/testing windows.

    scaler_covs    = Scaler()
    ts_covs_scaled = scaler_covs.fit_transform(ts_covs).astype(np.float32)
    scaler       = Scaler()
    train_scaled = scaler.fit_transform(train_target).astype(np.float32)
    n_eval   = min(HORIZON, len(test_target))
    df_train = pd.DataFrame({"Actual": train_target.values().flatten()}, index=train_target.time_index)
    df_test  = pd.DataFrame({"Actual": test_target[:n_eval].values().flatten()}, index=test_target[:n_eval].time_index)
    # Scale Covariates - (0,1) scale which neural network deals with better.

    def run_model(model, label):
        print(f"Training {label} model...")
        model.fit(series=train_scaled, past_covariates=ts_covs_scaled, epochs=50, verbose=False)  # train with past covariates
        pred_s   = model.predict(n=HORIZON, series=train_scaled, past_covariates=ts_covs_scaled, num_samples=NUM_SAMPLES, verbose=False)
        pred_out = scaler.inverse_transform(pred_s)                        # back to GBP/USD scale
        QUANTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        df_fc = pd.DataFrame(
            {f"q{q:.2f}": pred_out.quantile(q).values().flatten() for q in QUANTILES},
            index=pred_out.time_index                                      # index aligns with forecast dates
        )
        connector = pd.DataFrame(                                          # single-row bridge from last training price
            {col: df_train["Actual"].iloc[-1] for col in df_fc.columns},
            index=[df_train.index[-1]]
        )
        return df_fc, pd.concat([connector, df_fc]), pred_out 
    # Helper Function - Trains the model with past covariates and return the forecasted DataFrames + raw pred object.
  
    nbeats_model = NBEATSModel(
        input_chunk_length=INPUT_CHUNK_LENGTH,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        num_stacks=30,
        num_blocks=1, 
        num_layers=4,
        layer_widths=256,
        dropout=0.1,
        likelihood=QuantileRegression(),
        random_state=42,
        model_name="NBEATS_GBPUSD",
        force_reset=True,
        save_checkpoints=False,
        pl_trainer_kwargs={"accelerator": "mps"},
    )
    df_forecast_nbeats, df_forecast_plot_nbeats, pred_nbeats = run_model(nbeats_model, "N-BEATS")
    # Define and run the NBEATS model.

    nhits_model = NHiTSModel(
        input_chunk_length=INPUT_CHUNK_LENGTH,
        output_chunk_length=OUTPUT_CHUNK_LENGTH,
        num_stacks=3,
        num_blocks=1,
        num_layers=2,
        layer_widths=512,
        dropout=0.1,
        likelihood=QuantileRegression(),
        random_state=42,
        model_name="NHiTS_GBPUSD",
        force_reset=True,
        save_checkpoints=False,
        pl_trainer_kwargs={"accelerator": "mps"},
    )
    df_forecast_nhits, df_forecast_plot_nhits, pred_nhits = run_model(nhits_model, "NHiTS")
    # Define and run the NHiTS model.
  
    def coverage(df_fc, actuals, lo_col, hi_col):
        return (
            (actuals >= df_fc[lo_col].values[:n_eval]) &
            (actuals <= df_fc[hi_col].values[:n_eval])
        ).mean()
    # Helper Function - Compute the % of observed data (actuals) inside the 50% and 90% CIs.
  
    def plot_forecast(ax, df_fc, df_fc_plot, title, color):
        ax.plot(df_train.index, df_train["Actual"], color="black", linewidth=1, label="Training History")
        ax.plot(df_test.index,  df_test["Actual"],  color="black", linewidth=1, label="Actual (Test)")
        ax.plot(df_fc_plot.index, df_fc_plot["q0.50"], color=color, linewidth=1.5, label="Median Forecast")
        ax.fill_between(df_fc_plot.index, df_fc_plot["q0.05"], df_fc_plot["q0.95"], alpha=0.2, color=color, label="90% CI")
        ax.fill_between(df_fc_plot.index, df_fc_plot["q0.25"], df_fc_plot["q0.75"], alpha=0.35, color=color, label="50% CI") 
        ax.axvline(split_date, color="grey", linestyle="--", linewidth=1, label="Train/Test split") 
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("GBP/USD", fontweight="bold") 
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        #  Helper Function - Draw the forecast panel onto an axis.
        actuals_arr = df_test["Actual"].values
        cov_90  = coverage(df_fc, actuals_arr, "q0.05", "q0.95")
        cov_50  = coverage(df_fc, actuals_arr, "q0.25", "q0.75")
        stats_text = f"$\\bf{{Coverage}}$\n90% CI:  {cov_90:.0%}\n50% CI:  {cov_50:.0%}"
        ax.text(
            0.98, 0.04, stats_text,   
            transform=ax.transAxes,
            fontsize=9, verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="grey")
        )
        # Compute the CI actuals coverage and display as a stats box in the bottom right corner
   
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(20, 8), gridspec_kw={"height_ratios": [8, 8]}) 
    # Combined 2 Panel Figure - NBEATS (top) and NHiTS (bottom).
    plot_forecast(ax_top, df_forecast_nbeats, df_forecast_plot_nbeats, "N-BEATS Probabilistic Forecast — GBP/USD\nTrain: 2020–2023 | Test: 6-month Horizon from Jan 2024", color="blue")
    plot_forecast(ax_bot, df_forecast_nhits, df_forecast_plot_nhits, "NHiTS Probabilistic Forecast — GBP/USD\nTrain: 2020–2023 | Test: 6-month Horizon from Jan 2024", color="green")
    ax_top.set_xlabel("") 
    ax_bot.set_xlabel("Date", fontweight="bold") 
    plt.tight_layout() 
    plt.show() 
    # Plot the final graphs.

###############################################
########### EVALUATION AND ANALYSIS ###########
###############################################

    def wis_from_quantiles(actual, q_dict):
        if np.isnan(actual):
            return np.nan  
        qs     = sorted(q_dict.keys()) 
        K      = len(qs) // 2 
        median = q_dict.get(0.5, q_dict[qs[len(qs) // 2]]) 
        wis    = 0.5 * np.abs(actual - median)
        for k in range(K):
            alpha = 2 * qs[k] 
            L, U  = q_dict[qs[k]], q_dict[qs[-(k + 1)]] 
            score = U - L 
            if actual < L: score += (2 / alpha) * (L - actual)
            if actual > U: score += (2 / alpha) * (actual - U) 
            wis += score
        return wis / (K + 0.5) 
    # Calculated WIS (Weighted Interval Score): Metric for sharpness and coverage penalties across quantile intervals

    def print_metrics(label, pred, df_fc):
        print(f"\n{'='*40}")
        print(f"  {label} Evaluation Metrics")
        print(f"{'='*40}")
        pred_median = pred.quantile(0.5)
        print(f"RMSE:                  {rmse(test_target[:n_eval], pred_median[:n_eval]):.6f}")
        samples           = pred.all_values()
        forecast_ensemble = np.moveaxis(samples, -1, 0)
        obs               = test_target[:n_eval].values().flatten()
        crps_vals         = [ps.crps_ensemble(obs[t], forecast_ensemble[:, t, 0]) for t in range(n_eval)]
        print(f"CRPS (mean):           {np.mean(crps_vals):.6f}")
        quantiles_wis = [0.05, 0.25, 0.5, 0.75, 0.95]
        obs_vals      = test_target[:n_eval].values().flatten()
        wis_scores    = [
            wis_from_quantiles(obs_vals[t], {q: float(pred.quantile(q).values()[t, 0]) for q in quantiles_wis})
            for t in range(n_eval)
        ]
        print(f"WIS (mean):            {np.nanmean(wis_scores):.6f}")
        ci_90_width = df_fc["q0.95"] - df_fc["q0.05"]
        ci_50_width = df_fc["q0.75"] - df_fc["q0.25"]
        print(f"Mean 90% CI width:     {ci_90_width.mean():.6f}")
        print(f"Mean 50% CI width:     {ci_50_width.mean():.6f}")

        actuals   = test_target[:n_eval].values().flatten()
        inside_90 = ((actuals >= df_fc["q0.05"].values[:n_eval]) & (actuals <= df_fc["q0.95"].values[:n_eval])).mean()
        inside_50 = ((actuals >= df_fc["q0.25"].values[:n_eval]) & (actuals <= df_fc["q0.75"].values[:n_eval])).mean()
        print(f"Actuals in 90% CI:     {inside_90:.0%}")  
        print(f"Actuals in 50% CI:     {inside_50:.0%}")
    print_metrics("N-BEATS", pred_nbeats, df_forecast_nbeats) 
    print_metrics("NHiTS", pred_nhits, df_forecast_nhits)
    # Print WIS, RMSE and CRPS metrics for both models and print them.
