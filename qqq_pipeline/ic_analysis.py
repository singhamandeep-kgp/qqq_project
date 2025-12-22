import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_z_scores(daily, feature_columns):
    feature_zscore_columns = []
    for col in feature_columns:
        z_col = f"{col}_z"
        daily[z_col] = (daily[col] - daily[col].mean()) / daily[col].std()
        daily[f"{z_col}_chg1"] = daily[z_col].diff()
        feature_zscore_columns.extend([z_col, f"{z_col}_chg1"])
    return daily, feature_zscore_columns

def calculate_ics(daily, feature_zscore_columns):
    ic_rows = []
    n = len(daily)
    for col in feature_zscore_columns:
        previous_length = len(daily)
        df_ = daily[[col, "nextDayRet"]].dropna()
        if previous_length - len(df_) > 0:
            logging.info(f"Dropped {previous_length - len(df_)} rows due to NA values for feature {col}")
        ic, pval = spearmanr(df_[col], df_["nextDayRet"])
        tstat = ic * np.sqrt((n - 2) / (1 - ic**2)) if np.isfinite(ic) and abs(ic) < 1 else np.nan
        ic_rows.append({"feature": col, "IC_spearman": ic, "IC_tstat": tstat, "IC_pvalue": pval})
    return pd.DataFrame(ic_rows).sort_values("IC_spearman", ascending=False)

def plot_decile_curves(daily, top_features):
    fig, axes = plt.subplots(len(top_features), 1, figsize=(10, 3 * len(top_features)), sharex=True)
    axes = axes if len(top_features) > 1 else [axes]
    for ax, feat in zip(axes, top_features):
        df = daily[[feat, "nextDayRet"]].dropna()
        decile_mean = df.assign(decile=pd.qcut(df[feat], 10, labels=False, duplicates="drop")).groupby("decile")["nextDayRet"].mean()
        ax.plot(decile_mean.index, decile_mean.values, marker="o")
        ax.axhline(0, linestyle="--", linewidth=0.8)
        ax.set(title=f"Decile Curve: {feat}", ylabel="Avg nextDayRet")
    axes[-1].set_xlabel("Decile (0 = lowest, 9 = highest)")
    plt.tight_layout()
    plt.show()

def plot_rolling_ics(daily, top_features):
    fig, axes = plt.subplots(len(top_features), 1, figsize=(12, 3 * len(top_features)), sharex=True)
    axes = axes if len(top_features) > 1 else [axes]
    for ax, feat in zip(axes, top_features):
        df = daily[["tradeDate", feat, "nextDayRet"]].dropna()
        rolling_ic = df[feat].rank().rolling(252).corr(df["nextDayRet"].rank())
        ax.plot(pd.to_datetime(df["tradeDate"]), rolling_ic)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set(title=f"Rolling 252-Day Spearman IC: {feat}", ylabel="IC")
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.show()

def calculate_ic_and_plot(daily, feature_columns: list | None = None):
    daily = daily.copy()
    non_feature_cols = ['tradeDate', 'spot', 'expirDate', 'ret_1d', 'nextDayRet', 'dte', 'strike', 'delta', 'gamma', 'vega', 'theta', 'rho',
                         'vol_5d', 'vol_21d', 'vol_63d', 'callValue', 'callBidPrice', 'callAskPrice', 'callOpenInterest', 'callVolume', 'putValue',
                         'putBidPrice', 'putAskPrice', 'putOpenInterest', 'putVolume', 'is_monday', 'is_friday',
                         'is_option_expiry', 'month', 'is_month_end_trading', 'is_quarter_end_trading',
                         'is_year_end_trading', 'ret_1d', 'mid_call', 'mid_put', 'spotPrice']
    
    feature_columns = [col for col in daily.columns if col not in non_feature_cols] if feature_columns is None else feature_columns
    daily = daily[['tradeDate', 'nextDayRet'] + feature_columns]
    
    # Dropping the 1 NA due to nextDayRet
    pre_length = len(daily)
    daily.dropna(inplace=True)
    logging.info(f"Dropped {pre_length - len(daily)} rows due to NA values")

    daily, feature_zscore_columns = calculate_z_scores(daily, feature_columns)
    ic_summary = calculate_ics(daily, feature_zscore_columns)

    print(ic_summary)

    plot_decile_curves(daily, feature_zscore_columns)
    plot_rolling_ics(daily, feature_zscore_columns)