import numpy as np
import pandas as pd
import gc

expiry_dates = [
    pd.Timestamp('2020-01-17'),
    pd.Timestamp('2020-02-21'),
    pd.Timestamp('2020-03-20'),
    pd.Timestamp('2020-04-09'),
    pd.Timestamp('2020-04-17'),
    pd.Timestamp('2020-05-15'),
    pd.Timestamp('2020-06-19'),
    pd.Timestamp('2020-07-02'),
    pd.Timestamp('2020-07-17'),
    pd.Timestamp('2020-08-21'),
    pd.Timestamp('2020-09-18'),
    pd.Timestamp('2020-10-16'),
    pd.Timestamp('2020-11-20'),
    pd.Timestamp('2020-12-18'),
    pd.Timestamp('2020-12-24'),
    pd.Timestamp('2020-12-31'),
    pd.Timestamp('2021-01-15'),
    pd.Timestamp('2021-02-19'),
    pd.Timestamp('2021-03-19'),
    pd.Timestamp('2021-04-01'),
    pd.Timestamp('2021-04-16'),
    pd.Timestamp('2021-05-21'),
    pd.Timestamp('2021-06-18'),
    pd.Timestamp('2021-07-16'),
    pd.Timestamp('2021-08-20'),
    pd.Timestamp('2021-09-17'),
    pd.Timestamp('2021-10-15'),
    pd.Timestamp('2021-11-19'),
    pd.Timestamp('2021-12-17'),
    pd.Timestamp('2021-12-23'),
    pd.Timestamp('2022-01-21'),
    pd.Timestamp('2022-02-18'),
    pd.Timestamp('2022-03-18'),
    pd.Timestamp('2022-04-14'),
    pd.Timestamp('2022-05-20'),
    pd.Timestamp('2022-06-17'),
    pd.Timestamp('2022-07-15'),
    pd.Timestamp('2022-08-19'),
    pd.Timestamp('2022-09-16'),
    pd.Timestamp('2022-10-21'),
    pd.Timestamp('2022-11-18'),
    pd.Timestamp('2022-12-16'),
    pd.Timestamp('2023-01-20'),
    pd.Timestamp('2023-02-17'),
    pd.Timestamp('2023-03-17'),
    pd.Timestamp('2023-04-06'),
    pd.Timestamp('2023-04-21'),
    pd.Timestamp('2023-05-19'),
    pd.Timestamp('2023-06-16'),
    pd.Timestamp('2023-07-21'),
    pd.Timestamp('2023-08-18'),
    pd.Timestamp('2023-09-15'),
    pd.Timestamp('2023-10-20'),
    pd.Timestamp('2023-11-17'),
    pd.Timestamp('2023-12-15'),
    pd.Timestamp('2024-01-19'),
    pd.Timestamp('2024-02-16'),
    pd.Timestamp('2024-03-15'),
    pd.Timestamp('2024-03-28'),
    pd.Timestamp('2024-04-19'),
    pd.Timestamp('2024-05-17'),
    pd.Timestamp('2024-06-21'),
    pd.Timestamp('2024-07-19'),
    pd.Timestamp('2024-08-16'),
    pd.Timestamp('2024-09-20'),
    pd.Timestamp('2024-10-18'),
    pd.Timestamp('2024-11-15'),
    pd.Timestamp('2024-12-20'),
    pd.Timestamp('2025-01-17'),
    pd.Timestamp('2025-02-21'),
    pd.Timestamp('2025-03-21'),
    pd.Timestamp('2025-04-17'),
    pd.Timestamp('2025-05-16'),
    pd.Timestamp('2025-06-20'),
    pd.Timestamp('2025-07-03'),
    pd.Timestamp('2025-07-18'),
    pd.Timestamp('2025-08-15'),
    pd.Timestamp('2025-09-19'),
    pd.Timestamp('2025-10-17'),
    pd.Timestamp('2025-11-21'),
    pd.Timestamp('2025-12-19')
]

expiry_dates_set = set(expiry_dates) # Convert list to set for efficient lookup

def prepare_daily_dataframe(QQQ: pd.DataFrame) -> pd.DataFrame:
    daily = (
        QQQ.sort_values("tradeDate")
        .groupby("tradeDate", as_index=False)
        .agg(spot=("spotPrice", "first"))
    )
    return daily

def add_nextDayReturn(daily: pd.DataFrame) -> pd.DataFrame:
    daily["ret_1d"]       = daily["spot"].pct_change()
    daily["nextDayRet"]   = daily["ret_1d"].shift(-1)
    return daily

def add_realisedVolFeatures(daily: pd.DataFrame) -> pd.DataFrame:
    daily["vol_5d"]  = daily["ret_1d"].rolling(5).std()  * np.sqrt(252)
    daily["vol_21d"] = daily["ret_1d"].rolling(21).std() * np.sqrt(252)
    daily["vol_63d"] = daily["ret_1d"].rolling(63).std() * np.sqrt(252)
    return daily

def add_calendarEffectFeatures(daily: pd.DataFrame) -> pd.DataFrame:
    daily["is_monday"] = (daily["tradeDate"].dt.weekday == 0).astype(int)
    daily["is_friday"] = (daily["tradeDate"].dt.weekday == 4).astype(int)
    daily["is_option_expiry"] = daily["tradeDate"].isin(expiry_dates_set).astype(int)
    daily["month"] = daily["tradeDate"].dt.month

    # month-end (last trading day in each calendar month)
    last_trading_day_in_month = daily.groupby(daily["tradeDate"].dt.to_period("M"))["tradeDate"].transform("max")
    daily["is_month_end_trading"] = (daily["tradeDate"] == last_trading_day_in_month).astype(int)
    del last_trading_day_in_month

    # quarter-end (last trading day in each calendar quarter)
    last_trading_day_in_quarter = daily.groupby(daily["tradeDate"].dt.to_period("Q"))["tradeDate"].transform("max")
    daily["is_quarter_end_trading"] = (daily["tradeDate"] == last_trading_day_in_quarter).astype(int)
    del last_trading_day_in_quarter

    # year-end (last trading day in calendar year)
    last_trading_day_in_year = daily.groupby(daily["tradeDate"].dt.to_period("Y"))["tradeDate"].transform("max")
    daily["is_year_end_trading"] = (daily["tradeDate"] == last_trading_day_in_year).astype(int)
    del last_trading_day_in_year

    gc.collect()

    return daily





