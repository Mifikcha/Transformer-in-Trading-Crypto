import ccxt
import pandas as pd
import numpy as np
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import unquote


def _time_to_next_funding(ts):
    h = ts.hour
    if h < 8:
        next_h = 8
    elif h < 16:
        next_h = 16
    else:
        next_h = 24
    next_ts = ts.replace(hour=next_h % 24, minute=0, second=0, microsecond=0)
    if next_h == 24:
        next_ts += pd.Timedelta(days=1)
    return (next_ts - ts).total_seconds() / 60


def _build_http_session() -> requests.Session:
    """Session with retry policy for transient HTTP/TLS failures."""
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods={"GET"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _normalize_cursor(cursor: str | None) -> str | None:
    """Decode percent-encoded cursor once to avoid double-encoding on next request."""
    if not cursor:
        return None
    if "%" in cursor:
        decoded = unquote(cursor)
        return decoded if decoded else cursor
    return cursor


def run(
    input_path: str,
    output_path: str,
    symbol: str = "BTCUSDT",
    category: str = "linear",
) -> None:
    """Add funding rate, open interest, delta OI, basis_diff."""
    exchange = ccxt.bybit({"enableRateLimit": True})
    # ccxt symbol e.g. BTCUSDT -> BTC/USDT:USDT
    ccxt_symbol = f"{symbol[:3]}/{symbol[3:]}:{symbol[3:]}"

    print("=== Добавление funding rate, OI, ΔOI, basis динамики ===")
    df = pd.read_parquet(input_path)
    print(f"Загружено строк: {len(df):,}")
    print(f"Период: {df['ts'].min()} → {df['ts'].max()}")

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    min_ts_ms = int(df["ts"].min().timestamp() * 1000)
    max_ts_ms = int(df["ts"].max().timestamp() * 1000)

    print("\nЗагрузка funding rate history...")
    funding_raw = []
    since = min_ts_ms
    while True:
        try:
            data = exchange.fetch_funding_rate_history(
                ccxt_symbol, since=since, limit=200, params={"category": category}
            )
            if not data:
                break
            funding_raw.extend(data)
            since = data[-1]["timestamp"] + 1
            print(f"Funding загружено {len(funding_raw)} записей...")
            time.sleep(1.2)
        except Exception as e:
            print(f"Funding ошибка: {e}")
            break

    if funding_raw:
        df_funding = pd.DataFrame(funding_raw)[["timestamp", "fundingRate"]]
        df_funding["timestamp"] = pd.to_datetime(df_funding["timestamp"], unit="ms", utc=True)
        df_funding = df_funding.sort_values("timestamp").reset_index(drop=True)
    else:
        df_funding = pd.DataFrame(columns=["timestamp", "fundingRate"])
        print("Funding: данных нет")

    df = pd.merge_asof(
        df, df_funding, left_on="ts", right_on="timestamp", direction="backward"
    ).drop(columns=["timestamp"], errors="ignore")
    df["fundingRate"] = df["fundingRate"].ffill()
    df["funding_missing"] = df["fundingRate"].isna().astype(int)
    df["time_to_funding_min"] = df["ts"].apply(_time_to_next_funding)

    print("\nЗагрузка open interest history (Bybit V5 API)...")
    session = _build_http_session()
    oi_list = []
    cursor = None
    consecutive_errors = 0
    max_consecutive_errors = 12
    page_no = 0
    while True:
        params = {"category": category, "symbol": symbol, "intervalTime": "5min", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        url = "https://api.bybit.com/v5/market/open-interest"
        try:
            resp = session.get(url, params=params, timeout=20).json()
            if resp.get("retCode") != 0:
                msg = str(resp.get("retMsg", "Неизвестная ошибка"))
                # Retry on temporary API-side throttling/availability issues.
                if any(x in msg.lower() for x in ("too many", "limit", "timeout", "busy", "system")):
                    consecutive_errors += 1
                    sleep_s = min(30.0, 0.7 * (2 ** min(consecutive_errors, 6)))
                    print(f"OI временная ошибка API: {msg} | retry через {sleep_s:.1f}s (#{consecutive_errors})")
                    if consecutive_errors > max_consecutive_errors:
                        print("OI: превышен лимит временных ошибок, остановка загрузки.")
                        break
                    time.sleep(sleep_s)
                    continue
                print(f"OI ошибка: {msg}")
                break
            data = resp.get("result", {}).get("list", [])
            if not data:
                break
            consecutive_errors = 0
            oi_list.extend(data)
            page_no += 1
            # Stop once we've covered dataset start. This prevents unnecessary deep history pulls.
            oldest_ts_ms = min(int(row["timestamp"]) for row in data if "timestamp" in row)
            newest_ts_ms = max(int(row["timestamp"]) for row in data if "timestamp" in row)
            cursor = _normalize_cursor(resp.get("result", {}).get("nextPageCursor"))
            if not cursor:
                break
            print(
                f"OI загружено {len(oi_list)} записей | страница={page_no} | "
                f"диапазон={pd.to_datetime(oldest_ts_ms, unit='ms', utc=True)} -> "
                f"{pd.to_datetime(newest_ts_ms, unit='ms', utc=True)}"
            )
            if oldest_ts_ms <= min_ts_ms:
                print("OI: достигнута нижняя граница периода датасета, остановка пагинации.")
                break
            time.sleep(0.7)
        except Exception as e:
            consecutive_errors += 1
            sleep_s = min(30.0, 0.7 * (2 ** min(consecutive_errors, 6)))
            print(f"OI запрос ошибка: {e} | retry через {sleep_s:.1f}s (#{consecutive_errors})")
            if consecutive_errors > max_consecutive_errors:
                print("OI: превышен лимит сетевых ошибок, остановка загрузки.")
                break
            time.sleep(sleep_s)
            continue

    if oi_list:
        df_oi = pd.DataFrame(oi_list)[["timestamp", "openInterest"]]
        df_oi["timestamp"] = pd.to_datetime(df_oi["timestamp"].astype(int), unit="ms", utc=True)
        df_oi["openInterest"] = df_oi["openInterest"].astype(float)
        # Keep one value per timestamp and clip to dataset time range.
        df_oi = (
            df_oi.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .loc[lambda x: (x["timestamp"] >= pd.to_datetime(min_ts_ms, unit="ms", utc=True)) & (x["timestamp"] <= pd.to_datetime(max_ts_ms, unit="ms", utc=True))]
            .reset_index(drop=True)
        )
    else:
        df_oi = pd.DataFrame(columns=["timestamp", "openInterest"])
        print("OI: данных нет")

    # Force OI to the exact OHLCV timestamp grid (no lookahead): same number of rows as source df.
    ts_grid = pd.DataFrame({"ts": df["ts"]})
    if not df_oi.empty:
        df_oi_grid = pd.merge_asof(
            ts_grid,
            df_oi,
            left_on="ts",
            right_on="timestamp",
            direction="backward",
        ).drop(columns=["timestamp"], errors="ignore")
    else:
        df_oi_grid = ts_grid.copy()
        df_oi_grid["openInterest"] = np.nan

    df = pd.merge_asof(
        df, df_oi_grid, on="ts", direction="backward"
    )
    df["openInterest"] = df["openInterest"].ffill()
    df["oi_missing"] = df["openInterest"].isna().astype(int)
    df["delta_oi"] = df["openInterest"].diff().fillna(0)
    print(
        f"OI итог: rows_in_dataset={len(df):,}, rows_raw_oi={len(df_oi):,}, "
        f"non_null_oi_after_merge={int(df['openInterest'].notna().sum()):,}"
    )

    if "basis" in df.columns:
        df["basis_diff"] = df["basis"].diff().fillna(0)
    else:
        print("ВНИМАНИЕ: колонка 'basis' отсутствует — basis_diff не добавлен")

    deriv_cols = [
        "fundingRate", "time_to_funding_min", "openInterest", "delta_oi", "basis_diff",
        "funding_missing", "oi_missing",
    ]
    print("\nПроверка новых признаков (последние 5 строк):")
    print(df[deriv_cols].tail(5))
    print("\nПропуски в новых признаках:")
    print(df[deriv_cols].isna().sum())
    print("\nПример fundingRate (не NaN):")
    print(df[df["fundingRate"].notna()][["ts", "fundingRate"]].tail(3))
    print("\nПример OI (не NaN):")
    print(df[df["openInterest"].notna()][["ts", "openInterest", "delta_oi"]].tail(3))

    df.to_parquet(output_path, index=False)
    print(f"\nСохранено: {output_path}")
    print("Деривативные фичи добавлены. Готово к таргетам!")


if __name__ == "__main__":
    INPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_with_volume_stats.parquet"
    OUTPUT_PATH = "get_data/output/_main/intermediate/btcusdt_5m_with_derivatives.parquet"
    run(INPUT_PATH, OUTPUT_PATH, symbol="BTCUSDT", category="linear")
