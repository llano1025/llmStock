"""
Financial Modeling Prep (FMP) API Data Loader
Fetches earnings calendar and related financial data from FMP API
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_next_week_date_range() -> Tuple[str, str]:
    """
    Calculate the next Monday-Friday date range based on current day.

    Logic:
    - If today is Mon-Thu: Return rest of current week (today to Friday)
    - If today is Fri-Sun: Return next full week (next Monday to Friday)

    Returns:
        Tuple of (start_date, end_date) in 'YYYY-MM-DD' format
    """
    today = datetime.now()
    current_weekday = today.weekday()  # 0=Monday, 6=Sunday

    # If Friday (4), Saturday (5), or Sunday (6), get next week
    if current_weekday >= 4:
        # Calculate days until next Monday
        days_until_monday = (7 - current_weekday) if current_weekday != 6 else 1
        next_monday = today + timedelta(days=days_until_monday)
        start_date = next_monday
        end_date = next_monday + timedelta(days=4)  # Friday
        logger.info(f"Weekend/Friday detected. Analyzing next week: {start_date.date()} to {end_date.date()}")
    else:
        # Monday through Thursday - analyze rest of current week
        start_date = today
        days_until_friday = 4 - current_weekday  # Days until Friday
        end_date = today + timedelta(days=days_until_friday)
        logger.info(f"Weekday detected. Analyzing current week: {start_date.date()} to {end_date.date()}")

    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def fetch_earnings_calendar(start_date: str, end_date: str, api_key: str) -> Optional[pd.DataFrame]:
    """
    Fetch earnings calendar data from FMP API within the specified date range.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        api_key: FMP API key

    Returns:
        DataFrame with earnings calendar data or None if error occurs
        Columns: symbol, date, epsEstimated, revenueEstimated, epsActual (if available),
                 revenueActual (if available), lastUpdated
    """
    url = f"https://financialmodelingprep.com/stable/earnings-calendar?apikey={api_key}"
    params = {
        'from': start_date,
        'to': end_date
    }

    try:
        logger.info(f"Fetching earnings calendar from FMP API: {start_date} to {end_date}")
        response = requests.get(url, params=params, timeout=30)

        if response.status_code == 200:
            data = response.json()

            if not data:
                logger.warning("FMP API returned empty earnings calendar")
                return None

            earnings_df = pd.DataFrame(data)
            logger.info(f"Successfully fetched {len(earnings_df)} earnings records")
            return earnings_df
        else:
            logger.error(f"Error fetching earnings data from FMP API: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception when fetching FMP earnings calendar: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching FMP earnings calendar: {e}")
        return None


def get_upcoming_week_earnings(api_key: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Get earnings calendar for the upcoming Monday-Friday week.

    Args:
        api_key: FMP API key (if None, will try to get from environment)

    Returns:
        DataFrame with earnings calendar data or None if error occurs
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv('FMP_API_KEY')

    if not api_key or api_key == 'your_api_key_here':
        logger.error("FMP_API_KEY not found or invalid. Please set it in .env file")
        return None

    # Get date range
    start_date, end_date = get_next_week_date_range()

    # Fetch earnings calendar
    earnings_df = fetch_earnings_calendar(start_date, end_date, api_key)

    return earnings_df


def filter_earnings_by_symbols(earnings_df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    """
    Filter earnings calendar to only include specified stock symbols.

    Args:
        earnings_df: DataFrame with earnings calendar data
        symbols: List of stock symbols to filter for

    Returns:
        Filtered DataFrame
    """
    if earnings_df is None or earnings_df.empty:
        logger.warning("Empty earnings DataFrame provided for filtering")
        return pd.DataFrame()

    # Convert symbols to uppercase for matching
    symbols_upper = [s.upper() for s in symbols]

    # Filter DataFrame
    filtered_df = earnings_df[earnings_df['symbol'].str.upper().isin(symbols_upper)].copy()

    logger.info(f"Filtered earnings calendar: {len(filtered_df)} matches from {len(symbols)} requested symbols")

    return filtered_df


def get_earnings_for_active_stocks(api_key: Optional[str] = None, top_n: int = 20) -> pd.DataFrame:
    """
    Get top N stocks by trading volume from the upcoming earnings calendar.

    Args:
        api_key: FMP API key (optional)
        top_n: Number of top stocks to return (default: 20)

    Returns:
        DataFrame with earnings data + stock metrics (volume, price, market cap) for top N stocks by volume
    """
    # Get upcoming earnings
    earnings_df = get_upcoming_week_earnings(api_key)

    if earnings_df is None or earnings_df.empty:
        logger.warning("No earnings data available")
        return pd.DataFrame()

    # Get unique symbols from earnings calendar
    symbols = earnings_df['symbol'].unique().tolist()
    logger.info(f"Fetching stock data for {len(symbols)} symbols from earnings calendar...")

    # Fetch stock metrics (volume, price, market cap) for all symbols
    stock_metrics = []

    import yfinance as yf

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)

            # Get current quote data
            info = ticker.info
            hist = ticker.history(period='5d')  # Get last 5 days for volume average

            if hist.empty:
                logger.debug(f"No historical data for {symbol}, skipping")
                continue

            # Calculate metrics
            current_price = hist['Close'].iloc[-1] if not hist.empty else None
            avg_volume = hist['Volume'].mean() if not hist.empty else 0
            market_cap = info.get('marketCap', 0)

            if avg_volume > 0:  # Only include stocks with trading volume
                stock_metrics.append({
                    'Symbol': symbol.upper(),
                    'Price': current_price,
                    'Volume': avg_volume,
                    'Market_Cap': market_cap,
                    'Name': info.get('longName', symbol)
                })

        except Exception as e:
            logger.debug(f"Error fetching data for {symbol}: {e}")
            continue

    if not stock_metrics:
        logger.warning("Could not fetch stock metrics for any symbols")
        return pd.DataFrame()

    # Create DataFrame from stock metrics
    metrics_df = pd.DataFrame(stock_metrics)

    # Sort by volume (descending) and take top N
    metrics_df = metrics_df.sort_values('Volume', ascending=False).head(top_n)

    logger.info(f"Selected top {len(metrics_df)} stocks by volume from earnings calendar")

    # Merge with earnings data
    earnings_df['Symbol'] = earnings_df['symbol'].str.upper()

    merged_df = earnings_df.merge(
        metrics_df[['Symbol', 'Name', 'Price', 'Volume', 'Market_Cap']],
        on='Symbol',
        how='inner'
    )

    # Calculate days until earnings
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['days_until_earnings'] = (merged_df['date'] - datetime.now()).dt.days

    # Filter to only upcoming earnings (where actual results aren't available yet)
    # Keep stocks where epsActual is null/NaN (earnings haven't occurred yet)
    initial_count = len(merged_df)
    if 'epsActual' in merged_df.columns:
        merged_df = merged_df[merged_df['epsActual'].isna()].copy()
        filtered_count = initial_count - len(merged_df)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} stocks with past earnings (already reported)")

    # Sort by volume (highest first) to prioritize most liquid stocks
    merged_df = merged_df.sort_values('Volume', ascending=False)

    logger.info(f"Found {len(merged_df)} high-volume stocks with upcoming earnings")

    # Log top 5 for visibility
    if not merged_df.empty:
        logger.info("Top 5 stocks by volume:")
        for idx, row in merged_df.head(5).iterrows():
            logger.info(f"  {row['Symbol']}: Volume={row['Volume']:,.0f}, Earnings={row['date'].date()}")

    return merged_df


def test_fmp_connection(api_key: Optional[str] = None) -> bool:
    """
    Test FMP API connection and validity of API key.

    Args:
        api_key: FMP API key (if None, will try to get from environment)

    Returns:
        True if connection successful, False otherwise
    """
    if api_key is None:
        api_key = os.getenv('FMP_API_KEY')

    if not api_key or api_key == 'your_api_key_here':
        logger.error("FMP_API_KEY not found or invalid")
        return False

    # Test with a simple API call (get Apple's profile)
    url = f"https://financialmodelingprep.com/api/v3/profile/AAPL"
    params = {'apikey': api_key}

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0:
                logger.info("FMP API connection successful")
                return True
            else:
                logger.error("FMP API returned unexpected response")
                return False
        else:
            logger.error(f"FMP API connection failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error testing FMP connection: {e}")
        return False


if __name__ == "__main__":
    # Test the module
    print("Testing FMP Data Loader...")

    # Test date range calculation
    start, end = get_next_week_date_range()
    print(f"\nNext week date range: {start} to {end}")

    # Test API connection
    print("\nTesting FMP API connection...")
    if test_fmp_connection():
        print("✓ FMP API connection successful")

        # Test earnings calendar fetch
        print("\nFetching upcoming earnings...")
        earnings = get_upcoming_week_earnings()
        if earnings is not None and not earnings.empty:
            print(f"✓ Found {len(earnings)} earnings announcements")
            print("\nSample earnings data:")
            # Show epsActual if available (for past earnings)
            columns_to_show = ['symbol', 'date', 'epsEstimated']
            if 'epsActual' in earnings.columns:
                columns_to_show.append('epsActual')
            print(earnings[columns_to_show].head())
        else:
            print("✗ No earnings data retrieved")
    else:
        print("✗ FMP API connection failed. Please check your API key in .env file")
