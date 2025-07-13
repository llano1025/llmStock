import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import pytz
from datetime import datetime, timedelta, date
import pandas_ta as ta
import yahooquery as yq
from fake_useragent import UserAgent
from utils.utils_path import get_project_root


def get_stock_list(stock_name, stock_sector, stock_industry, my_holdings, flg_100_stock, flg_my_holdings,
                   flg_batch_search):
    """
    Get a list of stocks to analyze based on provided filters.
    
    Args:
        stock_name: Single stock symbol to analyze if not in batch mode
        stock_sector: Industry sector to filter stocks
        stock_industry: Specific industry to filter stocks
        my_holdings: List of personal stock holdings
        flg_100_stock: Boolean indicating if top 100 stocks should be used
        flg_my_holdings: Boolean indicating if personal holdings should be used
        flg_batch_search: Boolean indicating if sector/industry filtering should be used
        
    Returns:
        list: List of stock symbols to analyze
    """
    # Load the NASDAQ screener data
    project_root = get_project_root()
    historical_data_path = project_root / 'data' / 'nasdaq_screener.csv'
    nasdaq = pd.read_csv(str(historical_data_path))
    
    # Get top 100 stocks by market cap
    if flg_100_stock == 1:
        nasdaq = nasdaq[['Symbol', 'Market Cap']]
        nasdaq = nasdaq[nasdaq['Market Cap'].notnull()]
        nasdaq.sort_values(by='Market Cap', ascending=False, inplace=True)
        stocks = nasdaq['Symbol'].head(100).tolist()
    
    # Use personal holdings
    elif flg_my_holdings == 1:
        stocks = my_holdings
    
    # Filter by sector/industry
    elif flg_batch_search == 1:
        if len(stock_industry) == 0:
            nasdaq = nasdaq.loc[(nasdaq['Sector'] == stock_sector)]
            stocks = nasdaq['Symbol'].tolist()
        else:
            nasdaq = nasdaq.loc[(nasdaq['Sector'] == stock_sector) & (nasdaq['Industry'] == stock_industry)]
            stocks = nasdaq['Symbol'].tolist()
    
    # Use single stock
    else:
        stocks = [stock_name]

    return stocks


def get_stock_data(stock_symbol, start_date, end_date):
    """
    Fetch and process stock data with technical indicators.
    
    Args:
        stock_symbol: Stock ticker symbol
        start_date: Starting date for historical data
        end_date: Ending date for historical data
        
    Returns:
        tuple: (DataFrame with stock data, skip flag)
    """
    # Get the earliest available date for the stock
    hist_max = yf.download(stock_symbol, period='max',auto_adjust=True)
    columns_stripped = [col[0] for col in hist_max.columns]
    hist_max.columns = columns_stripped
    max_date = hist_max.index[0]
    max_date = pd.Timestamp(max_date)
    max_date = max_date.to_pydatetime()
    data_window = end_date - start_date
    flg_skip = 0

    # Download data based on date conditions
    if max_date < start_date:
        # Case 1: Requested start date is after stock's first available date
        hist = yf.download(stock_symbol, start=start_date, end=end_date, auto_adjust=True)
        columns_stripped = [col[0] if isinstance(col, (list, tuple)) else col for col in hist.columns]
        hist.columns = columns_stripped
        hist_ta = yf.download(stock_symbol, start=start_date - timedelta(days=1), end=end_date - timedelta(days=1), auto_adjust=True)
        columns_stripped = [col[0] if isinstance(col, (list, tuple)) else col for col in hist_ta.columns]
        hist_ta.columns = columns_stripped
        
        # Get market index data
        sp500_data = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True)
        sp500_data = sp500_data.rename(columns={'Close': 'S&P500 Close'})
        nasdaq_data = yf.download("^IXIC", start=start_date, end=end_date, auto_adjust=True)
        nasdaq_data = nasdaq_data.rename(columns={'Close': 'NASDAQ Close'})
        hist = fix_yfinance_columns(hist)
        hist = pd.concat([hist, sp500_data['S&P500 Close'], nasdaq_data['NASDAQ Close']], axis=1)
    
    elif start_date < max_date < end_date - timedelta(days=data_window * 1.5):
        # Case 2: Start date is before stock's first available date
        hist = yf.download(stock_symbol, start=max_date + timedelta(days=1), end=end_date)
        columns_stripped = [col[0] if isinstance(col, (list, tuple)) else col for col in hist.columns]
        hist.columns = columns_stripped
        hist_ta = yf.download(stock_symbol, start=max_date, end=end_date)
        columns_stripped = [col[0] if isinstance(col, (list, tuple)) else col for col in hist_ta.columns]
        hist_ta.columns = columns_stripped
        
        # Get market index data
        sp500_data = yf.download("^GSPC", start=max_date, end=end_date)
        sp500_data = sp500_data.rename(columns={'Close': 'S&P500 Close'})
        nasdaq_data = yf.download("^IXIC", start=max_date, end=end_date)
        nasdaq_data = nasdaq_data.rename(columns={'Close': 'NASDAQ Close'})
        hist = fix_yfinance_columns(hist)
        hist = pd.concat([hist, sp500_data['S&P500 Close'], nasdaq_data['NASDAQ Close']], axis=1)
    
    else:
        # Case 3: Skip data download
        flg_skip = 1
        return None, flg_skip

    # Process data and calculate basic technical indicators
    hist.insert(0, 'Prev Close', hist['Close'].shift(1))
    hist['EMA10'] = ta.ema(hist['Close'], length=10)
    hist['SMA20'] = ta.sma(hist['Close'], length=20)
    hist['RSI14'] = ta.rsi(hist_ta['Close'], length=14)

    # Try to add ETF holdings data if available
    try:
        etf_holdings = get_etf_holdings(stock_symbol)
        for holding in etf_holdings:
            holding_data = yf.download(holding, start=start_date, end=end_date)
            hist[f'{holding} Close'] = holding_data['Close']
    except Exception as error_message:
        print(f"Not an ETF or error getting holdings: {error_message}")
        # Add additional technical indicators if not an ETF
        a = ta.adx(hist_ta['High'], hist_ta['Low'], hist_ta['Close'], length=14)
        hist = hist.join(a)
        b = ta.aroon(hist_ta['High'], hist_ta['Low'], length=14)
        b = b.rename(columns={'AROONOSC_14': 'AROSC14'})
        hist = hist.join(b['AROSC14'])

    # Clean up missing values
    if hist.iloc[-1].isnull().any():
        hist = hist.iloc[(hist.isna().sum()).max()-1:]
        hist = hist.drop(hist.index[-1])
    else:
        hist = hist.iloc[(hist.isna().sum()).max():]

    return hist, flg_skip

def fix_yfinance_columns(df):
    """
    Fix column order for yFinance dataframe using column swapping to ensure standard order:
    ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Check if all expected columns exist
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"Missing one or more required columns. Expected: {expected_cols}")
    
    # Get current column order
    current_cols = list(df.columns)
    
    # If columns are not in expected order, swap them
    if current_cols != expected_cols:
        for i, expected_col in enumerate(expected_cols):
            current_idx = current_cols.index(expected_col)
            if current_idx != i:
                # Swap columns
                current_cols[i], current_cols[current_idx] = current_cols[current_idx], current_cols[i]
                # Apply swap to DataFrame
                df = df.reindex(columns=current_cols)
    
    return df

def get_etf_holdings(stock_name):
    """
    Get top holdings for an ETF.
    
    Args:
        stock_name: ETF ticker symbol
        
    Returns:
        list: List of stock symbols that are top holdings of the ETF
    """
    holdings = yq.Ticker(stock_name).fund_top_holdings
    holdings = holdings['symbol'].tolist()
    return holdings


def get_earning_data(stock_name):
    """
    Get earnings data for a stock from Yahoo Finance.
    
    Args:
        stock_name: Stock ticker symbol
        
    Returns:
        tuple: (DataFrame with earnings data, next earnings date, days until earnings)
    """
    # Send a GET request to the page and get the HTML content
    ua = UserAgent(browsers=['edge', 'chrome', 'firefox'])
    user_agent = ua.random
    headers = {'user-agent': user_agent}

    # Set the URL of the earnings page for the stock you're interested in
    url = f'https://finance.yahoo.com/calendar/earnings/?symbol={stock_name}'
    response = requests.get(url, headers=headers)
    html_content = response.content

    # Use BeautifulSoup to parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract the text values of the earnings date and result elements
    earnings_date = []
    earnings_EPS_estimate = []
    earnings_EPS_reported = []

    # Validate if the elements with 'aria-label' exist
    date_tags = soup.find_all('td', {'aria-label': 'Earnings Date'})
    eps_estimate_tags = soup.find_all('td', {'aria-label': 'EPS Estimate'})
    eps_reported_tags = soup.find_all('td', {'aria-label': 'Reported EPS'})

    for tag in date_tags:
        earnings_date.append(tag.text)
    for tag in eps_estimate_tags:
        earnings_EPS_estimate.append(tag.text)
    for tag in eps_reported_tags:
        earnings_EPS_reported.append(tag.text)

    try:
        # Prepare the dataframe for time alignment
        df = pd.DataFrame({
            'Earnings_date': earnings_date, 
            'EPS_estimate': earnings_EPS_estimate,
            'EPS_reported': earnings_EPS_reported
        })
        df.set_index('Earnings_date', inplace=True)
        df.index = df.index.astype(str).str.split(',').str[:2].str.join(',')
        df.index = pd.to_datetime(df.index, format='%b %d, %Y')
        df.index = df.index.strftime('%d-%m-%Y')
        df = df[~df.index.duplicated(keep='first')]

        # Find earliest future date
        today = datetime.today()
        future_earning = df.index[df['EPS_reported'] == '-'].tolist()
        future_earning = [datetime.strptime(date_str, '%d-%m-%Y') for date_str in future_earning]
        future_dates = [date for date in future_earning if date >= today]

        # Find the future business date
        earning_next = min(future_dates).strftime('%d-%m-%Y')
        us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
        us_eastern = pytz.timezone('US/Eastern')
        today = datetime.now(tz=us_eastern)
        future_business_days = pd.date_range(start=today, periods=365, freq=us_bd)
        future_business_days = [d.strftime('%d-%m-%Y') for d in future_business_days]

        # Find the earning delta (days until next earnings)
        if earning_next in future_business_days:
            earning_delta = future_business_days.index(earning_next) + 1
        else:
            # Convert given date to datetime object
            given_date = datetime.strptime(earning_next, '%d-%m-%Y').date()
            # Find the nearest future date
            nearest_date = None
            for date_str in future_business_days:
                date = datetime.strptime(date_str, '%d-%m-%Y').date()
                if date > given_date:
                    if nearest_date is None or date < nearest_date:
                        nearest_date = date
            nearest_date_str = nearest_date.strftime('%d-%m-%Y')
            earning_delta = future_business_days.index(nearest_date_str) + 1

    except Exception as exception:
        print(f"Error processing earnings data: {exception}")
        df = None
        earning_next = '-'
        earning_delta = 365

    return df, earning_next, earning_delta


def fetch_stock_data(stock_symbol, data_period):
    """
    Wrapper function to fetch stock data for a given period.
    
    Args:
        stock_symbol: Stock ticker symbol
        data_period: Number of days of historical data to fetch
        
    Returns:
        DataFrame: Processed stock data with technical indicators
    """
    start_date = datetime.now() - timedelta(days=data_period)
    end_date = datetime.now()
    try:
        raw_data, flg_skip = get_stock_data(stock_symbol, start_date, end_date)
        if flg_skip == 1:
            print(f"Skipping data download for {stock_symbol}")
            return None
        return raw_data
    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {e}")
        return None



def fetch_most_active_stocks():
    """
    Fetch and parse most active stocks from Yahoo Finance
    """
    url = "https://finance.yahoo.com/markets/stocks/most-active"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }
    
    try:
        # Get the page content
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table body
        tbody = soup.find('tbody')
        if not tbody:
            raise ValueError("Could not find the stock table")
        
        stocks_data = []
        
        # Process each row
        for row in tbody.find_all('tr'):
            try:
                # Find all cells in the row
                cells = row.find_all('td')
                if len(cells) < 10:  # Skip rows without enough data
                    continue
                
                # Extract stock data
                symbol = cells[0].find('a').text.strip()
                name = cells[1].div.text.strip()
                
                # Extract price from fin-streamer element
                price = float(cells[3].find('fin-streamer', {'data-field': 'regularMarketPrice'})['data-value'])
                
                # Extract volume
                volume = cells[6].find('fin-streamer', {'data-field': 'regularMarketVolume'})['data-value']
                volume = float(volume)
                
                # Extract market cap
                market_cap = cells[8].find('fin-streamer', {'data-field': 'marketCap'})['data-value']
                market_cap = float(market_cap)
                
                # Extract change percentage
                change_pct = cells[5].find('fin-streamer', {'data-field': 'regularMarketChangePercent'})['data-value']
                change_pct = float(change_pct)
                
                stocks_data.append({
                    'Symbol': symbol,
                    'Name': name,
                    'Price': price,
                    'Volume': volume,
                    'Market_Cap': market_cap,
                    'Change%': change_pct
                })
                
            except (AttributeError, KeyError, ValueError) as e:
                print(f"Error processing row: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(stocks_data)
        
        # Format the data
        df['Volume'] = df['Volume'].apply(lambda x: f"{x/1_000_000:.2f}M")
        df['Market_Cap'] = df['Market_Cap'].apply(lambda x: f"${x/1_000_000_000:.2f}B" if x < 1_000_000_000_000 
                                                else f"${x/1_000_000_000_000:.2f}T")
        df['Price'] = df['Price'].apply(lambda x: f"${x:.2f}")
        df['Change%'] = df['Change%'].apply(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")
        
        return df
        
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None