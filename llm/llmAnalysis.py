import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import re
from dotenv import load_dotenv
from functools import lru_cache
from urllib.parse import urljoin
from .llm_models import PredictionRecord, PerformanceRecord
from .llmTracking import PerformanceEvaluator, LLMReflectionEngine, PredictionTracker
import sqlite3

# Dynamic path setup - Add project root to Python path
current_file = Path(__file__).resolve()
project_root = current_file.parent
if project_root.name != 'llmStock':
    # If this file is in a subdirectory, go up to find project root
    for parent in current_file.parents:
        if parent.name == 'llmStock':
            project_root = parent
            break
sys.path.insert(0, str(project_root))

# External libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import requests
import yfinance as yf
import ta  # Technical Analysis library
from bs4 import BeautifulSoup

# Optional LLM API clients - only needed for specific providers
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None
    
try:
    import openai
except ImportError:
    openai = None

# Local imports
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Core application imports - these are required dependencies
from data.data_loader import fetch_stock_data, get_earning_data
from utils.utils_report import EmailSender
from utils.utils_path import (
    get_project_root as get_root_path, 
    get_save_path, 
    get_plot_path, 
    ensure_directories
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration settings for the application"""
    # Paths
    project_root: Path = None
    save_path: Path = None
    
    # Email settings
    smtp_server: str = None
    smtp_port: int = None
    sender_email: str = None
    sender_password: str = None
    recipient_emails: List[str] = field(default_factory=list)
    
    # Model settings
    default_model: str = None
    predict_window: int = None
    data_window: int = None
    algorithm: int = None  # Default algorithm
    
    # LLM API settings
    ollama_host: str = None
    lmstudio_host: str = None
    gemini_api_key: str = None
    gemini_model: str = None
    deepseek_api_key: str = None
    deepseek_model: str = None
    deepseek_host: str = None
    
    def __post_init__(self):
        """Load configuration from .env file and set defaults"""
        # Load environment variables from .env file
        # Try multiple locations for .env file
        env_paths = [
            Path('.env'),  # Current directory
            get_root_path() / '.env',  # Project root
            Path(__file__).parent.parent / '.env'  # Parent directory
        ]
        
        env_loaded = False
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                env_loaded = True
                break
        
        if not env_loaded:
            # Try to load from environment without file
            load_dotenv()
        
        # Paths - use relative paths as defaults
        self.project_root = Path(os.getenv('PROJECT_ROOT', str(get_root_path())))
        self.save_path = Path(os.getenv('SAVE_PATH', str(get_save_path())))
        
        # Ensure paths exist
        try:
            os.makedirs(self.save_path, exist_ok=True)
        except Exception:
            self.save_path = Path.cwd() / 'output'
            os.makedirs(self.save_path, exist_ok=True)
        
        # Ensure directories exist
        try:
            ensure_directories()
        except Exception:
            # Fallback directory creation
            os.makedirs(self.save_path, exist_ok=True)
            try:
                os.makedirs(get_plot_path(), exist_ok=True)
            except Exception:
                pass
        
        # Email settings
        self.smtp_server = os.getenv('SMTP_SERVER', "smtp.gmail.com")
        self.smtp_port = int(os.getenv('SMTP_PORT', "587"))
        self.sender_email = os.getenv('SENDER_EMAIL', "")
        self.sender_password = os.getenv('SENDER_PASSWORD', "")
        
        # Parse recipient emails from comma-separated string
        recipients_str = os.getenv('RECIPIENT_EMAILS', "")
        self.recipient_emails = [email.strip() for email in recipients_str.split(',') if email.strip()]
        
        # Model settings
        self.default_model = os.getenv('DEFAULT_MODEL', "phi4")  # Changed to phi4
        self.predict_window = int(os.getenv('PREDICT_WINDOW', "48"))
        self.data_window = int(os.getenv('DATA_WINDOW', "96"))
        self.algorithm = int(os.getenv('ALGORITHM', "7"))
        
        # LLM API settings
        self.ollama_host = os.getenv('OLLAMA_HOST', "http://localhost:11434")
        self.lmstudio_host = os.getenv('LMSTUDIO_HOST', "http://localhost:1234/v1")
        self.gemini_api_key = os.getenv('GEMINI_API_KEY', "")
        self.gemini_model = os.getenv('GEMINI_MODEL', "gemini-pro")
        self.deepseek_api_key = os.getenv('DEEPSEEK_API_KEY', "")
        self.deepseek_model = os.getenv('DEEPSEEK_MODEL', "deepseek-chat")
        self.deepseek_host = os.getenv('DEEPSEEK_HOST', "https://api.deepseek.com")
        
    @classmethod
    def create_env_template(cls, file_path: Path = None):
        """Create a template .env file with all configurable variables"""
        if file_path is None:
            file_path = get_root_path() / '.env.template'
            
        template = """# StockAnalyzer Configuration
# ============================
# REQUIRED CONFIGURATION:
# 1. Update email settings below with your credentials
# 2. Choose and configure at least one LLM provider (Ollama, Gemini, or DeepSeek)
# 3. Verify DEFAULT_MODEL matches your chosen LLM provider
# ============================

# Paths - Leave empty to use default relative paths
PROJECT_ROOT=
SAVE_PATH=

# Email settings - REQUIRED for sending analysis reports
# ======================================================
# For Gmail: Use app passwords, not your regular password
# Enable 2FA and generate app password at: https://myaccount.google.com/apppasswords
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-app-password
RECIPIENT_EMAILS=recipient1@example.com,recipient2@example.com

# Model settings
# ==============
# DEFAULT_MODEL should match your chosen LLM provider:
# - For Ollama: use model name like "phi4", "llama3.2:latest", "mistral", etc.
# - For Gemini: this setting is ignored, uses GEMINI_MODEL instead
# - For DeepSeek: this setting is ignored, uses DEEPSEEK_MODEL instead
# - For LM Studio: this setting is ignored, uses whatever model is loaded
DEFAULT_MODEL=phi4
PREDICT_WINDOW=48
DATA_WINDOW=96
ALGORITHM=7

# LLM API settings - Configure at least one provider
# ==================================================

# Option 1: Ollama (Local) - Requires Ollama running locally
# Download from: https://ollama.com/
OLLAMA_HOST=http://localhost:11434

# Option 2: LM Studio (Local) - Requires LM Studio running locally  
# Download from: https://lmstudio.ai/
LMSTUDIO_HOST=http://localhost:1234/v1

# Option 3: Google Gemini (Cloud) - Requires API key from Google AI Studio
# Get your key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=
GEMINI_MODEL=gemini-pro

# Option 4: DeepSeek (Cloud) - Requires API key from DeepSeek
# Get your key at: https://platform.deepseek.com/api_keys
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_HOST=https://api.deepseek.com
"""
        with open(file_path, 'w') as f:
            f.write(template)
        return file_path


class LLMProvider:
    """Base class for LLM providers"""
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM"""
        raise NotImplementedError("Subclasses must implement generate method")
    
    @classmethod
    def create(cls, provider_type: str, **kwargs) -> 'LLMProvider':
        """Factory method to create an LLM provider based on type"""
        providers = {
            'ollama': OllamaProvider,
            'lmstudio': LMStudioProvider,
            'gemini': GeminiProvider,
            'deepseek': DeepSeekProvider
        }
        
        if provider_type.lower() not in providers:
            raise ValueError(f"Unknown provider type: {provider_type}. Available providers: {list(providers.keys())}")
        
        return providers[provider_type.lower()](**kwargs)


class OllamaProvider(LLMProvider):
    """Ollama LLM provider"""
    def __init__(self, model_name: str = None, host: str = None, config: Config = None):
        """
        Initialize the Ollama provider
        
        Args:
            model_name: Name of the model to use
            host: Ollama API host URL
            config: Config object containing settings
        """
        if config:
            self.model_name = model_name or config.default_model
            self.host = host or config.ollama_host
        else:
            self.model_name = model_name or "llama3.1"
            self.host = host or "http://localhost:11434"
            
        # Test connection on initialization
        self._test_connection()
        
    def _test_connection(self) -> None:
        """Test the connection to the Ollama API"""
        try:
            response = requests.get(f"{self.host}/api/tags")
            if response.status_code != 200:
                logger.warning(f"Ollama API connection test failed: {response.status_code}")
            else:
                available_models = [tag['name'] for tag in response.json()['models']]
                if self.model_name not in available_models:
                    logger.warning(f"Model {self.model_name} not found in available models: {available_models}")
                else:
                    logger.info(f"Successfully connected to Ollama API with model {self.model_name}")
        except requests.RequestException as e:
            logger.warning(f"Failed to connect to Ollama API: {e}")
        
    def generate(self, prompt: str) -> str:
        """
        Generate response using Ollama API
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The model's response text
            
        Raises:
            Exception: If the API call fails
        """
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=180  # Add timeout to avoid hanging
            )
            response.raise_for_status()
            return response.json()['response']
        except requests.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise Exception(f"Ollama API error: {e}")


class LMStudioProvider(LLMProvider):
    """LM Studio provider"""
    def __init__(self, host: str = None, config: Config = None):
        """
        Initialize the LM Studio provider
        
        Args:
            host: LM Studio API host URL
            config: Config object containing settings
        """
        if config:
            self.host = host or config.lmstudio_host
        else:
            self.host = host or "http://localhost:1234/v1"
            
        # Test connection on initialization
        self._test_connection()
        
    def _test_connection(self) -> None:
        """Test the connection to the LM Studio API"""
        try:
            response = requests.get(f"{self.host}/models")
            if response.status_code != 200:
                logger.warning(f"LM Studio API connection test failed: {response.status_code}")
            else:
                logger.info("Successfully connected to LM Studio API")
        except requests.RequestException as e:
            logger.warning(f"Failed to connect to LM Studio API: {e}")
        
    def generate(self, prompt: str) -> str:
        """
        Generate response using LM Studio API
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The model's response text
            
        Raises:
            Exception: If the API call fails
        """
        try:
            response = requests.post(
                f"{self.host}/completions",
                json={
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                timeout=60  # Add timeout to avoid hanging
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.RequestException as e:
            logger.error(f"LM Studio API error: {e}")
            raise Exception(f"LM Studio API error: {e}")


class GeminiProvider(LLMProvider):
    """Google Gemini provider using new genai SDK"""
    def __init__(self, api_key: str = None, model_name: str = None, config: Config = None):
        """
        Initialize the Gemini provider
        
        Args:
            api_key: Google API key
            model_name: Gemini model name (default: gemini-2.5-flash)
            config: Config object containing settings
        """
        if genai is None or types is None:
            raise ImportError("google-genai package is required for Gemini provider. Install with: pip install google-genai")
            
        if config:
            self.api_key = api_key or getattr(config, 'gemini_api_key', None) or os.getenv('GEMINI_API_KEY')
            self.model_name = model_name or getattr(config, 'gemini_model', 'gemini-2.5-flash')
        else:
            self.api_key = api_key or os.getenv('GEMINI_API_KEY')
            self.model_name = model_name or 'gemini-2.5-flash'
            
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
            
        # Initialize the client
        self.client = genai.Client(api_key=self.api_key)
        
        # Test connection on initialization
        self._test_connection()
        
    def _test_connection(self) -> None:
        """Test the connection to the Gemini API"""
        try:
            # Try a simple generation to test the connection
            response = self.client.models.generate_content(
                model=self.model_name,
                contents="Hello",
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            logger.info(f"Successfully connected to Gemini API with model {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to connect to Gemini API: {e}")
        
    def generate(self, prompt: str) -> str:
        """
        Generate response using Gemini API
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The model's response text
            
        Raises:
            Exception: If the API call fails
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise Exception(f"Gemini API error: {e}")


class DeepSeekProvider(LLMProvider):
    """DeepSeek provider (using OpenAI-compatible API)"""
    def __init__(self, api_key: str = None, model_name: str = None, base_url: str = None, config: Config = None):
        """
        Initialize the DeepSeek provider
        
        Args:
            api_key: DeepSeek API key
            model_name: DeepSeek model name (default: deepseek-chat)
            base_url: DeepSeek API base URL
            config: Config object containing settings
        """
        if openai is None:
            raise ImportError("openai package is required for DeepSeek provider. Install with: pip install openai")
            
        if config:
            self.api_key = api_key or getattr(config, 'deepseek_api_key', None) or os.getenv('DEEPSEEK_API_KEY')
            self.model_name = model_name or getattr(config, 'deepseek_model', 'deepseek-chat')
            self.base_url = base_url or getattr(config, 'deepseek_host', 'https://api.deepseek.com')
        else:
            self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
            self.model_name = model_name or 'deepseek-chat'
            self.base_url = base_url or 'https://api.deepseek.com'
            
        if not self.api_key:
            raise ValueError("DeepSeek API key is required. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter.")
            
        # Initialize OpenAI client with DeepSeek endpoint
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Test connection on initialization
        self._test_connection()
        
    def _test_connection(self) -> None:
        """Test the connection to the DeepSeek API"""
        try:
            # Try a simple completion to test the connection
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            logger.info(f"Successfully connected to DeepSeek API with model {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to connect to DeepSeek API: {e}")
        
    def generate(self, prompt: str) -> str:
        """
        Generate response using DeepSeek API
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            The model's response text
            
        Raises:
            Exception: If the API call fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
                timeout=120
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise Exception(f"DeepSeek API error: {e}")


class AdvancedTechnicalAnalysis:
    """Advanced technical analysis leveraging capable LLM"""
    
    @staticmethod
    def calculate_comprehensive_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive set of indicators for deep analysis"""
        df = df.copy()
        
        # Price Action & Patterns
        df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low']
        df['S1'] = 2 * df['Pivot'] - df['High']
        df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
        df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
        
        # Advanced Momentum
        df['ROC'] = ta.momentum.roc(df['Close'], window=10)
        df['TSI'] = ta.momentum.tsi(df['Close'])
        df['UO'] = ta.momentum.ultimate_oscillator(df['High'], df['Low'], df['Close'])
        df['AO'] = ta.momentum.awesome_oscillator(df['High'], df['Low'])
        
        # Market Structure
        df['VWAP'] = (df['Volume'] * df['Close']).cumsum() / df['Volume'].cumsum()
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['Price_Spread'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Location'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Volatility Metrics
        df['Historical_Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        df['Parkinson_Volatility'] = np.sqrt(252 / (4 * np.log(2)) * (np.log(df['High'] / df['Low']) ** 2).rolling(window=20).mean())
        df['Garman_Klass'] = np.sqrt(252 * (0.5 * np.log(df['High'] / df['Low']) ** 2 - (2 * np.log(2) - 1) * np.log(df['Close'] / df['Open']) ** 2).rolling(window=20).mean())
        
        # Volume Profile
        df['Volume_Delta'] = df['Volume'] * np.where(df['Close'] > df['Open'], 1, -1)
        df['Cumulative_Delta'] = df['Volume_Delta'].cumsum()
        df['Volume_Rate'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Market Regime
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
        df['Market_Regime'] = pd.cut(df['ADX'], bins=[0, 20, 25, 100], labels=['Ranging', 'Developing', 'Trending'])
        
        # Microstructure
        df['Bid_Ask_Proxy'] = 2 * (df['Close'] - (df['High'] + df['Low']) / 2)
        df['Trade_Intensity'] = df['Volume'] / df['Price_Spread']
        
        # Standard indicators from original code
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)
        
        # Momentum Indicators
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
        df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=14)
        df['Williams_R'] = ta.momentum.WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close'], lbp=14).williams_r()
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # Volatility Indicators
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Middle'] = bollinger.bollinger_mavg()
        df['BB_Lower'] = bollinger.bollinger_lband()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        
        # Volume Indicators
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Additional Trend Indicators
        df['DI_plus'] = ta.trend.adx_pos(df['High'], df['Low'], df['Close'], window=14)
        df['DI_minus'] = ta.trend.adx_neg(df['High'], df['Low'], df['Close'], window=14)
        
        # Ichimoku Cloud
        ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
        df['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
        df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        df['Ichimoku_SpanA'] = ichimoku.ichimoku_a()
        df['Ichimoku_SpanB'] = ichimoku.ichimoku_b()
        
        # Fill missing values
        df = df.bfill().ffill()
        
        return df
    
    @staticmethod
    def detect_chart_patterns(df: pd.DataFrame, window: int = 50) -> Dict[str, Any]:
        """Detect complex chart patterns"""
        patterns = {
            'double_top': False,
            'double_bottom': False,
            'head_shoulders': False,
            'triangle': None,
            'flag': False,
            'channel': None
        }
        
        # Get recent price data
        recent = df.tail(window)
        highs = recent['High'].values
        lows = recent['Low'].values
        closes = recent['Close'].values
        
        # Double Top/Bottom Detection
        high_peaks = []
        low_troughs = []
        
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                high_peaks.append((i, highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                low_troughs.append((i, lows[i]))
        
        # Check for double top
        if len(high_peaks) >= 2:
            last_two_peaks = high_peaks[-2:]
            if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.03:
                patterns['double_top'] = True
        
        # Check for double bottom
        if len(low_troughs) >= 2:
            last_two_troughs = low_troughs[-2:]
            if abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1] < 0.03:
                patterns['double_bottom'] = True
        
        # Triangle Detection (simplified)
        upper_trend = np.polyfit(range(len(highs)), highs, 1)[0]
        lower_trend = np.polyfit(range(len(lows)), lows, 1)[0]
        
        if abs(upper_trend) < 0.01 and lower_trend > 0:
            patterns['triangle'] = 'ascending'
        elif upper_trend < 0 and abs(lower_trend) < 0.01:
            patterns['triangle'] = 'descending'
        elif upper_trend < 0 and lower_trend > 0:
            patterns['triangle'] = 'symmetrical'
        
        # Channel Detection
        close_trend = np.polyfit(range(len(closes)), closes, 1)[0]
        if abs(upper_trend - close_trend) < 0.1 and abs(lower_trend - close_trend) < 0.1:
            patterns['channel'] = 'parallel'
        
        return patterns
    
    @staticmethod
    def calculate_market_internals(ticker: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market internals and correlations"""
        try:
            # Get sector ETF for correlation
            sector_map = {
                'AAPL': 'XLK', 'MSFT': 'XLK', 'GOOGL': 'XLK', 'META': 'XLK',
                'AMZN': 'XLY', 'TSLA': 'XLY', 'NVDA': 'XLK', 'AMD': 'XLK'
            }
            
            # Get market indices - ensure timezone consistency
            spy_data = yf.download('SPY', period='3mo', progress=False, auto_adjust=True)
            vix_data = yf.download('^VIX', period='3mo', progress=False, auto_adjust=True)
            
            # Extract Close prices as Series
            spy = spy_data['Close'] if isinstance(spy_data['Close'], pd.Series) else spy_data['Close'].squeeze()
            vix = vix_data['Close'] if isinstance(vix_data['Close'], pd.Series) else vix_data['Close'].squeeze()
            
            # Get sector ETF if applicable
            sector_etf = sector_map.get(ticker, 'SPY')
            sector_data = yf.download(sector_etf, period='3mo', progress=False, auto_adjust=True)
            sector = sector_data['Close'] if isinstance(sector_data['Close'], pd.Series) else sector_data['Close'].squeeze()
            
            # Remove timezone info if present to avoid tz-aware/naive conflicts
            if hasattr(df.index, 'tz'):
                df = df.copy()
                df.index = df.index.tz_localize(None)
            
            if hasattr(spy.index, 'tz'):
                spy.index = spy.index.tz_localize(None)
            if hasattr(vix.index, 'tz'):
                vix.index = vix.index.tz_localize(None)
            if hasattr(sector.index, 'tz'):
                sector.index = sector.index.tz_localize(None)
            
            # Align dates between df and market data
            common_dates = df.index.intersection(spy.index)
            if len(common_dates) == 0:
                raise ValueError("No overlapping dates between stock and market data")
            
            # Get aligned data
            recent_close = df.loc[common_dates, 'Close']
            spy_aligned = spy.loc[common_dates]
            vix_aligned = vix.loc[common_dates] if len(vix.index.intersection(common_dates)) > 0 else pd.Series()
            sector_aligned = sector.loc[common_dates]
            
            # Calculate returns for beta calculation
            stock_returns = recent_close.pct_change().dropna()
            spy_returns = spy_aligned.pct_change().dropna()
            
            # Align returns
            aligned_dates = stock_returns.index.intersection(spy_returns.index)
            stock_returns_aligned = stock_returns.loc[aligned_dates]
            spy_returns_aligned = spy_returns.loc[aligned_dates]
            
            # Calculate beta only if we have enough data
            if len(stock_returns_aligned) > 1:
                # Ensure both are 1D numpy arrays
                stock_ret_array = stock_returns_aligned.values.flatten()
                spy_ret_array = spy_returns_aligned.values.flatten()
                
                # Calculate covariance and variance
                covariance = np.cov(stock_ret_array, spy_ret_array)[0, 1]
                spy_variance = np.var(spy_ret_array, ddof=1)
                beta = covariance / spy_variance if spy_variance != 0 else 1.0
            else:
                beta = 1.0
            
            internals = {
                'spy_correlation': recent_close.corr(spy_aligned) if len(spy_aligned) > 0 else 0,
                'sector_correlation': recent_close.corr(sector_aligned) if len(sector_aligned) > 0 else 0,
                'inverse_vix_correlation': recent_close.corr(vix_aligned) * -1 if len(vix_aligned) > 0 else 0,
                'beta': beta,
                'relative_strength': (stock_returns.mean() - spy_returns.mean()) * 252 if len(spy_returns) > 0 else 0
            }
                
            return internals
        except Exception as e:
            logger.warning(f"Error calculating market internals: {e}")
            return {
                'spy_correlation': 0,
                'sector_correlation': 0,
                'inverse_vix_correlation': 0,
                'beta': 1,
                'relative_strength': 0
            }


class MultiTimeframeAnalysis:
    """Analyze across multiple timeframes for comprehensive view"""
    
    @staticmethod
    def get_multi_timeframe_signals(ticker: str) -> Dict[str, Any]:
        """Get signals across multiple timeframes"""
        timeframes = {
            'intraday': '5d',
            'short_term': '1mo',
            'medium_term': '3mo',
            'long_term': '1y'
        }
        
        signals = {}
        
        for tf_name, period in timeframes.items():
            try:
                df = yf.Ticker(ticker).history(period=period)
                if len(df) > 0:
                    # Calculate basic indicators for each timeframe
                    df['SMA20'] = df['Close'].rolling(20).mean()
                    df['RSI'] = ta.momentum.rsi(df['Close'])
                    
                    latest = df.iloc[-1]
                    signals[tf_name] = {
                        'trend': 'bullish' if latest['Close'] > df['SMA20'].iloc[-1] else 'bearish',
                        'momentum': 'strong' if latest['RSI'] > 60 else 'weak' if latest['RSI'] < 40 else 'neutral',
                        'price_change': (latest['Close'] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
                    }
            except:
                signals[tf_name] = {'trend': 'unknown', 'momentum': 'unknown', 'price_change': 0}
        
        return signals


class OptionsFlowAnalysis:
    """Analyze options flow for additional insights"""
    
    @staticmethod
    def get_options_metrics(ticker: str) -> Dict[str, Any]:
        """Get options-based metrics"""
        try:
            stock = yf.Ticker(ticker)
            
            # Get options chain
            exp_dates = stock.options
            if not exp_dates:
                return {}
            
            # Use nearest expiration for analysis
            options = stock.option_chain(exp_dates[0])
            calls = options.calls
            puts = options.puts
            
            # Calculate Put/Call ratio
            put_volume = puts['volume'].sum()
            call_volume = calls['volume'].sum()
            pc_ratio = put_volume / call_volume if call_volume > 0 else 0
            
            # Calculate max pain
            current_price = stock.history(period='1d')['Close'].iloc[-1]
            
            # Options metrics
            metrics = {
                'put_call_ratio': pc_ratio,
                'call_volume': call_volume,
                'put_volume': put_volume,
                'iv_percentile': 'N/A',  # Would need historical IV data
                'options_sentiment': 'bullish' if pc_ratio < 0.7 else 'bearish' if pc_ratio > 1.3 else 'neutral'
            }
            
            return metrics
        except:
            return {
                'put_call_ratio': 0,
                'call_volume': 0,
                'put_volume': 0,
                'options_sentiment': 'unknown'
            }


class RealTimeMonitor:
    """Monitor positions in real-time with alerts"""
    
    @staticmethod
    def create_monitoring_prompt(ticker: str, entry_price: float, 
                               current_price: float, holding_days: int,
                               original_analysis: str) -> str:
        """Create prompt for position monitoring"""
        
        pnl = (current_price - entry_price) / entry_price * 100
        
        prompt = f"""Review and update the trading position in {ticker}:

POSITION DETAILS:
- Entry Price: ${entry_price:.2f}
- Current Price: ${current_price:.2f}
- P&L: {pnl:+.2f}%
- Holding Period: {holding_days} days

ORIGINAL ANALYSIS SUMMARY:
{original_analysis[:500]}...

Based on current market conditions and technical indicators:
1. Should the position be maintained, scaled, or closed?
2. Should stop loss or take profit levels be adjusted?
3. Any new risks or catalysts to consider?
4. Updated price targets?

Provide specific actionable recommendations."""
        
        return prompt


def create_advanced_analysis_prompt(ticker: str, df: pd.DataFrame, 
                                  sentiment_analysis: Dict[str, Any],
                                  market_internals: Dict[str, Any],
                                  patterns: Dict[str, Any],
                                  multi_tf_signals: Dict[str, Any]) -> str:
    """Create comprehensive prompt for capable LLM"""
    
    latest = df.iloc[-1]
    week_ago = df.iloc[-5] if len(df) > 5 else df.iloc[0]
    month_ago = df.iloc[-22] if len(df) > 22 else df.iloc[0]
    
    # Prepare pattern description
    pattern_desc = []
    if patterns['double_top']:
        pattern_desc.append("Double Top (bearish reversal)")
    if patterns['double_bottom']:
        pattern_desc.append("Double Bottom (bullish reversal)")
    if patterns['triangle']:
        pattern_desc.append(f"{patterns['triangle'].title()} Triangle")
    if patterns['channel']:
        pattern_desc.append(f"{patterns['channel'].title()} Channel")
    
    prompt = f"""Perform an institutional-grade technical analysis of {ticker} with actionable trading recommendations.

## CURRENT MARKET SNAPSHOT
Price: ${latest['Close']:.2f}
Daily Change: {((latest['Close'] - latest['Open']) / latest['Open'] * 100):.2f}%
Weekly Change: {((latest['Close'] - week_ago['Close']) / week_ago['Close'] * 100):.2f}%
Monthly Change: {((latest['Close'] - month_ago['Close']) / month_ago['Close'] * 100):.2f}%

## MARKET MICROSTRUCTURE
VWAP: ${latest['VWAP']:.2f} (Price vs VWAP: {((latest['Close'] - latest['VWAP']) / latest['VWAP'] * 100):.2f}%)
Close Location: {latest['Close_Location']:.2%} (1 = high of day, 0 = low of day)
Trade Intensity: {latest['Trade_Intensity']:.2f}
Volume Rate: {latest['Volume_Rate']:.2f}x average
Cumulative Delta: {latest['Cumulative_Delta']:,.0f}

## VOLATILITY PROFILE
Historical Volatility (20d): {latest['Historical_Volatility']:.1%}
Parkinson Volatility: {latest['Parkinson_Volatility']:.1%}
Garman-Klass Volatility: {latest['Garman_Klass']:.1%}
ATR: ${latest['ATR']:.2f} ({(latest['ATR'] / latest['Close'] * 100):.1f}% of price)
Market Regime: {latest['Market_Regime']}

## TECHNICAL INDICATORS
**Trend Following:**
- SMA20: ${latest['SMA_20']:.2f} (Price {('above' if latest['Close'] > latest['SMA_20'] else 'below')} by {abs((latest['Close'] - latest['SMA_20']) / latest['SMA_20'] * 100):.1f}%)
- SMA50: ${latest['SMA_50']:.2f} (Price {('above' if latest['Close'] > latest['SMA_50'] else 'below')} by {abs((latest['Close'] - latest['SMA_50']) / latest['SMA_50'] * 100):.1f}%)
- SMA200: ${latest['SMA_200']:.2f} (Price {('above' if latest['Close'] > latest['SMA_200'] else 'below')} by {abs((latest['Close'] - latest['SMA_200']) / latest['SMA_200'] * 100):.1f}%)
- ADX: {latest['ADX']:.1f} (Trend Strength: {('Strong' if latest['ADX'] > 25 else 'Weak')})

**Momentum Oscillators:**
- RSI(14): {latest['RSI']:.1f}
- Stochastic %K: {latest['Stoch_K']:.1f}
- Williams %R: {latest['Williams_R']:.1f}
- TSI: {latest['TSI']:.2f}
- Ultimate Oscillator: {latest['UO']:.1f}
- Awesome Oscillator: {latest['AO']:.2f}

**Volume Analysis:**
- OBV Trend: {('Positive' if latest['OBV'] > df['OBV'].iloc[-20] else 'Negative')}
- CMF: {latest['CMF']:.3f} ({('Accumulation' if latest['CMF'] > 0 else 'Distribution')})
- MFI: {latest['MFI']:.1f}

**Volatility Bands:**
- BB Upper: ${latest['BB_Upper']:.2f}
- BB Lower: ${latest['BB_Lower']:.2f}
- BB Position: {((latest['Close'] - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']) * 100):.0f}%

## CHART PATTERNS DETECTED
{', '.join(pattern_desc) if pattern_desc else 'No significant patterns detected'}

## MARKET INTERNALS & CORRELATIONS
- SPY Correlation (3mo): {market_internals['spy_correlation']:.2f}
- Sector Correlation: {market_internals['sector_correlation']:.2f}
- Inverse VIX Correlation: {market_internals['inverse_vix_correlation']:.2f}
- Beta: {market_internals['beta']:.2f}
- Relative Strength vs Market: {market_internals['relative_strength']:.1%} annualized

## MULTI-TIMEFRAME ANALYSIS
{json.dumps(multi_tf_signals, indent=2)}

## NEWS SENTIMENT
{sentiment_analysis['summary']}
Key Events Impact: {sentiment_analysis['price_impact']}
Sentiment Confidence: {sentiment_analysis['confidence']:.1%}

## PIVOT LEVELS
- R2: ${latest['R2']:.2f}
- R1: ${latest['R1']:.2f}
- Pivot: ${latest['Pivot']:.2f}
- S1: ${latest['S1']:.2f}
- S2: ${latest['S2']:.2f}

REQUIRED ANALYSIS:

1. **Market Context & Regime Analysis**
   - Identify the current market regime (trending/ranging/volatile)
   - Assess institutional positioning based on volume patterns
   - Evaluate sector rotation implications

2. **Technical Setup Quality Score (1-10)**
   - Rate the current technical setup
   - Identify confluence of signals
   - Assess risk/reward ratio

3. **Entry Strategy**
   - Primary entry point with specific price
   - Alternative entry scenarios
   - Position sizing recommendation (as % of portfolio)

4. **Exit Strategy**
   - Take profit targets (T1, T2, T3) with rationale
   - Stop loss placement with reasoning
   - Trailing stop strategy

5. **Risk Analysis**
   - Key risk factors (technical and fundamental)
   - Probability of success (percentage)
   - Maximum drawdown expectation

6. **Time Horizon**
   - Expected holding period
   - Key dates/events to watch
   - Catalyst timeline

7. **Alternative Scenarios**
   - Bull case with probability
   - Base case with probability  
   - Bear case with probability

FINAL RECOMMENDATION:
Provide a clear action with the EXACT format:
RECOMMENDATION: [BUY/SELL/HOLD] (Confidence: [HIGH/MEDIUM/LOW])

Include specific reasoning for the confidence level based on:
- Signal confluence
- Market regime alignment
- Risk/reward ratio
- Time frame convergence"""
    
    return prompt


class StockAnalyzer:
    """Stock analysis and prediction class"""
    
    def __init__(self, llm_provider: LLMProvider, config: Config = None):
        """
        Initialize the StockAnalyzer with an LLM provider and configuration
        
        Args:
            llm_provider: An LLM provider instance
            config: Configuration settings (optional)
        """
        self.llm = llm_provider
        self.config = config or Config()
        
    def get_stock_data(self, ticker: str, period: str = '1y') -> Tuple[pd.DataFrame, Optional[pd.DataFrame], int]:
        """
        Fetch stock data using yfinance
        
        Args:
            ticker: Stock ticker symbol
            period: Time period to fetch ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            Tuple of (historical dataframe, advanced dataframe, earnings delta)
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

            # Return advanced data
            adv_df = fetch_stock_data(ticker, 1260)
            earning_table, earning_next, earning_delta = get_earning_data(ticker)
            
            return df, adv_df, earning_delta
        
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {e}")
            return None, None, 365  # Default earnings delta

    @staticmethod
    def fetch_yahoo_news(ticker: str) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Fetches news articles from Yahoo Finance. This version includes a "session warm-up"
        to handle advanced bot detection on high-traffic tickers like AMZN.
        Updated to handle the new Yahoo Finance HTML structure.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'AMZN', 'LCID')

        Returns:
            List of news article dictionaries.
        """
        session = requests.Session()
        base_url = "https://finance.yahoo.com"
        
        # Use a more realistic, static User-Agent. fake_useragent can sometimes
        # generate strings that are on blocklists. A common, modern one is better.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
        }

        # The news URL we want to access
        news_url = f"{base_url}/quote/{ticker}/news?p={ticker}"
        
        articles = []
        
        try:
            # --- SESSION WARM-UP ---
            # Visit the main quote page first to get necessary cookies. This is crucial
            # for high-traffic tickers like AMZN, which use this to block simple scrapers.
            warmup_url = f"{base_url}/quote/{ticker}"
            logger.info(f"Warming up session for {ticker} at: {warmup_url}")
            session.get(warmup_url, headers=headers, timeout=10)
            
            # Now, make the actual request to the news page using the warmed-up session
            logger.info(f"Fetching news for {ticker} from: {news_url}")
            response = session.get(news_url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Updated parsing logic for new Yahoo Finance structure
            # Look for h3 elements with the specific class that contain article titles
            title_elements = soup.find_all('h3', {'class': lambda x: x and 'clamp' in x and 'yf-' in x})

            if not title_elements:
                logger.warning(f"Could not find any news items for {ticker}. The page may be structured differently or have no news.")
                return []

            for title_elem in title_elements:
                try:
                    # Extract title
                    title = title_elem.get_text(strip=True)
                    if not title:
                        continue
                    
                    # Find the corresponding summary paragraph (should be the next sibling or nearby)
                    summary = ""
                    summary_elem = title_elem.find_next_sibling('p', {'class': lambda x: x and 'clamp' in x and 'yf-' in x})
                    if summary_elem:
                        # Extract text and clean up HTML comments
                        summary_text = summary_elem.get_text(strip=True)
                        # Remove HTML comment markers if present
                        summary = summary_text.replace('<!-- HTML_TAG_START -->', '').replace('<!-- HTML_TAG_END -->', '').strip()
                    
                    # Try to find article URL - look for links in or around the title element
                    article_url = ""
                    link_elem = title_elem.find('a')
                    if not link_elem:
                        # Look for a parent or nearby element that might contain the link
                        parent = title_elem.parent
                        if parent:
                            link_elem = parent.find('a')
                    
                    if link_elem and link_elem.get('href'):
                        relative_url = link_elem['href']
                        article_url = urljoin(base_url, relative_url)
                    
                    # Try to extract source and timestamp - these might be in nearby elements
                    source = "Unknown"
                    timestamp = "Unknown"
                    
                    # Look for source information in subsequent elements
                    current_elem = summary_elem if summary_elem else title_elem
                    for sibling in current_elem.find_next_siblings():
                        if sibling.name in ['div', 'span'] and sibling.get_text(strip=True):
                            text_content = sibling.get_text(strip=True)
                            # Simple heuristic: if it contains time-related words or source patterns
                            if any(word in text_content.lower() for word in ['ago', 'hours', 'minutes', 'days', 'reuters', 'bloomberg', 'yahoo']):
                                if source == "Unknown":
                                    source = text_content
                                elif timestamp == "Unknown":
                                    timestamp = text_content
                                break
                    
                    # Try to find related tickers - look for ticker symbols in nearby elements
                    related_tickers = []
                    ticker_container = title_elem.find_next('div', {'class': lambda x: x and 'related' in x.lower() if x else False})
                    if ticker_container:
                        ticker_links = ticker_container.find_all('a')
                        related_tickers = [link.get_text(strip=True) for link in ticker_links if link.get_text(strip=True)]

                    # Only add article if we have at least a title
                    if title:
                        articles.append({
                            'title': title,
                            'summary': summary,
                            'source': source,
                            'timestamp': timestamp,
                            'url': article_url,
                            'related_tickers': related_tickers
                        })

                except Exception as e:
                    logger.warning(f"Error parsing individual article for {ticker}: {e}")
                    continue

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.error(f"Failed to fetch news for {ticker} (404 Not Found). The ticker may be invalid or you may be blocked. URL used: {news_url}")
            else:
                logger.error(f"HTTP Error fetching news for {ticker}: {e}")
        except requests.RequestException as e:
            logger.error(f"Network error fetching news for {ticker}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while parsing news for {ticker}: {e}", exc_info=True)

        return articles

    def _parse_llm_response(self, response_text: str, response_type: str = 'sentiment') -> Dict[str, Any]:
        """
        Parse LLM response and attempt to extract JSON data, with robust fallback handling
        
        Args:
            response_text: Raw response text from LLM
            response_type: Type of response to parse ('sentiment' or 'relevance')
        
        Returns:
            Parsed response with appropriate structure based on response_type
        """
        
        def _fix_json_formatting(json_str: str) -> str:
            """Fix common JSON formatting issues"""
            # Remove any leading/trailing whitespace
            json_str = json_str.strip()
            
            # Fix unquoted string values after colons
            # This regex finds patterns like: "key": unquoted text that should be quoted
            # It looks for a colon followed by whitespace, then captures text until a comma or closing brace
            def fix_unquoted_values(match):
                key = match.group(1)
                value = match.group(2).strip()
                
                # Skip if value is already quoted, a number, boolean, or null
                if (value.startswith('"') and value.endswith('"')) or \
                value in ['true', 'false', 'null'] or \
                re.match(r'^-?\d+\.?\d*$', value):
                    return match.group(0)
                
                # Quote the value and escape internal quotes
                escaped_value = value.replace('"', '\\"').replace('\\', '\\\\')
                return f'"{key}": "{escaped_value}"'
            
            # Pattern to match unquoted string values
            pattern = r'"([^"]+)":\s*([^,}]+?)(?=\s*[,}])'
            json_str = re.sub(pattern, fix_unquoted_values, json_str)
            
            return json_str
        
        def _extract_fields_with_regex(text: str, response_type: str) -> Dict[str, Any]:
            """Extract fields using regex as fallback"""
            result = {}
            
            if response_type == 'relevance':
                # Extract is_relevant
                is_relevant_match = re.search(r'"is_relevant":\s*(true|false)', text, re.IGNORECASE)
                result['is_relevant'] = is_relevant_match.group(1).lower() == 'true' if is_relevant_match else False
                
                # Extract confidence
                confidence_match = re.search(r'"confidence":\s*([\d.]+)', text)
                result['confidence'] = float(confidence_match.group(1)) if confidence_match else 0.0
                
                # Extract reasoning (everything between "reasoning": and the next " or end)
                reasoning_match = re.search(r'"reasoning":\s*"?(.*?)(?=",|\s*}|$)', text, re.DOTALL)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
                    # Clean up common artifacts
                    reasoning = reasoning.replace('\\"', '"').replace('\\\\', '\\')
                    reasoning = reasoning.rstrip('",}')
                    result['reasoning'] = reasoning
                else:
                    result['reasoning'] = 'Unable to extract reasoning'
                    
            else:  # sentiment analysis
                # Extract sentiment
                sentiment_match = re.search(r'"sentiment":\s*"?([^",}\s]+)', text, re.IGNORECASE)
                result['sentiment'] = sentiment_match.group(1).strip('"') if sentiment_match else 'neutral'
                
                # Extract confidence
                confidence_match = re.search(r'"confidence":\s*([\d.]+)', text)
                result['confidence'] = float(confidence_match.group(1)) if confidence_match else 0.5
                
                # Extract price_impact
                price_impact_match = re.search(r'"price_impact":\s*"?([^",}\s]+)', text, re.IGNORECASE)
                result['price_impact'] = price_impact_match.group(1).strip('"') if price_impact_match else 'neutral'
                
                # Extract impact_timeframe
                timeframe_match = re.search(r'"impact_timeframe":\s*"?([^",}\s]+)', text, re.IGNORECASE)
                result['impact_timeframe'] = timeframe_match.group(1).strip('"') if timeframe_match else 'short-term'
                
                # Set default arrays
                result['key_points'] = ['Analysis extracted via regex fallback']
                result['risk_factors'] = ['Parsing uncertainty']
            
            return result
        
        try:
            # First, try to find JSON content between triple backticks
            if '```json' in response_text and '```' in response_text:
                json_str = response_text.split('```json')[1].split('```')[0].strip()
            # Try to find content between curly braces
            elif '{' in response_text and '}' in response_text:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                json_str = response_text[start_idx:end_idx]
            else:
                # Use the entire response
                json_str = response_text.strip()
            
            # Try to parse as-is first
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try to fix formatting issues
                fixed_json_str = _fix_json_formatting(json_str)
                try:
                    return json.loads(fixed_json_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed even after formatting fixes: {e}")
                    raise e
                    
        except Exception as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Raw response: {response_text}")
            
            # Try regex extraction as fallback
            try:
                extracted_result = _extract_fields_with_regex(response_text, response_type)
                logger.info(f"Successfully extracted fields using regex fallback: {extracted_result}")
                return extracted_result
            except Exception as regex_error:
                logger.error(f"Regex fallback also failed: {regex_error}")
            
            # Return appropriate default structure based on response type
            if response_type == 'relevance':
                return {
                    'is_relevant': False,
                    'confidence': 0.0,
                    'reasoning': 'Failed to parse model response'
                }
            else:  # sentiment analysis default
                return {
                    'sentiment': 'neutral',
                    'confidence': 0.5,
                    'price_impact': 'neutral',
                    'impact_timeframe': 'short-term',
                    'key_points': ['Unable to parse model response'],
                    'risk_factors': ['Analysis uncertainty']
                }

    def _validate_json_response(self, parsed_response: Dict[str, Any], response_type: str = 'sentiment') -> Dict[str, Any]:
        """
        Validate and sanitize parsed JSON response to ensure all required fields are present
        
        Args:
            parsed_response: Parsed JSON response
            response_type: Type of response to validate ('sentiment' or 'relevance')
        
        Returns:
            Validated and sanitized response
        """
        if response_type == 'relevance':
            # Define expected fields and types for relevance check
            expected_fields = {
                'is_relevant': bool,
                'confidence': float,
                'reasoning': str
            }
            
            # Set default values for missing or invalid fields
            default_values = {
                'is_relevant': False,
                'confidence': 0.0,
                'reasoning': 'Invalid response structure'
            }
        else:  # sentiment analysis
            # Define expected fields and types for sentiment analysis
            expected_fields = {
                'sentiment': str,
                'confidence': float,
                'price_impact': str,
                'impact_timeframe': str,
                'key_points': list,
                'risk_factors': list
            }
            
            # Set default values for missing or invalid fields
            default_values = {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'price_impact': 'neutral',
                'impact_timeframe': 'short-term',
                'key_points': ['Invalid response structure'],
                'risk_factors': ['Analysis uncertainty']
            }

        # Validate and sanitize the response
        validated_response = {}
        for field, expected_type in expected_fields.items():
            try:
                value = parsed_response.get(field, default_values[field])
                if not isinstance(value, expected_type):
                    if expected_type == float:
                        value = float(value)  # Try to convert to float
                    elif expected_type == bool:
                        value = bool(value)   # Try to convert to bool
                    elif expected_type == str:
                        value = str(value)    # Try to convert to string
                    elif expected_type == list and not isinstance(value, list):
                        value = [value] if value else []  # Convert to list or empty list
                    else:
                        value = default_values[field]
                validated_response[field] = value
            except (ValueError, TypeError):
                validated_response[field] = default_values[field]

        return validated_response

    def analyze_sentiment(self, ticker: str, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment for multiple articles and provide a comprehensive summary.
        Includes relevance checking to filter out unrelated articles.
        
        Args:
            ticker: Stock ticker symbol
            articles: List of article dictionaries
            
        Returns:
            Dictionary with sentiment analysis results
        """
        analyzed_articles = []
        overall_sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_confidence = 0
        key_events = []
        price_impacts = []

        # Skip analysis if no articles
        if not articles:
            return {
                'overall_sentiment': 'neutral',
                'confidence': 0.0,
                'summary': f'No news articles available for {ticker}',
                'key_events': [],
                'price_impact': 'neutral',
                'detailed_analyses': [],
                'articles_processed': {'total': 0, 'relevant': 0}
            }

        # Use ThreadPoolExecutor for concurrent article analysis
        with ThreadPoolExecutor(max_workers=min(10, len(articles))) as executor:
            futures = []
            
            for article in articles:
                futures.append(executor.submit(self._analyze_single_article, ticker, article))
            
            for future in futures:
                try:
                    result = future.result()
                    if result:
                        article_analysis, relevance_result, analysis = result
                        
                        # Update sentiment counts
                        overall_sentiment_counts[analysis['sentiment']] += 1
                        total_confidence += analysis['confidence']
                        
                        # Store key events and price impacts
                        key_events.extend([{
                            'point': point,
                            'timeframe': analysis['impact_timeframe'],
                            'confidence': analysis['confidence']
                        } for point in analysis['key_points']])
                        
                        price_impacts.append({
                            'impact': analysis['price_impact'],
                            'confidence': analysis['confidence']
                        })
                        
                        analyzed_articles.append(article_analysis)
                except Exception as e:
                    logger.error(f"Error analyzing article: {e}")
                    continue

        # Calculate overall metrics
        total_articles = len(analyzed_articles)
        if total_articles == 0:
            return {
                'overall_sentiment': 'neutral',
                'confidence': 0.0,
                'summary': f'No relevant articles available for analysis of {ticker}',
                'key_events': [],
                'price_impact': 'neutral',
                'detailed_analyses': [],
                'articles_processed': {'total': len(articles), 'relevant': 0}
            }
        
        # Determine overall sentiment
        overall_sentiment = max(overall_sentiment_counts.items(), key=lambda x: x[1])[0]
        avg_confidence = total_confidence / total_articles
        
        # Sort and filter key events by confidence
        key_events.sort(key=lambda x: x['confidence'], reverse=True)
        top_key_events = key_events[:5]  # Get top 5 most confident events
        
        # Calculate weighted price impact
        impact_weights = {'likely positive': 1, 'neutral': 0, 'likely negative': -1}
        weighted_impact = sum(
            impact_weights.get(impact['impact'], 0) * impact['confidence']
            for impact in price_impacts
        ) / total_articles
        
        overall_price_impact = (
            'likely positive' if weighted_impact > 0.2
            else 'likely negative' if weighted_impact < -0.2
            else 'neutral'
        )
        
        # Generate summary
        summary = f"""Recent news analysis for {ticker}:
        - Total Articles Analyzed: {total_articles} (filtered from {len(articles)} total articles)
        - Overall Sentiment: {overall_sentiment} (confidence: {avg_confidence:.2f})
        - Key Events:
        {chr(10).join(f'  * {event["point"]} ({event["timeframe"]})' for event in top_key_events)}
        - Expected Price Impact: {overall_price_impact}
        """
        logger.info(summary)
        
        return {
            'overall_sentiment': overall_sentiment,
            'confidence': avg_confidence,
            'summary': summary,
            'key_events': top_key_events,
            'price_impact': overall_price_impact,
            'detailed_analyses': analyzed_articles,
            'articles_processed': {
                'total': len(articles),
                'relevant': total_articles
            }
        }

    def _analyze_single_article(self, ticker: str, article: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        """
        Analyze a single article for relevance and sentiment
        
        Args:
            ticker: Stock ticker symbol
            article: Article dictionary
            
        Returns:
            Tuple of (article analysis, relevance result, sentiment analysis) or None if failed
        """
        try:
            # Combine title and summary for analysis
            full_text = f"{article['title']} {article['summary']}"
            
            # First, check if the article is relevant to the ticker
            relevance_prompt = f"""Determine if this news article is directly related to {ticker} stock or company.
            News: {full_text}

            Response format:
            {{
                "is_relevant": true/false,
                "confidence": <float 0-1>,
                "reasoning": "brief explanation"
            }}"""
            
            relevance_response = self.llm.generate(relevance_prompt)
            relevance_result = self._validate_json_response(
                self._parse_llm_response(relevance_response, response_type='relevance'),
                response_type='relevance'
            )
            
            # Skip article if it's not relevant
            if not relevance_result.get('is_relevant', False):
                return None
                
            # Proceed with sentiment analysis only for relevant articles
            sentiment_prompt = f"""Analyze this news about {ticker} stock and return a JSON response.
            News: {full_text}

            Response format:
            {{
                "sentiment": "positive/negative/neutral",
                "confidence": <float 0-1>,
                "price_impact": "likely positive/negative/neutral",
                "impact_timeframe": "short-term/medium-term/long-term",
                "key_points": ["point1", "point2"],
                "risk_factors": ["risk1", "risk2"]
            }}"""
            
            sentiment_response = self.llm.generate(sentiment_prompt)
            
            analysis = self._validate_json_response(
                self._parse_llm_response(sentiment_response, response_type='sentiment'),
                response_type='sentiment'
            )
            
            article_analysis = {
                'article': article,
                'analysis': analysis,
                'relevance': {
                    'confidence': relevance_result['confidence'],
                    'reasoning': relevance_result['reasoning']
                }
            }
            
            return article_analysis, relevance_result, analysis
                
        except Exception as e:
            logger.error(f"Error analyzing article: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators using the advanced analysis class
        
        Args:
            df: Historical price dataframe
            
        Returns:
            Dataframe with added technical indicators
        """
        return AdvancedTechnicalAnalysis.calculate_comprehensive_indicators(df)
    
    def calculate_predictions(self, ticker: str, adv_df: pd.DataFrame, algorithm: int, data_window: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate price predictions using machine learning models
        
        Args:
            ticker: Stock ticker symbol
            adv_df: Advanced dataframe with technical indicators
            algorithm: Algorithm ID to use
            data_window: Window size for prediction
            
        Returns:
            Tuple of (past predictions, future predictions)
        """
        if adv_df is None:
            return np.array([]), np.array([])
            
        # ML predictions disabled - exc_torch dependency removed
        logger.info(f"ML predictions disabled for {ticker} - exc_torch dependency removed")
        return np.array([]), np.array([])

    def plot_technical_analysis(self, ticker: str, df: pd.DataFrame, adv_df: Optional[pd.DataFrame], 
                               predict_window: int, predicted_p: np.ndarray, predicted_f: np.ndarray, 
                               earning_delta: int) -> plt.Figure:
        """
        Create technical analysis plots with candlestick chart
        
        Args:
            ticker: Stock ticker symbol
            df: Historical price dataframe with technical indicators
            adv_df: Advanced dataframe (optional)
            predict_window: Window size for prediction
            predicted_p: Past predictions
            predicted_f: Future predictions
            earning_delta: Days until next earnings
            
        Returns:
            Matplotlib figure object
        """
        try:
            # Clear any previous plots
            plt.clf()
            fig = plt.figure(figsize=(15, 15))
            gs = fig.add_gridspec(5, 1, height_ratios=[3, 1, 1, 1, 2])
            
            # Candlestick chart with indicators
            ax1 = fig.add_subplot(gs[0])
            
            # Plot candlesticks
            candlestick_data = df[['Open', 'High', 'Low', 'Close']]
            up = df[df.Close >= df.Open]
            down = df[df.Close < df.Open]
            
            # Plot candlesticks
            width = 0.8
            width2 = 0.1
            
            # Up candlesticks
            ax1.bar(up.index, up.Close-up.Open, width, bottom=up.Open, color='g', alpha=0.7)
            ax1.bar(up.index, up.High-up.Close, width2, bottom=up.Close, color='g', alpha=0.7)
            ax1.bar(up.index, up.Low-up.Open, width2, bottom=up.Open, color='g', alpha=0.7)
            
            # Down candlesticks
            ax1.bar(down.index, down.Close-down.Open, width, bottom=down.Open, color='r', alpha=0.7)
            ax1.bar(down.index, down.High-down.Open, width2, bottom=down.Open, color='r', alpha=0.7)
            ax1.bar(down.index, down.Low-down.Close, width2, bottom=down.Close, color='r', alpha=0.7)
            
            # Add moving averages and Bollinger Bands
            ax1.plot(df.index, df['SMA_20'], label='SMA 20', color='blue', alpha=0.7)
            ax1.plot(df.index, df['SMA_50'], label='SMA 50', color='red', alpha=0.7)
            ax1.plot(df.index, df['BB_Upper'], label='BB Upper', color='gray', linestyle='--', alpha=0.5)
            ax1.plot(df.index, df['BB_Lower'], label='BB Lower', color='gray', linestyle='--', alpha=0.5)
            
            ax1.set_title(f'{ticker} Technical Analysis')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True)

            # Volume subplot
            ax2 = fig.add_subplot(gs[1])
            ax2.bar(df.index, df['Volume'], color=['green' if c >= o else 'red' for c, o in zip(df['Close'], df['Open'])], alpha=0.7)
            ax2.set_ylabel('Volume')
            ax2.grid(True)

            # MACD
            ax3 = fig.add_subplot(gs[2])
            ax3.plot(df.index, df['MACD'], label='MACD', color='blue')
            ax3.plot(df.index, df['MACD_Signal'], label='Signal', color='red')
            ax3.bar(df.index, df['MACD'] - df['MACD_Signal'], 
                    color=['green' if x >= 0 else 'red' for x in df['MACD'] - df['MACD_Signal']], 
                    alpha=0.3)
            ax3.set_ylabel('MACD')
            ax3.legend()
            ax3.grid(True)

            # RSI
            ax4 = fig.add_subplot(gs[3])
            ax4.plot(df.index, df['RSI'], label='RSI', color='purple')
            ax4.axhline(y=70, color='r', linestyle='--', alpha=0.5)
            ax4.axhline(y=30, color='g', linestyle='--', alpha=0.5)
            ax4.set_ylabel('RSI')
            ax4.legend()
            ax4.grid(True)

            # ML Predictions subplot removed - exc_torch dependency eliminated
            # Adding earnings info subplot instead
            ax5 = fig.add_subplot(gs[4])
            
            # Show recent price trend instead of ML predictions
            recent_days = min(30, len(df))
            recent_data = df.tail(recent_days)
            ax5.plot(range(len(recent_data)), recent_data['Close'], color='blue', label='Recent Price Trend')
            
            # Add earnings line if applicable
            if earning_delta <= 30:
                ax5.axvline(x=len(recent_data) + earning_delta, color='red', label=f'Earnings in {earning_delta} days', linestyle='--')
            
            ax5.set_title(f"Recent Price Trend (30 days)")
            ax5.set_xlabel('Days')
            ax5.set_ylabel('Price')
            ax5.legend(loc='upper left')
            ax5.grid(True)

            plt.tight_layout()
            return fig
        except Exception as e:
            logger.error(f"Error plotting technical analysis: {e}")
            # Return a simple figure with error message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error generating plot: {str(e)}", 
                    ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            return fig
    
    @staticmethod
    def extract_recommendation(analysis_text: str) -> Tuple[str, str]:
        """
        Extract recommendation and confidence level from analysis text with robust pattern matching
        
        Args:
            analysis_text: The full analysis text containing the recommendation
            
        Returns:
            Tuple of (action, confidence) - Both uppercase strings
        """
        # Default values
        action = "HOLD"
        confidence = "LOW"
        
        # More robust patterns that capture any confidence level
        patterns = [
            # Format: RECOMMENDATION: ACTION (Confidence: LEVEL) - captures any confidence
            r"RECOMMENDATION:\s*(BUY|SELL|HOLD)\s*\(Confidence:\s*([A-Z-]+)\)",
            
            # Format with lowercase and flexible spacing - captures any confidence  
            r"recommendation:\s*(buy|sell|hold)\s*\(confidence:\s*([a-z-]+)\)",
            
            # Format with possible line breaks - captures any confidence
            r"recommendation:\s*(buy|sell|hold)[\s\n]*\(confidence:[\s\n]*([a-z-]+)\)",
            
            # Format without 'recommendation:' prefix - captures any confidence
            r"(buy|sell|hold)\s*\(confidence:\s*([a-z-]+)\)",
            
            # Format with different bracket styles - captures any confidence
            r"recommendation:\s*(buy|sell|hold)\s*[\(\[\{]\s*confidence:\s*([a-z-]+)\s*[\)\]\}]",
            
            # Fallback: look for FINAL RECOMMENDATION pattern
            r"FINAL\s+RECOMMENDATION:\s*(BUY|SELL|HOLD)\s*\(Confidence:\s*([A-Z-]+)\)",
        ]
        
        # Find all matches and take the last one (final recommendation)
        for pattern in patterns:
            matches = list(re.finditer(pattern, analysis_text, re.IGNORECASE))
            if matches:
                last_match = matches[-1]
                action = last_match.group(1).upper()
                confidence = last_match.group(2).upper().replace('-', '_')  # Convert hyphens to underscores
                break
        
        # Additional cleanup for confidence levels
        confidence_mapping = {
            'MEDIUM_HIGH': 'MEDIUM-HIGH',
            'MEDIUM_LOW': 'MEDIUM-LOW',
            'HIGH': 'HIGH',
            'MEDIUM': 'MEDIUM', 
            'LOW': 'LOW'
        }
        
        confidence = confidence_mapping.get(confidence, confidence)
        
        return action, confidence

    def get_market_sentiment(self) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Optional[str], Optional[str]]:
        """
        Get overall market sentiment indicators
        
        Returns:
            Tuple of (sentiment dict, VIX level, safe haven demand)
        """
        try:
            # Fetch key market indicators
            gold = yf.Ticker("GLD")  # SPDR Gold Shares ETF
            bonds = yf.Ticker("TLT")  # 20+ Year Treasury Bond ETF
            vix = yf.Ticker("^VIX")  # Volatility Index
            sp500 = yf.Ticker("^GSPC")  # S&P 500
            dollar = yf.Ticker("DX-Y.NYB")  # US Dollar Index
            
            # Get latest data
            gold_data = gold.history(period="5d")
            bonds_data = bonds.history(period="5d")
            vix_data = vix.history(period="5d")
            sp500_data = sp500.history(period="5d")
            dollar_data = dollar.history(period="5d")
            
            # Calculate 5-day trends
            sentiment = {
                'Gold': {
                    'price': gold_data['Close'][-1],
                    'change_5d': ((gold_data['Close'][-1] - gold_data['Close'][0]) / gold_data['Close'][0] * 100),
                    'trend': "rising" if gold_data['Close'][-1] > gold_data['Close'][0] else "falling"
                },
                'Bonds': {
                    'price': bonds_data['Close'][-1],
                    'change_5d': ((bonds_data['Close'][-1] - bonds_data['Close'][0]) / bonds_data['Close'][0] * 100),
                    'trend': "rising" if bonds_data['Close'][-1] > bonds_data['Close'][0] else "falling"
                },
                'VIX': {
                    'price': vix_data['Close'][-1],
                    'change_5d': ((vix_data['Close'][-1] - vix_data['Close'][0]) / vix_data['Close'][0] * 100),
                    'trend': "rising" if vix_data['Close'][-1] > vix_data['Close'][0] else "falling"
                },
                'S&P500': {
                    'price': sp500_data['Close'][-1],
                    'change_5d': ((sp500_data['Close'][-1] - sp500_data['Close'][0]) / sp500_data['Close'][0] * 100),
                    'trend': "rising" if sp500_data['Close'][-1] > sp500_data['Close'][0] else "falling"
                },
                'USD': {
                    'price': dollar_data['Close'][-1],
                    'change_5d': ((dollar_data['Close'][-1] - dollar_data['Close'][0]) / dollar_data['Close'][0] * 100),
                    'trend': "rising" if dollar_data['Close'][-1] > dollar_data['Close'][0] else "falling"
                }
            }
            
            # Determine risk sentiment
            vix_level = "high" if vix_data['Close'][-1] > 20 else "low"
            safe_haven_demand = "high" if sentiment['Gold']['trend'] == "rising" and sentiment['Bonds']['trend'] == "rising" else \
                            "low" if sentiment['Gold']['trend'] == "falling" and sentiment['Bonds']['trend'] == "falling" else "mixed"
            
            return sentiment, vix_level, safe_haven_demand
        except Exception as e:
            logger.error(f"Error fetching market sentiment: {e}")
            return None, None, None

    def get_institutional_grade_analysis(self, df: pd.DataFrame, ticker: str) -> Tuple[str, str, str]:
        """Get institutional-grade analysis using capable LLM"""
        try:
            # Calculate all advanced indicators
            df = AdvancedTechnicalAnalysis.calculate_comprehensive_indicators(df)
            
            # Detect chart patterns
            patterns = AdvancedTechnicalAnalysis.detect_chart_patterns(df)
            
            # Get market internals
            market_internals = AdvancedTechnicalAnalysis.calculate_market_internals(ticker, df)
            
            # Multi-timeframe analysis
            multi_tf_signals = MultiTimeframeAnalysis.get_multi_timeframe_signals(ticker)

            # Options flow analysis
            options_metrics = OptionsFlowAnalysis.get_options_metrics(ticker)
            
            # Get news sentiment (limit to 10 most recent for efficiency)
            articles = self.fetch_yahoo_news(ticker)[:10]
            sentiment_analysis = self.analyze_sentiment(ticker, articles)
            
            # Create comprehensive prompt
            prompt = create_advanced_analysis_prompt(
                ticker, df, sentiment_analysis, market_internals, 
                patterns, multi_tf_signals
            )
            
            # If options data available, append to prompt
            if options_metrics:
                prompt += f"""

    ## OPTIONS FLOW ANALYSIS
    Put/Call Ratio: {options_metrics['put_call_ratio']:.2f}
    Call Volume: {options_metrics['call_volume']:,}
    Put Volume: {options_metrics['put_volume']:,}
    Options Sentiment: {options_metrics['options_sentiment']}
    """
            
            # Get analysis from capable LLM
            analysis = self.llm.generate(prompt)
            action, confidence = self.extract_recommendation(analysis)
            
            return analysis, action, confidence
            
        except Exception as e:
            logger.error(f"Error in institutional-grade analysis: {e}")
            return "Analysis failed", "HOLD", "LOW"

    def get_llm_analysis(self, df: pd.DataFrame, ticker: str) -> Tuple[str, str, str]:
        """
        Wrapper method that calls the institutional-grade analysis
        """
        return self.get_institutional_grade_analysis(df, ticker)


# Integration with existing StockAnalyzer class
class EnhancedStockAnalyzer(StockAnalyzer):
    """Enhanced analyzer with prediction tracking and feedback"""
    
    def __init__(self, llm_provider: LLMProvider, config: Config = None):
        super().__init__(llm_provider, config)
        
        # Initialize tracking components
        db_path = get_root_path() / 'predictions.db'
        self.tracker = PredictionTracker(db_path)
        self.evaluator = PerformanceEvaluator(self.tracker)
        self.reflection_engine = LLMReflectionEngine(llm_provider, self.tracker)
        
    def get_llm_analysis(self, df: pd.DataFrame, ticker: str) -> Tuple[str, str, str]:
        """Get LLM analysis with historical context"""
        
        # Get base analysis
        df = AdvancedTechnicalAnalysis.calculate_comprehensive_indicators(df)
        patterns = AdvancedTechnicalAnalysis.detect_chart_patterns(df)
        market_internals = AdvancedTechnicalAnalysis.calculate_market_internals(ticker, df)
        multi_tf_signals = MultiTimeframeAnalysis.get_multi_timeframe_signals(ticker)
        
        # Get news sentiment
        articles = self.fetch_yahoo_news(ticker)[:10]
        sentiment_analysis = self.analyze_sentiment(ticker, articles)
        
        # Create base prompt
        base_prompt = create_advanced_analysis_prompt(
            ticker, df, sentiment_analysis, market_internals, 
            patterns, multi_tf_signals
        )
        
        # Enhance with historical insights
        enhanced_prompt = self.reflection_engine.get_enhanced_analysis_prompt(base_prompt, ticker)
        
        # Get analysis
        analysis = self.llm.generate(enhanced_prompt)
        action, confidence = self.extract_recommendation(analysis)
        
        # Record prediction
        latest = df.iloc[-1]
        prediction = PredictionRecord(
            ticker=ticker,
            prediction_date=datetime.now().strftime('%Y-%m-%d'),
            recommendation=action,
            confidence=confidence,
            entry_price=latest['Close'],
            target_price=None,  # Extract from analysis if available
            stop_loss=None,     # Extract from analysis if available
            predicted_timeframe='short-term',  # Extract from analysis
            technical_indicators={
                'RSI': latest['RSI'],
                'MACD': latest['MACD'],
                'MACD_Signal': latest['MACD_Signal'],
                'SMA_20': latest['SMA_20'],
                'SMA_50': latest['SMA_50'],
                'ATR': latest['ATR']
            },
            llm_analysis=analysis,
            sentiment_data=sentiment_analysis
        )
        
        self.tracker.record_prediction(prediction)
        
        return analysis, action, confidence
        
    def check_and_update_predictions(self):
        """Check open predictions and update their performance"""
        
        open_predictions = self.tracker.get_open_predictions()
        
        for pred in open_predictions:
            try:
                # Get current data
                ticker = pred['ticker']
                df = yf.Ticker(ticker).history(period='1mo')
                
                if df.empty:
                    continue
                    
                df = self.calculate_technical_indicators(df)
                current_price = df['Close'].iloc[-1]
                
                # Evaluate performance
                performance = self.evaluator.evaluate_prediction(pred, current_price, df)
                
                if performance:
                    self.tracker.update_performance(performance)
                    logger.info(f"Updated performance for {ticker}: {performance.outcome} ({performance.actual_return:.2f}%)")
                    
            except Exception as e:
                logger.error(f"Error evaluating prediction for {pred['ticker']}: {e}")
                
    def generate_performance_report(self, days: int = 30) -> str:
        """Generate a performance report for all tickers"""
        
        report = "# Prediction Performance Report\n\n"
        report += f"Period: Last {days} days\n\n"
        
        # Get list of all tickers with predictions
        conn = sqlite3.connect(self.tracker.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT ticker FROM predictions')
        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        for ticker in tickers:
            stats = self.tracker.get_performance_stats(ticker, days)
            
            if stats['total_predictions'] > 0:
                report += f"## {ticker}\n"
                report += f"- Total Predictions: {stats['total_predictions']}\n"
                report += f"- Success Rate: {stats['success_rate']:.2%}\n"
                report += f"- Average Return: {stats['avg_return']:.2f}%\n\n"
                
                # Get reflection
                reflection = self.reflection_engine.generate_reflection(ticker, days)
                report += f"### Reflection\n{reflection['reflection']}\n\n"
                report += f"### Improvements\n{reflection['improvements']}\n\n"
                
        return report
    

