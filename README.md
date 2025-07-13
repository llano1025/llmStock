# LLM Stock Analysis System

A comprehensive stock analysis system powered by Large Language Models (LLMs) that combines technical analysis, sentiment analysis, and machine learning predictions with performance tracking and reflection capabilities.

## 🆕 What's New

- **Enhanced Provider Selection**: Command-line provider switching with `--provider` option
- **Updated Gemini API**: Now uses latest `google-genai` SDK with `gemini-2.5-flash` model
- **Auto-Detection**: Intelligent provider selection based on available API keys
- **Improved CLI**: Better help system with `--list-providers` and `--help` options
- **Robust Fallbacks**: Automatic fallback to Ollama if selected provider fails

## Features

- 🤖 **Multi-LLM Support**: Ollama, LM Studio, Google Gemini (2.5-flash), and DeepSeek with auto-detection
- 🔄 **Provider Selection**: Command-line provider switching with automatic fallback
- 📊 **Advanced Technical Analysis**: 20+ indicators, pattern detection, multi-timeframe analysis
- 📰 **News Sentiment Analysis**: Yahoo Finance news with relevance filtering
- 📈 **Performance Tracking**: SQLite-based prediction tracking with outcome evaluation
- 🧠 **Reflection System**: AI-powered performance analysis and improvement suggestions
- 📧 **Email Reports**: Automated HTML reports with charts and analysis
- 🎯 **Institutional-Grade Analysis**: Professional-level prompts and comprehensive evaluation

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd llmStock

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create your configuration file:

```bash
# The system will create a template if .env doesn't exist
python main.py  # Creates .env.template with all options

# Or copy the example if available
cp .env.example .env
```

Edit `.env` with your settings:

```env
# Email settings (Required)
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-app-password
RECIPIENT_EMAILS=recipient@example.com

# Choose at least one LLM provider:

# Option 1: Ollama (Local)
OLLAMA_HOST=http://localhost:11434
DEFAULT_MODEL=phi4

# Option 2: Google Gemini (Cloud)
GEMINI_API_KEY=your-api-key
GEMINI_MODEL=gemini-2.5-flash

# Option 3: DeepSeek (Cloud)
DEEPSEEK_API_KEY=your-api-key
DEEPSEEK_MODEL=deepseek-chat

# Option 4: LM Studio (Local)
LMSTUDIO_HOST=http://localhost:1234/v1
```

### 3. Run Analysis

```bash
# Run stock analysis with auto-detected provider (default)
python main.py

# Run with specific LLM provider
python main.py --provider gemini
python main.py --provider ollama
python main.py --provider deepseek
python main.py --provider lmstudio

# Generate weekly reflection reports
python main.py --mode reflect --provider gemini

# Test prediction tracking system
python main.py --mode test --provider auto

# List available providers and configuration help
python main.py --list-providers

# Show detailed help
python main.py --help
```

## LLM Provider System

The system supports multiple LLM providers with automatic detection and fallback capabilities. Choose the provider that best fits your needs:

- **Local Providers**: Ollama, LM Studio (no API costs, requires local setup)
- **Cloud Providers**: Google Gemini, DeepSeek (API costs, no local setup)

### Auto-Detection Logic

When using `--provider auto` (default), the system automatically selects the best available provider:

1. **Gemini**: If `GEMINI_API_KEY` is configured
2. **DeepSeek**: If `DEEPSEEK_API_KEY` is configured  
3. **Ollama**: Default fallback (assumes local Ollama installation)

### Error Handling

- If the selected provider fails to initialize, the system automatically falls back to Ollama
- Clear error messages guide you to fix configuration issues
- All provider failures are logged for debugging

## LLM Provider Setup

### Option 1: Ollama (Local)

1. Install Ollama from [ollama.com](https://ollama.com/)
2. Pull a model: `ollama pull phi4`
3. Set `DEFAULT_MODEL=phi4` in `.env`

### Option 2: Google Gemini (Cloud)

1. Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Set `GEMINI_API_KEY=your-key` in `.env`
3. Uses latest `gemini-2.5-flash` model with optimized thinking configuration

### Option 3: DeepSeek (Cloud)

1. Get API key from [DeepSeek Platform](https://platform.deepseek.com/api_keys)
2. Set `DEEPSEEK_API_KEY=your-key` in `.env`

### Option 4: LM Studio (Local)

1. Install LM Studio from [lmstudio.ai](https://lmstudio.ai/)
2. Load a model and start the server
3. Set `LMSTUDIO_HOST=http://localhost:1234/v1` in `.env`

## Email Setup (Gmail)

1. Enable 2-Factor Authentication on your Gmail account
2. Generate an App Password at [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
3. Use the app password (not your regular password) in `SENDER_PASSWORD`

## Architecture

### Core Components

```
llmStock/
├── main.py                 # CLI entry point
├── llm/                    # LLM analysis modules
│   ├── llmAnalysis.py      # Core analysis engine
│   ├── llmTracking.py      # Performance tracking
│   ├── llm_models.py       # Data models
│   ├── stock_analysis_runner.py  # Business logic
│   └── reflection_runner.py      # Reflection system
├── data/
│   └── data_loader.py      # Stock data fetching
├── utils/
│   ├── utils_path.py       # Path management
│   └── utils_report.py     # Email reporting
└── requirements.txt        # Dependencies
```

### Key Classes

- **`StockAnalyzer`**: Base analyzer with LLM integration
- **`EnhancedStockAnalyzer`**: Extended analyzer with tracking
- **`PredictionTracker`**: SQLite-based prediction storage
- **`LLMReflectionEngine`**: Performance analysis and insights
- **`StockAnalysisRunner`**: Main workflow orchestration

## Technical Analysis Features

### Advanced Indicators
- Market microstructure (VWAP, price spread, close location)
- Multiple volatility measures (Parkinson, Garman-Klass)
- Volume profile analysis
- Market regime detection (ranging/developing/trending)
- 20+ standard technical indicators (RSI, MACD, Bollinger Bands, etc.)

### Pattern Detection
- Double tops/bottoms
- Triangle patterns (ascending, descending, symmetrical)
- Channel formations
- Head and shoulders (planned)

### Multi-timeframe Analysis
- Intraday (5 days)
- Short-term (1 month)
- Medium-term (3 months)
- Long-term (1 year)

## Prediction Tracking

The system automatically tracks all predictions and evaluates their performance:

1. **Prediction Recording**: Every LLM analysis creates a prediction record
2. **Performance Monitoring**: Daily checks update prediction outcomes
3. **Reflection**: Weekly AI-powered analysis of performance patterns
4. **Improvement**: Historical insights incorporated into future analysis

### Database Schema

```sql
-- Prediction records
predictions: id, ticker, date, recommendation, confidence, entry_price, 
            target_price, stop_loss, timeframe, technical_indicators, 
            llm_analysis, sentiment_data

-- Performance outcomes  
performance: prediction_id, ticker, outcome_date, exit_price, 
            actual_return, predicted_return, days_held, outcome, 
            market_condition

-- AI reflections
llm_feedback: ticker, analysis_date, performance_summary, 
             improvement_suggestions, confidence_analysis
```

## News Sentiment Analysis

- **Source**: Yahoo Finance news with session warm-up for high-traffic tickers
- **Processing**: Multi-threaded with relevance filtering
- **Analysis**: Sentiment scoring, confidence assessment, key event extraction
- **Integration**: Incorporated into technical analysis for comprehensive view

## Configuration Options

### Model Settings
- `DEFAULT_MODEL`: Primary LLM model name
- `PREDICT_WINDOW`: Prediction timeframe (default: 48 hours)
- `DATA_WINDOW`: Historical data window (default: 96 periods)
- `ALGORITHM`: Analysis algorithm ID (default: 7)

### Path Settings
- `PROJECT_ROOT`: Project base directory (auto-detected if empty)
- `SAVE_PATH`: Output directory for reports and data

## Command Line Interface

### Help and Discovery

```bash
# Show detailed help with all options
python main.py --help

# List all available providers with setup instructions
python main.py --list-providers
```

### Basic Usage

```bash
# Run stock analysis with performance tracking
python main.py --mode analyze [--provider PROVIDER]

# Generate weekly reflection on all tracked predictions  
python main.py --mode reflect [--provider PROVIDER]

# Test prediction tracking system with sample data
python main.py --mode test [--provider PROVIDER]

# Show available providers with configuration examples
python main.py --list-providers

# Show detailed help
python main.py --help
```

### Provider Selection

| Provider | Description | Requirements |
|----------|-------------|-------------|
| `auto` | Auto-detect based on configuration (default) | At least one provider configured |
| `ollama` | Local Ollama server | Ollama running on localhost:11434 |
| `lmstudio` | Local LM Studio server | LM Studio running on localhost:1234 |
| `gemini` | Google Gemini API (gemini-2.5-flash) | Valid GEMINI_API_KEY |
| `deepseek` | DeepSeek API | Valid DEEPSEEK_API_KEY |

**Auto-detection priority**: Gemini API key → DeepSeek API key → Ollama (default)

### Advanced Examples

```bash
# Run analysis with specific provider
python main.py --mode analyze --provider gemini

# Generate reflection with cloud provider
python main.py --mode reflect --provider deepseek

# Test with local Ollama
python main.py --mode test --provider ollama

# Auto-detect provider (checks API keys, falls back to Ollama)
python main.py --mode analyze --provider auto
```

## Output

### Email Reports
- **Summary Table**: All analyzed stocks with recommendations
- **Detailed Analysis**: Individual stock breakdowns with technical charts
- **Performance Summary**: Recent prediction success rates
- **Interactive HTML**: Clickable navigation between sections

### Performance Reports
- **Weekly Reflections**: AI analysis of prediction accuracy
- **Improvement Suggestions**: Specific recommendations for better results
- **Historical Trends**: Long-term performance patterns

## Dependencies

### Core Requirements
- Python 3.8+
- pandas, numpy, matplotlib
- yfinance, yahooquery (market data)
- ta, pandas-ta (technical analysis)
- requests, beautifulsoup4 (news scraping)

### LLM Providers (Optional)
- `google-genai>=0.3.0` (Gemini with new SDK)
- `openai>=1.0.0` (DeepSeek - OpenAI compatible)

### Note on Compatibility
- **NumPy**: Requires `numpy<2.0` due to pandas-ta compatibility
- **Gemini API**: Updated to use new `google-genai` SDK with improved performance
- **Python**: Requires Python 3.8+ for full compatibility

## Troubleshooting

### Common Issues

1. **Import Errors**: 
   - Run `pip install -r requirements.txt` to install all dependencies
   - Check that `pandas_ta` and `google-genai` are properly installed

2. **Provider Issues**:
   - **Gemini**: Verify API key at [Google AI Studio](https://aistudio.google.com/app/apikey)
   - **DeepSeek**: Check API key at [DeepSeek Platform](https://platform.deepseek.com/api_keys)
   - **Ollama**: Ensure Ollama is running (`ollama serve`) and model is pulled
   - **LM Studio**: Verify server is running with a model loaded

3. **Email Issues**: 
   - Use Gmail app passwords (not regular password)
   - Enable 2FA and generate app password

4. **Auto-detection**: 
   - System falls back to Ollama if no API keys are configured
   - Check `.env` file for proper API key configuration

5. **News Scraping**: Some tickers may have limited news availability

### Logs
- Application logs: `stock_analyzer.log`
- LLM module logs: `llm/stock_analyzer.log`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Getting Started Checklist

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Configure at least one LLM provider (see setup sections above)
3. ✅ Set up email credentials in `.env` file
4. ✅ Test provider selection: `python main.py --list-providers`
5. ✅ Run your first analysis: `python main.py --provider auto`

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Always consult with qualified financial professionals before making investment decisions. Past performance does not guarantee future results.