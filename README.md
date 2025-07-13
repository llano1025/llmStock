# LLM Stock Analysis System

A comprehensive stock analysis system powered by Large Language Models (LLMs) that combines technical analysis, sentiment analysis, and machine learning predictions with performance tracking and reflection capabilities.

## Features

- 🤖 **Multi-LLM Support**: Ollama, LM Studio, Google Gemini, and DeepSeek
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

Copy the example configuration and customize it:

```bash
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
GEMINI_MODEL=gemini-pro

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

# Generate weekly reflection reports
python main.py --mode reflect --provider gemini

# Test prediction tracking system
python main.py --mode test

# List available providers
python main.py --list-providers
```

## LLM Provider Setup

### Option 1: Ollama (Local)

1. Install Ollama from [ollama.com](https://ollama.com/)
2. Pull a model: `ollama pull phi4`
3. Set `DEFAULT_MODEL=phi4` in `.env`

### Option 2: Google Gemini (Cloud)

1. Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Set `GEMINI_API_KEY=your-key` in `.env`

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

```bash
# Run stock analysis with performance tracking
python main.py --mode analyze [--provider PROVIDER]

# Generate weekly reflection on all tracked predictions  
python main.py --mode reflect [--provider PROVIDER]

# Test prediction tracking system with sample data
python main.py --mode test [--provider PROVIDER]

# List available LLM providers
python main.py --list-providers

# Show help
python main.py --help
```

### Provider Selection

- **`--provider auto`** (default): Auto-detect based on configuration
- **`--provider ollama`**: Use local Ollama server
- **`--provider lmstudio`**: Use local LM Studio server  
- **`--provider gemini`**: Use Google Gemini API
- **`--provider deepseek`**: Use DeepSeek API

Auto-detection priority: Gemini API key → DeepSeek API key → Ollama (default)

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
- `google-genai` (Gemini)
- `openai` (DeepSeek)

### Note on Compatibility
The system requires `numpy<2.0` due to pandas-ta compatibility. Future versions will support numpy 2.x when pandas-ta is updated.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
2. **Email Issues**: Verify Gmail app password and 2FA setup
3. **LLM Errors**: Check API keys and model availability
4. **News Scraping**: Some tickers may have limited news availability

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

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Always consult with qualified financial professionals before making investment decisions. Past performance does not guarantee future results.