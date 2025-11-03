"""
Earnings Analysis Runner

This module orchestrates the complete earnings analysis workflow,
including prediction generation for pre-earnings and post-earnings movement,
performance tracking, and reporting.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from llm.llmAnalysis import (
    EnhancedStockAnalyzer,
    Config,
    AdvancedTechnicalAnalysis,
    MultiTimeframeAnalysis
)
from llm.llmEarningsTracking import (
    EarningsPredictionTracker,
    EarningsPerformanceEvaluator,
    EarningsLLMReflectionEngine
)
from llm.llm_models import EarningsPredictionRecord
from data.fmp_data_loader import get_earnings_for_active_stocks

logger = logging.getLogger(__name__)


class EarningsAnalysisRunner:
    """Main runner for earnings analysis workflow"""

    def __init__(self, stock_analyzer: EnhancedStockAnalyzer, config: Config, email_sender):
        self.analyzer = stock_analyzer
        self.config = config
        self.email_sender = email_sender

        # Initialize earnings tracking components
        db_path = Path(config.project_root) / "earnings_predictions.db"
        self.earnings_tracker = EarningsPredictionTracker(db_path)
        self.evaluator = EarningsPerformanceEvaluator(self.earnings_tracker)
        self.reflection_engine = EarningsLLMReflectionEngine(
            self.earnings_tracker,
            self.analyzer.llm
        )

    def check_open_earnings_predictions(self) -> List[Dict[str, Any]]:
        """Check and update open earnings predictions"""
        logger.info("Checking open earnings predictions...")

        try:
            # Import data loader function
            from data.data_loader import fetch_stock_data

            updated_predictions = self.evaluator.check_and_update_predictions(fetch_stock_data)

            if updated_predictions:
                logger.info(f"Updated {len(updated_predictions)} earnings predictions")
                for update in updated_predictions:
                    logger.info(f"  {update['ticker']}: {update['outcome']}, Return: {update['total_return']:.2f}%")
            else:
                logger.info("No earnings predictions needed updating")

            return updated_predictions

        except Exception as e:
            logger.error(f"Error checking open earnings predictions: {e}")
            return []

    def run_weekly_earnings_performance_check(self) -> Dict[str, Any]:
        """Run weekly performance analysis for earnings"""
        logger.info("Running weekly earnings performance check...")

        try:
            performance_summary = self.earnings_tracker.get_performance_statistics(days=7)

            if performance_summary and performance_summary.get('total_predictions', 0) > 0:
                logger.info(f"Weekly earnings performance: "
                          f"{performance_summary.get('success_rate', 0)*100:.1f}% success rate, "
                          f"{performance_summary.get('avg_total_return', 0):.2f}% avg return")
            else:
                logger.info("No recent earnings predictions to analyze")

            return performance_summary

        except Exception as e:
            logger.error(f"Error running weekly earnings performance check: {e}")
            return {}

    def fetch_earnings_calendar_stocks(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """Fetch top N stocks by volume with upcoming earnings from FMP API"""
        logger.info(f"Fetching top {top_n} stocks by volume with upcoming earnings...")

        try:
            # Get top stocks by volume from earnings calendar
            earnings_df = get_earnings_for_active_stocks(api_key=self.config.fmp_api_key, top_n=top_n)

            if earnings_df is None or earnings_df.empty:
                logger.warning("No earnings data available for upcoming week")
                return []

            # Convert to list of dicts
            earnings_stocks = earnings_df.to_dict('records')

            logger.info(f"Found {len(earnings_stocks)} high-volume stocks with earnings in upcoming week")
            for stock in earnings_stocks[:5]:  # Log first 5
                logger.info(f"  {stock['Symbol']}: Volume={stock['Volume']:,.0f}, Earnings={stock['date']} ({stock['days_until_earnings']} days)")

            return earnings_stocks

        except Exception as e:
            logger.error(f"Error fetching earnings calendar stocks: {e}")
            return []

    def process_single_ticker_earnings(self, ticker: str, earnings_info: Dict, plot_path: Path) -> Optional[Dict[str, Any]]:
        """Process earnings analysis for a single ticker"""
        logger.info(f"Processing earnings analysis for {ticker}")

        try:
            # Fetch stock data
            df, _, _ = self.analyzer.get_stock_data(ticker, period='1y')

            if df is None or df.empty:
                logger.warning(f"No stock data available for {ticker}")
                return None

            # Calculate comprehensive technical indicators
            df = self.analyzer.calculate_technical_indicators(df)
            df = AdvancedTechnicalAnalysis.calculate_comprehensive_indicators(df)

            # Get current price
            current_price = df['Close'].iloc[-1]

            # Enhance sentiment analysis with earnings context
            earnings_context = {
                'earnings_date': str(earnings_info.get('date', '')),
                'days_until_earnings': earnings_info.get('days_until_earnings', 0),
                'eps_estimate': earnings_info.get('epsEstimated'),
                'revenue_estimate': earnings_info.get('revenueEstimated'),
                'eps_actual': earnings_info.get('epsActual'),
                'revenue_actual': earnings_info.get('revenueActual')
            }

            # Get LLM analysis with earnings context
            llm_result = self._get_earnings_analysis(df, ticker, earnings_context)

            if not llm_result:
                logger.warning(f"Could not generate earnings analysis for {ticker}")
                return None

            # Create earnings prediction record
            prediction = EarningsPredictionRecord(
                ticker=ticker,
                prediction_date=datetime.now().strftime('%Y-%m-%d'),
                earnings_date=earnings_context['earnings_date'],
                days_until_earnings=earnings_context['days_until_earnings'],
                pre_earnings_recommendation=llm_result.get('pre_earnings_recommendation', 'HOLD'),
                post_earnings_direction=llm_result.get('post_earnings_direction', 'NEUTRAL'),
                confidence=llm_result.get('confidence', 'MEDIUM'),
                entry_price=current_price,
                pre_earnings_target=llm_result.get('pre_earnings_target'),
                post_earnings_target=llm_result.get('post_earnings_target'),
                stop_loss=llm_result.get('stop_loss'),
                eps_estimate=earnings_info.get('epsEstimated'),
                eps_actual=earnings_info.get('epsActual'),  # Populated if earnings already occurred
                revenue_estimate=earnings_info.get('revenueEstimated'),
                revenue_actual=earnings_info.get('revenueActual'),  # Populated if earnings already occurred
                technical_indicators=self._extract_technical_indicators(df),
                llm_analysis=llm_result.get('analysis', ''),
                sentiment_data=llm_result.get('sentiment_data', {}),
                earnings_context=earnings_context
            )

            # Record prediction
            prediction_id = self.earnings_tracker.record_prediction(prediction)

            # Create plot with earnings date marked
            plot_file = self._create_earnings_plot(ticker, df, earnings_info, plot_path)

            result = {
                'ticker': ticker,
                'current_price': current_price,
                'earnings_date': earnings_context['earnings_date'],
                'days_until_earnings': earnings_context['days_until_earnings'],
                'prediction_id': prediction_id,
                'prediction': prediction,
                'plot_file': plot_file,
                'summary': self._create_ticker_summary(ticker, prediction, earnings_info)
            }

            logger.info(f"Generated earnings analysis for {ticker}")
            return result

        except Exception as e:
            logger.error(f"Error processing earnings for {ticker}: {e}")
            return None

    def _get_earnings_analysis(self, df, ticker: str, earnings_context: Dict) -> Optional[Dict]:
        """Get LLM analysis for earnings with comprehensive technical and sentiment analysis"""

        try:
            import json
            import re

            # Get latest data
            latest = df.iloc[-1]
            week_ago = df.iloc[-5] if len(df) > 5 else df.iloc[0]
            month_ago = df.iloc[-22] if len(df) > 22 else df.iloc[0]

            # Fetch and analyze news sentiment
            logger.info(f"Fetching news for {ticker}...")
            articles = self.analyzer.fetch_yahoo_news(ticker)[:10]  # Top 10 articles
            sentiment_analysis = self.analyzer.analyze_sentiment(ticker, articles)

            # Calculate market internals
            logger.info(f"Calculating market internals for {ticker}...")
            market_internals = AdvancedTechnicalAnalysis.calculate_market_internals(ticker, df)

            # Detect chart patterns
            logger.info(f"Detecting chart patterns for {ticker}...")
            patterns = AdvancedTechnicalAnalysis.detect_chart_patterns(df)

            # Get multi-timeframe signals
            logger.info(f"Analyzing multi-timeframe signals for {ticker}...")
            multi_tf_signals = MultiTimeframeAnalysis.get_multi_timeframe_signals(ticker)

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

            # Build actual results section if available
            actuals_section = ""
            if earnings_context.get('eps_actual') is not None:
                actuals_section = f"\n- EPS Actual: ${earnings_context['eps_actual']}"
            if earnings_context.get('revenue_actual') is not None:
                actuals_section += f"\n- Revenue Actual: ${earnings_context['revenue_actual']}"

            # Create comprehensive earnings-specific prompt
            earnings_prompt = f"""Perform an institutional-grade EARNINGS analysis for {ticker} with actionable trading recommendations.

## EARNINGS CONTEXT
- Earnings Date: {earnings_context['earnings_date']}
- Days Until Earnings: {earnings_context['days_until_earnings']}
- EPS Estimate: ${earnings_context.get('eps_estimate', 'N/A')}
- Revenue Estimate: ${earnings_context.get('revenue_estimate', 'N/A')}{actuals_section}

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

## VOLATILITY PROFILE & MARKET REGIME
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

## PIVOT LEVELS (Support/Resistance)
- R2: ${latest['R2']:.2f}
- R1: ${latest['R1']:.2f}
- Pivot: ${latest['Pivot']:.2f}
- S1: ${latest['S1']:.2f}
- S2: ${latest['S2']:.2f}

REQUIRED EARNINGS ANALYSIS:

1. **PRE-EARNINGS ANALYSIS**
   - Evaluate the technical setup leading into earnings
   - Assess momentum and positioning
   - Recommend position (BUY/SELL/HOLD) for pre-earnings period
   - Provide pre-earnings target price based on technical levels and sentiment
   - Consider volatility expansion typical before earnings

2. **POST-EARNINGS DIRECTION PREDICTION**
   - Predict direction after earnings announcement (UP/DOWN/NEUTRAL)
   - Provide post-earnings target price
   - Identify key factors that could drive the move (technical, sentiment, expectations)
   - Assess potential for earnings surprise based on sentiment and positioning

3. **RISK MANAGEMENT**
   - Suggest stop loss levels based on technical support/resistance
   - Identify key risks to watch (technical breakdown, sentiment reversal, macro factors)
   - Provide overall confidence level (HIGH/MEDIUM/LOW) based on:
     * Signal confluence across indicators
     * Sentiment alignment with technical setup
     * Market regime and volatility considerations
     * Historical earnings behavior patterns
   - Specify risk level (HIGH/MEDIUM/LOW)

4. **MARKET CONTEXT**
   - Consider current market regime (trending/ranging/volatile)
   - Evaluate institutional positioning via volume patterns
   - Assess how broader market correlations may impact post-earnings move

STRUCTURED OUTPUT:
Provide your analysis followed by this EXACT JSON format:

```json
{{
    "pre_earnings_recommendation": "BUY/SELL/HOLD",
    "post_earnings_direction": "UP/DOWN/NEUTRAL",
    "confidence": "HIGH/MEDIUM/LOW",
    "pre_earnings_target": <float or null>,
    "post_earnings_target": <float or null>,
    "stop_loss": <float or null>,
    "analysis": "detailed analysis text incorporating technical setup, sentiment, and earnings context",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risk_level": "HIGH/MEDIUM/LOW"
}}
```

Ensure all numeric values are provided as numbers, not strings. Use null for missing values."""

            # Get LLM response
            logger.info(f"Generating LLM earnings analysis for {ticker}...")
            response = self.analyzer.llm.generate(earnings_prompt)

            # Parse JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                # Add sentiment data to result for storage
                result['sentiment_data'] = {
                    'overall_sentiment': sentiment_analysis['overall_sentiment'],
                    'confidence': sentiment_analysis['confidence'],
                    'price_impact': sentiment_analysis['price_impact'],
                    'articles_processed': sentiment_analysis['articles_processed']
                }
                return result
            else:
                logger.warning(f"Could not parse JSON from LLM response for {ticker}")
                return None

        except Exception as e:
            logger.error(f"Error getting earnings analysis: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_technical_indicators(self, df) -> Dict[str, float]:
        """Extract key technical indicators from DataFrame"""
        try:
            return {
                'rsi': float(df['RSI'].iloc[-1]) if 'RSI' in df.columns else 0.0,
                'macd': float(df['MACD'].iloc[-1]) if 'MACD' in df.columns else 0.0,
                'macd_signal': float(df['MACD_Signal'].iloc[-1]) if 'MACD_Signal' in df.columns else 0.0,
                'ma20': float(df['EMA_20'].iloc[-1]) if 'MA20' in df.columns else 0.0,
                'ma50': float(df['EMA_50'].iloc[-1]) if 'MA50' in df.columns else 0.0,
                'volume': float(df['Volume'].iloc[-1]) if 'Volume' in df.columns else 0.0,
                'atr': float(df['ATR'].iloc[-1]) if 'ATR' in df.columns else 0.0
            }
        except Exception as e:
            logger.error(f"Error extracting technical indicators: {e}")
            return {}

    def _create_ticker_summary(self, ticker: str, prediction: EarningsPredictionRecord,
                              earnings_info: Dict) -> str:
        """Create a summary of earnings prediction for a ticker"""
        summary = f"{ticker} Earnings Analysis:\n"
        summary += f"- Earnings Date: {prediction.earnings_date} ({prediction.days_until_earnings} days)\n"
        summary += f"- Pre-Earnings: {prediction.pre_earnings_recommendation}\n"
        summary += f"- Post-Earnings: {prediction.post_earnings_direction}\n"
        summary += f"- Confidence: {prediction.confidence}\n"

        if prediction.eps_estimate:
            summary += f"- EPS Estimate: ${prediction.eps_estimate:.2f}\n"

        if prediction.pre_earnings_target:
            summary += f"- Pre-Earnings Target: ${prediction.pre_earnings_target:.2f}\n"

        if prediction.post_earnings_target:
            summary += f"- Post-Earnings Target: ${prediction.post_earnings_target:.2f}\n"

        return summary

    def _create_earnings_plot(self, ticker: str, df, earnings_info: Dict, plot_path: Path) -> str:
        """Create visualization for earnings analysis with earnings date marked"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

            # Plot 1: Price chart with MAs and earnings date
            ax1.plot(df.index, df['Close'], label='Close Price', color='black', linewidth=2)
            ax1.plot(df.index, df['EMA_20'], label='20-day MA', color='blue', alpha=0.7)
            ax1.plot(df.index, df['EMA_50'], label='50-day MA', color='red', alpha=0.7)

            # Mark earnings date
            earnings_date = earnings_info.get('date')
            if earnings_date:
                ax1.axvline(x=earnings_date, color='green', linestyle='--', linewidth=2,
                          label=f'Earnings: {earnings_date}', alpha=0.7)

            ax1.set_title(f'{ticker} - Earnings Analysis ({df.index[0].date()} to {df.index[-1].date()})')
            ax1.set_ylabel('Price ($)')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)

            # Plot 2: Volume
            colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red'
                     for i in range(len(df))]
            ax2.bar(df.index, df['Volume'], color=colors, alpha=0.5)
            ax2.set_ylabel('Volume')
            ax2.grid(True, alpha=0.3)

            # Plot 3: RSI
            ax3.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=2)
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
            ax3.set_ylabel('RSI')
            ax3.set_xlabel('Date')
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_file = plot_path / f"{ticker}_earnings_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            plt.close(fig)

            return str(plot_file)

        except Exception as e:
            logger.error(f"Error creating earnings plot for {ticker}: {e}")
            return None

    def cleanup_plot_files(self, plot_files: List[str]):
        """Clean up generated plot files"""
        for plot_file in plot_files:
            try:
                if plot_file and os.path.exists(plot_file):
                    os.remove(plot_file)
                    logger.debug(f"Removed plot file: {plot_file}")
            except Exception as e:
                logger.warning(f"Could not remove plot file {plot_file}: {e}")


def run_earnings_analysis(provider_type='auto'):
    """Main function to run complete earnings analysis"""
    try:
        logger.info("Starting comprehensive earnings analysis...")

        # Import necessary components
        from llm.llmAnalysis import Config
        from utils.utils_report import EmailSender
        from utils.utils_path import get_plot_path

        # Initialize configuration
        config = Config()

        # Verify FMP API key
        if not config.fmp_api_key or config.fmp_api_key == 'your_api_key_here':
            logger.error("FMP_API_KEY not set in .env file. Please add your FMP API key.")
            return

        if not config.sender_email or not config.sender_password or not config.recipient_emails:
            logger.error("Email credentials are not set in .env file")
            return

        # Create LLM provider
        from main import create_llm_provider, create_secondary_llm_provider
        llm_provider = create_llm_provider(config)
        secondary_llm_provider = create_secondary_llm_provider(config)

        # Create analyzer
        analyzer = EnhancedStockAnalyzer(llm_provider, config, secondary_llm_provider)

        # Initialize email sender
        email_sender = EmailSender(
            smtp_server=config.smtp_server,
            smtp_port=config.smtp_port,
            sender_email=config.sender_email,
            sender_password=config.sender_password
        )

        # Create runner
        runner = EarningsAnalysisRunner(analyzer, config, email_sender)

        # Execute workflow
        runner.check_open_earnings_predictions()
        performance_summary = runner.run_weekly_earnings_performance_check()

        # Fetch top 20 stocks by volume from earnings calendar
        earnings_stocks = runner.fetch_earnings_calendar_stocks(top_n=20)

        if not earnings_stocks:
            logger.warning("No stocks with upcoming earnings found")
            return

        # Setup plot directory
        plt_path = get_plot_path()
        os.makedirs(plt_path, exist_ok=True)

        # Process each ticker with earnings (already filtered to top 20 by volume)
        analysis_results = []
        plot_files = []

        for earnings_info in earnings_stocks:
            ticker = earnings_info['Symbol']
            result = runner.process_single_ticker_earnings(ticker, earnings_info, plt_path)
            if result:
                analysis_results.append(result)
                if result['plot_file']:
                    plot_files.append(result['plot_file'])

        # Send report
        if analysis_results:
            email_sender.send_earnings_report(
                recipient_emails=config.recipient_emails,
                subject=f"Earnings Analysis Report - {len(analysis_results)} Stocks with Upcoming Earnings",
                analysis_results=analysis_results,
                plot_files=plot_files,
                performance_summary=performance_summary
            )
        else:
            logger.warning("No earnings analysis results generated")

        # Cleanup
        runner.cleanup_plot_files(plot_files)
        plt.close('all')

        logger.info("Earnings analysis completed successfully")

    except Exception as e:
        logger.critical(f"Critical error in earnings analysis: {e}")
        import traceback
        traceback.print_exc()


def run_earnings_reflection():
    """Generate reflection on earnings prediction performance"""
    try:
        logger.info("Starting earnings reflection analysis...")

        # Import necessary components
        from llm.llmAnalysis import Config
        from utils.utils_report import EmailSender

        # Initialize configuration
        config = Config()

        if not config.sender_email or not config.sender_password or not config.recipient_emails:
            logger.error("Email credentials are not set in .env file")
            return

        # Create LLM provider
        from main import create_llm_provider
        llm_provider = create_llm_provider(config)

        # Initialize tracker and reflection engine
        db_path = Path(config.project_root) / "earnings_predictions.db"
        tracker = EarningsPredictionTracker(db_path)
        reflection_engine = EarningsLLMReflectionEngine(tracker, llm_provider)

        # Generate reflection
        reflection = reflection_engine.generate_reflection(days=90)

        # Get performance statistics
        stats = tracker.get_performance_statistics(days=90)
        beat_miss_stats = tracker.get_performance_by_beat_miss(days=90)

        # Create email sender and send reflection report
        email_sender = EmailSender(
            smtp_server=config.smtp_server,
            smtp_port=config.smtp_port,
            sender_email=config.sender_email,
            sender_password=config.sender_password
        )

        email_sender.send_earnings_reflection_report(
            recipient_emails=config.recipient_emails,
            subject="Earnings Prediction Performance Reflection",
            reflection_text=reflection,
            stats=stats,
            beat_miss_stats=beat_miss_stats,
            days=90
        )

        logger.info("Earnings reflection report sent successfully")

    except Exception as e:
        logger.critical(f"Critical error in earnings reflection: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run earnings analysis
    run_earnings_analysis()
