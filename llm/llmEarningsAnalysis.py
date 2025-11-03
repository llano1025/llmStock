"""
Earnings Analysis Module

This module provides specialized earnings analysis functionality,
including pre-earnings and post-earnings predictions with comprehensive
technical analysis and sentiment integration.
"""

import logging
import json
import re
from typing import Dict, Optional
from pathlib import Path

from .llmAnalysis import EnhancedStockAnalyzer, AdvancedTechnicalAnalysis, MultiTimeframeAnalysis
from .llmEarningsTracking import EarningsPredictionTracker

logger = logging.getLogger(__name__)


class EarningsAnalyzer(EnhancedStockAnalyzer):
    """
    Specialized analyzer for earnings events with dual prediction structure:
    - Pre-earnings recommendation (BUY/SELL/HOLD)
    - Post-earnings direction prediction (UP/DOWN/NEUTRAL)
    """

    def __init__(self, llm_provider, config, secondary_llm_provider=None, tracker=None):
        """
        Initialize the EarningsAnalyzer

        Args:
            llm_provider: Primary LLM provider for analysis
            config: Configuration settings
            secondary_llm_provider: Optional secondary LLM for lightweight tasks
            tracker: Optional EarningsPredictionTracker instance (will create if not provided)
        """
        super().__init__(llm_provider, config, secondary_llm_provider)

        # Initialize earnings tracker if not provided
        if tracker is None:
            db_path = Path(config.project_root) / "earnings_predictions.db"
            self.earnings_tracker = EarningsPredictionTracker(db_path)
        else:
            self.earnings_tracker = tracker

    def get_earnings_llm_analysis(self, df, ticker: str, earnings_context: Dict) -> Optional[Dict]:
        """
        Generate comprehensive earnings analysis with LLM integration

        Includes technical indicators, news sentiment, market internals,
        chart patterns, and multi-timeframe analysis specifically tailored
        for earnings events.

        Args:
            df: DataFrame with stock data and technical indicators
            ticker: Stock ticker symbol
            earnings_context: Dict containing:
                - earnings_date: Date of earnings announcement
                - days_until_earnings: Days until earnings
                - eps_estimate: EPS estimate (optional)
                - revenue_estimate: Revenue estimate (optional)
                - eps_actual: Actual EPS if earnings occurred (optional)
                - revenue_actual: Actual revenue if earnings occurred (optional)

        Returns:
            Dict with analysis results or None if error:
                - pre_earnings_recommendation: BUY/SELL/HOLD
                - post_earnings_direction: UP/DOWN/NEUTRAL
                - confidence: HIGH/MEDIUM/LOW
                - pre_earnings_target: float or None
                - post_earnings_target: float or None
                - stop_loss: float or None
                - analysis: detailed analysis text
                - key_factors: list of strings
                - risk_level: HIGH/MEDIUM/LOW
                - sentiment_data: dict with sentiment analysis results
        """
        try:
            # Get latest data
            latest = df.iloc[-1]
            week_ago = df.iloc[-5] if len(df) > 5 else df.iloc[0]
            month_ago = df.iloc[-22] if len(df) > 22 else df.iloc[0]

            # 1. Fetch and analyze news sentiment
            logger.info(f"Fetching news for {ticker}...")
            articles = self.fetch_yahoo_news(ticker)[:10]  # Top 10 articles
            sentiment_analysis = self.analyze_sentiment(ticker, articles)

            # 2. Calculate market internals
            logger.info(f"Calculating market internals for {ticker}...")
            market_internals = AdvancedTechnicalAnalysis.calculate_market_internals(ticker, df)

            # 3. Detect chart patterns
            logger.info(f"Detecting chart patterns for {ticker}...")
            patterns = AdvancedTechnicalAnalysis.detect_chart_patterns(df)

            # 4. Get multi-timeframe signals
            logger.info(f"Analyzing multi-timeframe signals for {ticker}...")
            multi_tf_signals = MultiTimeframeAnalysis.get_multi_timeframe_signals(ticker)

            # 5. Create earnings-specific prompt
            earnings_prompt = self._create_earnings_analysis_prompt(
                df, ticker, earnings_context, latest, week_ago, month_ago,
                sentiment_analysis, market_internals, patterns, multi_tf_signals
            )

            # 6. Get LLM response
            logger.info(f"Generating LLM earnings analysis for {ticker}...")
            response = self.llm.generate(earnings_prompt)

            # 7. Parse JSON from response
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

    def _create_earnings_analysis_prompt(self, df, ticker, earnings_context, latest,
                                        week_ago, month_ago, sentiment_analysis,
                                        market_internals, patterns, multi_tf_signals) -> str:
        """
        Create comprehensive earnings-specific LLM prompt

        Args:
            df: DataFrame with stock data
            ticker: Stock ticker
            earnings_context: Dict with earnings information
            latest: Latest row from DataFrame
            week_ago, month_ago: Historical rows for comparison
            sentiment_analysis: News sentiment results
            market_internals: Market correlation data
            patterns: Detected chart patterns
            multi_tf_signals: Multi-timeframe analysis results

        Returns:
            Formatted prompt string
        """
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

        return earnings_prompt

    def extract_earnings_technical_indicators(self, df) -> Dict[str, float]:
        """
        Extract key technical indicators from DataFrame for earnings tracking

        Args:
            df: DataFrame with calculated technical indicators

        Returns:
            Dict with indicator names and values
        """
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
