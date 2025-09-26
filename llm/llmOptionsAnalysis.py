"""
Enhanced Options Analysis System with LLM Integration

This module extends the stock analysis framework to provide comprehensive options
analysis including Greeks calculation, volatility analysis, and LLM-driven
call/put predictions with performance tracking.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import norm
from pathlib import Path

# Import base classes
from .llmAnalysis import EnhancedStockAnalyzer, Config, LLMProvider
from .llmTracking import PredictionTracker, PerformanceEvaluator, LLMReflectionEngine
from .llm_models import (
    OptionsPredictionRecord, OptionsPerformanceRecord, OptionsLLMFeedback,
    PredictionRecord, PerformanceRecord
)

logger = logging.getLogger(__name__)


class OptionsGreeksCalculator:
    """Calculate options Greeks using Black-Scholes model"""

    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> Dict[str, float]:
        """
        Calculate options Greeks using Black-Scholes formula

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            sigma: Implied volatility
            option_type: 'CALL' or 'PUT'

        Returns:
            Dictionary containing delta, gamma, theta, vega, rho
        """
        try:
            if T <= 0:
                return {
                    'delta': 1.0 if option_type == 'CALL' and S > K else 0.0,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0,
                    'rho': 0.0
                }

            # Calculate d1 and d2
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)

            # Standard normal CDF and PDF
            N_d1 = norm.cdf(d1)
            N_d2 = norm.cdf(d2)
            n_d1 = norm.pdf(d1)

            if option_type.upper() == 'CALL':
                delta = N_d1
                rho = K * T * np.exp(-r * T) * N_d2 / 100
            else:  # PUT
                delta = N_d1 - 1
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

            # Greeks that are same for calls and puts
            gamma = n_d1 / (S * sigma * np.sqrt(T))
            theta = (-S * n_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2) / 365
            vega = S * n_d1 * np.sqrt(T) / 100

            return {
                'delta': round(delta, 4),
                'gamma': round(gamma, 4),
                'theta': round(theta, 4),
                'vega': round(vega, 4),
                'rho': round(rho, 4)
            }

        except Exception as e:
            logger.warning(f"Greeks calculation failed: {e}")
            return {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}


class OptionsDataFetcher:
    """Fetch and process options chain data"""

    @staticmethod
    def get_options_chain(ticker: str, expiration_days: List[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch options chain data for specified expiration timeframes

        Args:
            ticker: Stock symbol
            expiration_days: List of days to expiration (default: [7, 14, 21, 28, 56, 84, 175])

        Returns:
            Dictionary with expiration dates as keys and options dataframes as values
        """
        if expiration_days is None:
            expiration_days = [7, 14, 21, 28, 56, 84, 175]

        try:
            stock = yf.Ticker(ticker)
            options_dates = stock.options

            if not options_dates:
                logger.warning(f"No options data available for {ticker}")
                return {}

            current_date = datetime.now().date()
            target_dates = []

            # Find closest expiration dates to our target days
            for target_days in expiration_days:
                target_date = current_date + timedelta(days=target_days)
                closest_date = min(options_dates,
                                 key=lambda x: abs((datetime.strptime(x, '%Y-%m-%d').date() - target_date).days))
                if closest_date not in target_dates:
                    target_dates.append(closest_date)

            options_data = {}
            for exp_date in target_dates:
                try:
                    option_chain = stock.option_chain(exp_date)

                    # Combine calls and puts with type indicator
                    calls = option_chain.calls.copy()
                    calls['option_type'] = 'CALL'
                    puts = option_chain.puts.copy()
                    puts['option_type'] = 'PUT'

                    combined = pd.concat([calls, puts], ignore_index=True)
                    combined['expiration'] = exp_date
                    combined['days_to_expiration'] = (datetime.strptime(exp_date, '%Y-%m-%d').date() - current_date).days

                    options_data[exp_date] = combined

                except Exception as e:
                    logger.warning(f"Failed to fetch options for {ticker} {exp_date}: {e}")
                    continue

            return options_data

        except Exception as e:
            logger.error(f"Failed to fetch options chain for {ticker}: {e}")
            return {}

    @staticmethod
    def analyze_options_flow(options_df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Analyze options volume and open interest patterns

        Args:
            options_df: Options chain dataframe
            current_price: Current stock price

        Returns:
            Dictionary with flow analysis metrics
        """
        if options_df.empty:
            return {}

        try:
            # Separate calls and puts
            calls = options_df[options_df['option_type'] == 'CALL']
            puts = options_df[options_df['option_type'] == 'PUT']

            # Calculate key metrics
            total_call_volume = calls['volume'].fillna(0).sum()
            total_put_volume = puts['volume'].fillna(0).sum()
            total_call_oi = calls['openInterest'].fillna(0).sum()
            total_put_oi = puts['openInterest'].fillna(0).sum()

            put_call_volume_ratio = total_put_volume / max(total_call_volume, 1)
            put_call_oi_ratio = total_put_oi / max(total_call_oi, 1)

            # Find max pain (strike with highest total open interest)
            strike_oi = options_df.groupby('strike')[['openInterest']].sum()
            max_pain_strike = strike_oi.idxmax()['openInterest'] if not strike_oi.empty else current_price

            # ITM vs OTM analysis
            call_itm_volume = calls[calls['strike'] <= current_price]['volume'].fillna(0).sum()
            call_otm_volume = calls[calls['strike'] > current_price]['volume'].fillna(0).sum()
            put_itm_volume = puts[puts['strike'] >= current_price]['volume'].fillna(0).sum()
            put_otm_volume = puts[puts['strike'] < current_price]['volume'].fillna(0).sum()

            return {
                'put_call_volume_ratio': round(put_call_volume_ratio, 3),
                'put_call_oi_ratio': round(put_call_oi_ratio, 3),
                'total_call_volume': int(total_call_volume),
                'total_put_volume': int(total_put_volume),
                'max_pain_strike': float(max_pain_strike),
                'call_itm_volume': int(call_itm_volume),
                'call_otm_volume': int(call_otm_volume),
                'put_itm_volume': int(put_itm_volume),
                'put_otm_volume': int(put_otm_volume)
            }

        except Exception as e:
            logger.warning(f"Options flow analysis failed: {e}")
            return {}


class OptionsAnalyzer(EnhancedStockAnalyzer):
    """Enhanced options analyzer with LLM integration and performance tracking"""

    def __init__(self, llm_provider: LLMProvider, config: Config = None):
        super().__init__(llm_provider, config)
        self.greeks_calculator = OptionsGreeksCalculator()
        self.data_fetcher = OptionsDataFetcher()

        # Initialize options tracking
        try:
            from utils.utils_path import get_project_root
            db_path = get_project_root() / 'options_predictions.db'
        except ImportError:
            # Fallback if utils.utils_path is not available
            db_path = Path.cwd() / 'options_predictions.db'
        self.options_tracker = OptionsTracker(db_path)

    def get_best_options_candidates(self, options_data: Dict[str, pd.DataFrame],
                                  current_price: float, technical_indicators: Dict[str, float],
                                  sentiment_score: float, ticker: str = None, sentiment_data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Identify the most promising options candidates based on technical analysis

        Args:
            options_data: Dictionary of options chains by expiration
            current_price: Current stock price
            technical_indicators: Technical analysis results
            sentiment_score: News sentiment score

        Returns:
            List of promising options candidates
        """
        candidates = []

        # Get trend direction from technical indicators
        rsi = technical_indicators.get('RSI_14', 50)
        macd_signal = technical_indicators.get('MACD_signal', 'NEUTRAL')
        bb_position = technical_indicators.get('bb_position', 0.5)  # 0-1 scale within Bollinger Bands

        # Determine directional bias
        bullish_signals = 0
        bearish_signals = 0

        if rsi < 30:
            bullish_signals += 1
        elif rsi > 70:
            bearish_signals += 1

        if macd_signal == 'BUY':
            bullish_signals += 1
        elif macd_signal == 'SELL':
            bearish_signals += 1

        if sentiment_score > 0.1:
            bullish_signals += 1
        elif sentiment_score < -0.1:
            bearish_signals += 1

        if bb_position < 0.2:
            bullish_signals += 1
        elif bb_position > 0.8:
            bearish_signals += 1

        # Determine primary bias
        if bullish_signals > bearish_signals:
            primary_bias = 'BULLISH'
            secondary_bias = 'BEARISH'
        elif bearish_signals > bullish_signals:
            primary_bias = 'BEARISH'
            secondary_bias = 'BULLISH'
        else:
            primary_bias = 'NEUTRAL'
            secondary_bias = 'NEUTRAL'

        # Analyze each expiration
        for exp_date, df in options_data.items():
            if df.empty:
                continue

            days_to_exp = df.iloc[0]['days_to_expiration']

            # Skip very short-term options (< 5 days) to avoid extreme time decay
            if days_to_exp < 5:
                continue

            # Apply progressive filtering to find liquid options
            original_count = len(df)
            logger.debug(f"Expiration {exp_date}: Starting with {original_count} options")

            # Apply progressive liquidity filtering with more lenient criteria

            # Level 1: Strict filtering for high-liquidity options
            strict_df = df[(df['volume'].fillna(0) > 10) & (df['openInterest'].fillna(0) > 50)]
            if not strict_df.empty:
                df = strict_df
                logger.debug(f"Expiration {exp_date}: {len(df)} options pass strict filter (vol>10, OI>50)")
            else:
                # Level 2: Moderate filtering
                moderate_df = df[(df['volume'].fillna(0) > 0) & (df['openInterest'].fillna(0) > 10)]
                if not moderate_df.empty:
                    df = moderate_df
                    logger.debug(f"Expiration {exp_date}: {len(df)} options pass moderate filter (vol>0, OI>10)")
                else:
                    # Level 3: Relaxed filtering
                    relaxed_df = df[(df['volume'].fillna(0) > 0) | (df['openInterest'].fillna(0) > 5)]
                    if not relaxed_df.empty:
                        df = relaxed_df
                        logger.debug(f"Expiration {exp_date}: {len(df)} options pass relaxed filter (vol>0 OR OI>5)")
                    else:
                        # Level 4: Minimal filtering - any open interest or recent volume
                        minimal_df = df[(df['openInterest'].fillna(0) > 0) | (df['volume'].fillna(0) > 0)]
                        if not minimal_df.empty:
                            df = minimal_df
                            logger.debug(f"Expiration {exp_date}: {len(df)} options pass minimal filter (OI>0 OR vol>0)")
                        else:
                            # Level 5: Emergency fallback - any option with a price
                            emergency_df = df[df['lastPrice'].fillna(0) > 0]
                            if not emergency_df.empty:
                                df = emergency_df
                                logger.debug(f"Expiration {exp_date}: {len(df)} options pass emergency filter (price>0)")
                            else:
                                # No valid options for this expiration
                                logger.warning(f"Expiration {exp_date}: No valid options found after all filtering levels")
                                continue

            # Focus on ATM options for low-liquidity stocks
            if len(df) < 10:  # Low liquidity detected
                # Prioritize options within ±20% of current price
                atm_df = df[abs(df['strike'] - current_price) <= (current_price * 0.20)]
                if not atm_df.empty:
                    df = atm_df
                    logger.debug(f"Expiration {exp_date}: Focused on {len(df)} ATM options (±20%) for low liquidity")

            if df.empty:
                continue

            # Find ATM, ITM, and OTM strikes based on bias
            atm_strike = df.loc[df['strike'].sub(current_price).abs().idxmin(), 'strike']

            for bias in [primary_bias, secondary_bias] if secondary_bias != 'NEUTRAL' else [primary_bias]:
                if bias == 'NEUTRAL':
                    continue

                option_type = 'CALL' if bias == 'BULLISH' else 'PUT'
                type_df = df[df['option_type'] == option_type].copy()

                if type_df.empty:
                    continue

                # Find optimal strikes based on liquidity and Greeks potential
                for strike_type in ['ATM', 'ITM', 'OTM']:
                    if strike_type == 'ATM':
                        target_strikes = type_df[abs(type_df['strike'] - current_price) <= 0.02 * current_price]
                    elif strike_type == 'ITM':
                        if option_type == 'CALL':
                            target_strikes = type_df[(type_df['strike'] < current_price) &
                                                   (type_df['strike'] >= current_price * 0.95)]
                        else:
                            target_strikes = type_df[(type_df['strike'] > current_price) &
                                                   (type_df['strike'] <= current_price * 1.05)]
                    else:  # OTM
                        if option_type == 'CALL':
                            target_strikes = type_df[(type_df['strike'] > current_price) &
                                                   (type_df['strike'] <= current_price * 1.1)]
                        else:
                            target_strikes = type_df[(type_df['strike'] < current_price) &
                                                   (type_df['strike'] >= current_price * 0.9)]

                    if target_strikes.empty:
                        continue

                    # Select best candidate based on volume and bid-ask spread
                    target_strikes['bid_ask_spread'] = target_strikes['ask'] - target_strikes['bid']
                    target_strikes['spread_ratio'] = target_strikes['bid_ask_spread'] / target_strikes['bid'].clip(lower=0.01)

                    # Prefer high volume, low spread options
                    target_strikes['score'] = (target_strikes['volume'].fillna(0) * 0.4 +
                                             target_strikes['openInterest'].fillna(0) * 0.3 -
                                             target_strikes['spread_ratio'] * 1000 * 0.3)

                    best_option = target_strikes.loc[target_strikes['score'].idxmax()]

                    candidates.append({
                        'expiration_date': exp_date,
                        'days_to_expiration': days_to_exp,
                        'option_type': option_type,
                        'strike': best_option['strike'],
                        'strike_type': strike_type,
                        'bias': bias,
                        'current_price': best_option['lastPrice'],
                        'bid': best_option['bid'],
                        'ask': best_option['ask'],
                        'volume': best_option['volume'],
                        'open_interest': best_option['openInterest'],
                        'implied_volatility': best_option['impliedVolatility']
                    })

        # Sort by score and return top candidates
        technical_candidates = sorted(candidates, key=lambda x: x['volume'] * x['open_interest'], reverse=True)[:10]

        # If we found good technical candidates, return them
        if technical_candidates:
            logger.debug(f"Found {len(technical_candidates)} candidates using technical analysis")
            return technical_candidates

        # LLM fallback: if no technical candidates found, use LLM intelligence
        if ticker and sentiment_data is not None:
            logger.info(f"No technical candidates found for {ticker}, trying LLM-assisted selection")
            llm_candidates = self._llm_assisted_candidate_selection(
                options_data, current_price, technical_indicators, sentiment_data, ticker
            )

            if llm_candidates:
                logger.info(f"LLM fallback found {len(llm_candidates)} candidates for {ticker}")
                return llm_candidates

        logger.warning(f"No options candidates found for {ticker} using technical or LLM methods")

        # Emergency fallback: Create basic ATM options if no candidates found
        logger.info(f"Attempting emergency options selection for {ticker}")
        emergency_candidates = self._create_emergency_options_candidates(options_data, current_price, technical_indicators)

        if emergency_candidates:
            logger.info(f"Emergency fallback found {len(emergency_candidates)} basic options for {ticker}")
            return emergency_candidates

        logger.error(f"Complete failure: No options candidates found for {ticker} even with emergency fallback")
        return []

    def _create_emergency_options_candidates(self, options_data: Dict[str, pd.DataFrame],
                                           current_price: float, technical_indicators: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Emergency fallback to create basic ATM options when all other methods fail

        Args:
            options_data: Dictionary of options chains by expiration
            current_price: Current stock price
            technical_indicators: Technical analysis results

        Returns:
            List of basic options candidates
        """
        candidates = []

        try:
            # Get RSI for basic directional bias
            rsi = technical_indicators.get('RSI_14', 50)

            # Simple directional bias
            if rsi < 40:
                primary_type = 'CALL'  # Oversold, expect bounce
                secondary_type = 'PUT'
            elif rsi > 60:
                primary_type = 'PUT'   # Overbought, expect pullback
                secondary_type = 'CALL'
            else:
                primary_type = 'CALL'  # Neutral bias toward calls
                secondary_type = 'PUT'

            logger.debug(f"Emergency selection: RSI={rsi:.1f}, primary={primary_type}")

            # Find the first two available expirations
            available_expirations = sorted(options_data.keys())[:2]

            for exp_date in available_expirations:
                df = options_data[exp_date]

                if df.empty:
                    continue

                # Find strikes closest to current price for each type
                for option_type in [primary_type, secondary_type]:
                    type_options = df[df['option_type'] == option_type].copy()

                    if type_options.empty:
                        continue

                    # Find ATM strike (closest to current price)
                    type_options['strike_diff'] = abs(type_options['strike'] - current_price)
                    atm_option = type_options.loc[type_options['strike_diff'].idxmin()]

                    # Only select if it has some basic validity
                    if (pd.notna(atm_option['lastPrice']) and
                        atm_option['lastPrice'] > 0 and
                        atm_option['strike'] > 0):

                        candidate = {
                            'expiration_date': exp_date,
                            'days_to_expiration': atm_option['days_to_expiration'],
                            'option_type': option_type,
                            'strike': atm_option['strike'],
                            'strike_type': 'ATM_EMERGENCY',
                            'bias': 'EMERGENCY_SELECTION',
                            'current_price': atm_option['lastPrice'],
                            'bid': atm_option.get('bid', 0),
                            'ask': atm_option.get('ask', 0),
                            'volume': atm_option.get('volume', 0),
                            'open_interest': atm_option.get('openInterest', 0),
                            'implied_volatility': atm_option.get('impliedVolatility', 0.3)
                        }

                        candidates.append(candidate)
                        logger.debug(f"Emergency selected: {option_type} ${atm_option['strike']:.2f} {exp_date}")

                        # Limit to 2 options per ticker to keep it simple
                        if len(candidates) >= 2:
                            break

                if len(candidates) >= 2:
                    break

        except Exception as e:
            logger.error(f"Error in emergency options selection: {e}")

        return candidates

    def _llm_assisted_candidate_selection(self, options_data: Dict[str, pd.DataFrame],
                                        current_price: float, technical_indicators: Dict[str, float],
                                        sentiment_data: Dict[str, Any], ticker: str) -> List[Dict[str, Any]]:
        """
        Use LLM intelligence to identify options candidates when technical filtering fails

        Args:
            options_data: Dictionary of options chains by expiration
            current_price: Current stock price
            technical_indicators: Technical analysis results
            sentiment_data: News sentiment analysis
            ticker: Stock symbol

        Returns:
            List of LLM-recommended options candidates
        """
        try:
            logger.info(f"Using LLM-assisted candidate selection for {ticker}")

            # Prepare simplified options data for LLM analysis
            options_summary = self._prepare_options_summary_for_llm(options_data, current_price)

            if not options_summary:
                logger.warning(f"No options data to analyze for {ticker}")
                return []

            # Create LLM prompt for options candidate selection
            prompt = self._create_candidate_selection_prompt(
                ticker, options_summary, current_price, technical_indicators, sentiment_data
            )

            # Get LLM recommendations
            llm_response = self.llm.generate(prompt)
            logger.info(f"{llm_response}")

            # Parse LLM recommendations into candidate format
            candidates = self._parse_llm_candidate_recommendations(llm_response, options_data, current_price)

            logger.info(f"LLM identified {len(candidates)} potential options candidates for {ticker}")
            return candidates

        except Exception as e:
            logger.error(f"Error in LLM-assisted candidate selection for {ticker}: {e}")
            return []

    def _prepare_options_summary_for_llm(self, options_data: Dict[str, pd.DataFrame], current_price: float) -> str:
        """Prepare a concise options summary for LLM analysis"""
        summary_lines = []

        for exp_date, df in options_data.items():
            if df.empty:
                continue

            days_to_exp = df.iloc[0]['days_to_expiration']

            # Get calls and puts near the money
            atm_range = current_price * 0.1  # Within 10% of current price
            near_money = df[abs(df['strike'] - current_price) <= atm_range]

            if near_money.empty:
                continue

            calls = near_money[near_money['option_type'] == 'CALL']
            puts = near_money[near_money['option_type'] == 'PUT']

            summary_lines.append(f"Expiration {exp_date} ({days_to_exp} days):")

            if not calls.empty:
                best_call = calls.loc[calls['openInterest'].fillna(0).idxmax()]
                summary_lines.append(f"  Best CALL: ${best_call['strike']:.2f} strike, IV: {best_call['impliedVolatility']:.1%}, Vol: {best_call['volume']}, OI: {best_call['openInterest']}")

            if not puts.empty:
                best_put = puts.loc[puts['openInterest'].fillna(0).idxmax()]
                summary_lines.append(f"  Best PUT: ${best_put['strike']:.2f} strike, IV: {best_put['impliedVolatility']:.1%}, Vol: {best_put['volume']}, OI: {best_put['openInterest']}")

        return "\n".join(summary_lines) if summary_lines else ""

    def _create_candidate_selection_prompt(self, ticker: str, options_summary: str,
                                         current_price: float, technical_indicators: Dict[str, float],
                                         sentiment_data: Dict[str, Any]) -> str:
        """Create LLM prompt for options candidate selection"""

        prompt = f"""You are an institutional options trader evaluating {ticker} options opportunities.

CURRENT SITUATION:
- Stock Price: ${current_price:.2f}
- RSI: {technical_indicators.get('RSI_14', 50):.1f}
- Recent Sentiment: {sentiment_data.get('overall_sentiment', 'neutral')}
- Key Events: {sentiment_data.get('key_events', 'None identified')}

AVAILABLE OPTIONS:
{options_summary}

TRADING CONTEXT:
This stock has limited options liquidity, but we need to identify the best available opportunities despite lower volume/open interest.

ANALYSIS REQUIREMENTS:
1. Evaluate each expiration timeframe for potential opportunities
2. Consider risk/reward despite limited liquidity
3. Factor in technical setup and recent news sentiment
4. Recommend 2-3 best options candidates with reasoning

RESPONSE FORMAT (REQUIRED - JSON ONLY):
{{
  "recommendations": [
    {{
      "option_type": "CALL",
      "strike_price": {current_price:.2f},
      "expiration_date": "2025-XX-XX",
      "recommendation": "BUY",
      "target_premium": 2.50,
      "confidence": "HIGH",
      "reasoning": "Detailed explanation for why this option despite liquidity constraints",
      "risk_level": "MEDIUM"
    }}
  ]
}}

Provide 2-3 recommendations in valid JSON format. Focus on options that offer the best risk/reward given the available choices.
"""

        return prompt

    def _parse_llm_candidate_recommendations(self, llm_response: str, options_data: Dict[str, pd.DataFrame],
                                           current_price: float) -> List[Dict[str, Any]]:
        """Parse LLM response into candidate format with JSON support and fallbacks"""
        candidates = []

        # Try JSON parsing first
        json_candidates = self._parse_json_recommendations(llm_response, options_data)
        if json_candidates:
            logger.debug(f"Successfully parsed {len(json_candidates)} candidates from JSON")
            return json_candidates

        # Fallback to text parsing
        logger.debug("JSON parsing failed, attempting text parsing")
        text_candidates = self._parse_text_recommendations(llm_response, options_data, current_price)
        if text_candidates:
            logger.debug(f"Successfully parsed {len(text_candidates)} candidates from text")
            return text_candidates

        logger.warning("Both JSON and text parsing failed for LLM candidate recommendations")
        return []

    def _parse_json_recommendations(self, llm_response: str, options_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Parse JSON format LLM recommendations"""
        import json
        import re

        try:
            # Try to extract JSON from the response
            json_text = llm_response.strip()

            # Look for JSON in markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)

            # Try to find JSON object in the text
            if not json_text.startswith('{'):
                json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)

            # Parse the JSON
            response_data = json.loads(json_text)
            recommendations = response_data.get('recommendations', [])

            candidates = []
            for rec in recommendations:
                try:
                    option_type = rec.get('option_type', '').upper()
                    strike_price = float(rec.get('strike_price', 0))
                    exp_date_str = rec.get('expiration_date', '')

                    # Find matching expiration date
                    exp_date = self._find_closest_expiration(exp_date_str, options_data)

                    if exp_date and exp_date in options_data:
                        # Find matching option in data
                        exp_df = options_data[exp_date]
                        matching_options = exp_df[
                            (exp_df['option_type'] == option_type) &
                            (abs(exp_df['strike'] - strike_price) < 0.5)
                        ]

                        if not matching_options.empty:
                            best_match = matching_options.iloc[0]

                            candidate = {
                                'expiration_date': exp_date,
                                'days_to_expiration': best_match['days_to_expiration'],
                                'option_type': option_type,
                                'strike': strike_price,
                                'strike_type': 'LLM_SELECTED',
                                'bias': 'LLM_RECOMMENDATION',
                                'current_price': best_match['lastPrice'],
                                'bid': best_match['bid'],
                                'ask': best_match['ask'],
                                'volume': best_match['volume'],
                                'open_interest': best_match['openInterest'],
                                'implied_volatility': best_match['impliedVolatility'],
                                'llm_reasoning': rec.get('reasoning', ''),
                                'llm_confidence': rec.get('confidence', 'MEDIUM'),
                                'target_premium': rec.get('target_premium')
                            }

                            candidates.append(candidate)
                            logger.debug(f"JSON parsed: {option_type} ${strike_price} {exp_date}")

                except Exception as e:
                    logger.warning(f"Error parsing individual JSON recommendation: {e}")
                    continue

            return candidates[:5]

        except Exception as e:
            logger.debug(f"JSON parsing failed: {e}")
            return []

    def _parse_text_recommendations(self, llm_response: str, options_data: Dict[str, pd.DataFrame],
                                   current_price: float) -> List[Dict[str, Any]]:
        """Fallback text parsing for non-JSON responses"""
        import re

        candidates = []

        try:
            lines = llm_response.split('\n')

            for line in lines:
                line = line.strip()

                # Look for various option line formats
                if any(pattern in line.upper() for pattern in ['OPTION:', '**OPTION:', 'CALL $', 'PUT $']):
                    try:
                        # Extract option type
                        option_type = 'CALL' if 'CALL' in line.upper() else 'PUT'

                        # Extract strike price
                        strike_match = re.search(r'\$?(\d+\.?\d*)', line)
                        strike_price = float(strike_match.group(1)) if strike_match else current_price

                        # Extract expiration date
                        exp_date = self._find_closest_expiration(line, options_data)

                        if exp_date and exp_date in options_data:
                            # Find matching option in data
                            exp_df = options_data[exp_date]
                            matching_options = exp_df[
                                (exp_df['option_type'] == option_type) &
                                (abs(exp_df['strike'] - strike_price) < 0.5)
                            ]

                            if not matching_options.empty:
                                best_match = matching_options.iloc[0]

                                candidate = {
                                    'expiration_date': exp_date,
                                    'days_to_expiration': best_match['days_to_expiration'],
                                    'option_type': option_type,
                                    'strike': strike_price,
                                    'strike_type': 'LLM_SELECTED',
                                    'bias': 'LLM_RECOMMENDATION',
                                    'current_price': best_match['lastPrice'],
                                    'bid': best_match['bid'],
                                    'ask': best_match['ask'],
                                    'volume': best_match['volume'],
                                    'open_interest': best_match['openInterest'],
                                    'implied_volatility': best_match['impliedVolatility']
                                }

                                candidates.append(candidate)
                                logger.debug(f"Text parsed: {option_type} ${strike_price} {exp_date}")

                    except Exception as e:
                        logger.debug(f"Error parsing text line '{line}': {e}")
                        continue

        except Exception as e:
            logger.warning(f"Error in text parsing: {e}")

        return candidates[:5]

    def _find_closest_expiration(self, text: str, options_data: Dict[str, pd.DataFrame]) -> str:
        """Find the closest matching expiration date from text"""
        import re
        from datetime import datetime

        # Try to find exact date match first
        for exp_date in options_data.keys():
            if exp_date in text or exp_date.replace('-', '') in text.replace('-', ''):
                return exp_date

        # Try to extract date patterns
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{2}-\d{2}-\d{4})',  # MM-DD-YYYY
            r'(\d{1,2}/\d{1,2}/\d{4})',  # M/D/YYYY
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    date_str = match.group(1)
                    # Try to parse and match with available dates
                    for exp_date in options_data.keys():
                        if date_str.replace('/', '-').replace('-', '') in exp_date.replace('-', ''):
                            return exp_date
                except:
                    continue

        # Fallback to closest expiration
        if options_data:
            return list(options_data.keys())[0]

        return None

    def get_options_llm_analysis(self, ticker: str, candidates: List[Dict[str, Any]],
                               technical_indicators: Dict[str, float], sentiment_data: Dict[str, Any]) -> List[OptionsPredictionRecord]:
        """
        Generate LLM analysis for options candidates and create prediction records
        """
        predictions = []
        current_price = candidates[0]['current_price'] if candidates else 0.0

        # Get current stock data for context
        df, _, _ = self.get_stock_data(ticker, '3mo')
        if df is None or df.empty:
            logger.error(f"Cannot fetch stock data for {ticker}")
            return []

        current_price = df.iloc[-1]['Close']

        for candidate in candidates:  # Analyze top 5 candidates
            try:
                # Calculate Greeks
                T = candidate['days_to_expiration'] / 365.0
                greeks = self.greeks_calculator.calculate_greeks(
                    S=current_price,
                    K=candidate['strike'],
                    T=T,
                    r=0.05,  # Risk-free rate assumption
                    sigma=candidate['implied_volatility'],
                    option_type=candidate['option_type']
                )

                # Create analysis prompt
                prompt = self._create_options_analysis_prompt(
                    ticker, candidate, technical_indicators, sentiment_data, greeks
                )

                # Get LLM analysis
                logger.info(f"Generating individual analysis for {ticker} {candidate['option_type']} ${candidate['strike']}")
                analysis = self.llm.generate(prompt)

                # Log the raw LLM response for debugging
                logger.debug(f"Raw LLM analysis response for {ticker}: {analysis[:500]}...")

                # Extract structured recommendation from analysis
                logger.debug(f"Attempting to extract recommendation from analysis for {ticker}")
                analysis_result = self._extract_options_recommendation(analysis)

                # Extract fields from dictionary
                recommendation = analysis_result.get('recommendation', 'HOLD')
                confidence = analysis_result.get('confidence', 'MEDIUM')
                target_premium = analysis_result.get('target_premium')
                max_loss = analysis_result.get('max_loss')

                # Log extraction results
                logger.info(f"Extracted from {ticker} analysis: {recommendation} ({confidence}), "
                           f"target=${target_premium}, max_loss=${max_loss}")

                # Convert technical indicators to JSON-serializable format
                serializable_indicators = {}
                for key, value in technical_indicators.items():
                    try:
                        if hasattr(value, 'item'):  # pandas scalar
                            serializable_indicators[key] = value.item()
                        elif hasattr(value, 'iloc') and len(value) > 0:  # pandas Series
                            serializable_indicators[key] = value.iloc[0]
                        elif hasattr(value, '__len__') and len(value) == 1:  # single-element array
                            serializable_indicators[key] = float(value[0])
                        elif isinstance(value, (int, float, str, bool)):
                            serializable_indicators[key] = value
                        else:
                            serializable_indicators[key] = str(value)
                    except Exception as conv_error:
                        logger.debug(f"Error converting technical indicator {key} (value: {value}, type: {type(value)}): {conv_error}")
                        serializable_indicators[key] = str(value)

                # Create prediction record
                prediction = OptionsPredictionRecord(
                    ticker=ticker,
                    prediction_date=datetime.now().strftime('%Y-%m-%d'),
                    option_type=candidate['option_type'],
                    strike_price=candidate['strike'],
                    expiration_date=candidate['expiration_date'],
                    days_to_expiration=candidate['days_to_expiration'],
                    recommendation=recommendation,
                    confidence=confidence,
                    entry_premium=candidate['current_price'],
                    target_premium=target_premium,
                    max_loss=max_loss,
                    underlying_price=current_price,
                    implied_volatility=candidate['implied_volatility'],
                    volume=self._safe_int_convert(candidate['volume']),
                    open_interest=self._safe_int_convert(candidate['open_interest']),
                    greeks=greeks,
                    technical_indicators=serializable_indicators,
                    llm_analysis=analysis,
                    sentiment_data=sentiment_data
                )

                predictions.append(prediction)

            except Exception as e:
                logger.error(f"Error analyzing options candidate for {ticker}: {e}")
                continue

        return predictions

    def _safe_int_convert(self, value):
        """Safely convert pandas scalars and other types to integer"""
        try:
            if value is None or (hasattr(value, '__len__') and len(value) == 0):
                return 0
            if hasattr(value, 'item'):  # pandas scalar
                return int(value.item())
            elif hasattr(value, 'iloc') and len(value) > 0:  # pandas Series
                return int(value.iloc[0])
            elif hasattr(value, '__len__') and len(value) == 1:  # single-element array
                return int(value[0])
            elif hasattr(value, '__iter__') and not isinstance(value, str):  # multi-element array
                # Convert first element if iterable contains multiple elements
                first_val = next(iter(value), 0)
                return int(first_val) if first_val is not None else 0
            else:
                return int(value)
        except (ValueError, TypeError, IndexError, StopIteration) as e:
            logger.debug(f"Error converting value {value} (type: {type(value)}) to int: {e}")
            return 0

    def _create_options_analysis_prompt(self, ticker: str, candidate: Dict[str, Any],
                                       technical_indicators: Dict[str, Any], sentiment_data: Dict[str, Any],
                                       greeks: Dict[str, float]) -> str:
        """Create comprehensive options analysis prompt for LLM"""

        prompt = f"""You are an institutional options trader analyzing {ticker}. Provide a comprehensive options analysis.

OPTION DETAILS:
- Type: {candidate['option_type']}
- Strike: ${candidate['strike']:.2f}
- Expiration: {candidate['expiration_date']} ({candidate['days_to_expiration']} days)
- Current Premium: ${candidate['current_price']:.2f}
- Bid/Ask: ${candidate['bid']:.2f}/${candidate['ask']:.2f}
- Volume: {candidate['volume']}
- Open Interest: {candidate['open_interest']}
- IV: {candidate['implied_volatility']:.1%}

GREEKS:
- Delta: {greeks['delta']:.3f}
- Gamma: {greeks['gamma']:.3f}
- Theta: {greeks['theta']:.3f}
- Vega: {greeks['vega']:.3f}
- Rho: {greeks['rho']:.3f}

TECHNICAL ANALYSIS:
"""

        for key, value in technical_indicators.items():
            if isinstance(value, (int, float)):
                prompt += f"- {key}: {value:.2f}\n"
            else:
                prompt += f"- {key}: {value}\n"

        prompt += f"""
SENTIMENT ANALYSIS:
- Overall Sentiment: {sentiment_data.get('overall_sentiment', 'NEUTRAL')}
- Confidence: {sentiment_data.get('confidence', 0.5):.2f}
- Key Events: {sentiment_data.get('key_events', 'None identified')}

ANALYSIS REQUIREMENTS:
1. Evaluate the risk/reward profile of this specific option
2. Assess the impact of time decay (theta) on profitability
3. Consider implied volatility levels and potential IV crush
4. Analyze the probability of profit based on technical setup
5. Determine optimal entry and exit strategy

RESPONSE FORMAT (REQUIRED - JSON ONLY):
{{
  "recommendation": "BUY",
  "confidence": "HIGH",
  "target_premium": 2.50,
  "max_loss": 1.00,
  "reasoning": "Detailed institutional-grade analysis considering all Greeks, market conditions, and risk management",
  "risk_factors": ["Time decay risk", "IV crush potential"],
  "probability_of_profit": 65,
  "key_levels": {{
    "stop_loss": 1.50,
    "profit_target_1": 2.25,
    "profit_target_2": 3.00
  }}
}}

Provide valid JSON response with institutional-grade analysis.
"""

        return prompt

    def _extract_options_recommendation(self, analysis: str) -> dict:
        """Extract structured recommendation from LLM analysis with JSON support and fallbacks"""

        # Try JSON parsing first
        logger.debug("Attempting JSON extraction...")
        json_result = self._extract_json_recommendation(analysis)
        if json_result:
            logger.info(f"JSON extraction successful")
            return json_result
        else:
            logger.debug("JSON extraction failed")

        # Fallback to text parsing
        logger.debug("Attempting text extraction fallback...")
        text_result = self._extract_text_recommendation(analysis)
        logger.info(f"Text extraction result: {text_result}")

        # Convert text result tuple to dictionary format for consistency
        if text_result and len(text_result) == 4:
            recommendation, confidence, target_premium, max_loss = text_result
            return {
                'recommendation': recommendation,
                'confidence': confidence,
                'target_premium': target_premium,
                'max_loss': max_loss
            }
        else:
            # Return default dictionary if text extraction also fails
            return {
                'recommendation': 'HOLD',
                'confidence': 'MEDIUM',
                'target_premium': None,
                'max_loss': None
            }

    def _extract_json_recommendation(self, analysis: str) -> Optional[dict]:
        """Extract recommendation from JSON format - returns full JSON structure from individual analysis"""
        import json
        import re

        try:
            # Try to extract JSON from the response
            json_text = analysis.strip()

            # Look for JSON in markdown code blocks first
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            elif not json_text.startswith('{'):
                # Try to find JSON object in the text
                json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)

            # Parse the JSON
            data = json.loads(json_text)
            logger.debug(f"Successfully parsed JSON with keys: {list(data.keys())}")

            # Extract and validate core recommendation fields
            recommendation = data.get('recommendation', 'HOLD').upper()
            confidence = data.get('confidence', 'MEDIUM').upper()
            target_premium = data.get('target_premium')
            max_loss = data.get('max_loss')

            # Handle alternative field names for target_premium
            if target_premium is None:
                # Check for nested key_levels structure
                key_levels = data.get('key_levels', {})
                if isinstance(key_levels, dict):
                    target_premium = key_levels.get('profit_target_1') or key_levels.get('profit_target_2')

            # Handle alternative field names for max_loss
            if max_loss is None:
                key_levels = data.get('key_levels', {})
                if isinstance(key_levels, dict):
                    max_loss = key_levels.get('stop_loss')

            # Convert string numbers to float with better error handling
            def safe_float_convert(value, field_name):
                if value is None:
                    return None
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    try:
                        # Handle string values like "$2.50" or "2.50"
                        clean_value = value.replace('$', '').replace(',', '').strip()
                        return float(clean_value)
                    except ValueError:
                        logger.debug(f"Could not convert {field_name} '{value}' to float")
                        return None
                return None

            target_premium = safe_float_convert(target_premium, "target_premium")
            max_loss = safe_float_convert(max_loss, "max_loss")

            # Validate core fields
            if recommendation not in ['BUY', 'SELL', 'HOLD']:
                logger.warning(f"Invalid recommendation '{recommendation}', defaulting to HOLD")
                recommendation = 'HOLD'

            if confidence not in ['HIGH', 'MEDIUM', 'LOW']:
                logger.warning(f"Invalid confidence '{confidence}', defaulting to MEDIUM")
                confidence = 'MEDIUM'

            # Update the original data with validated values
            data['recommendation'] = recommendation
            data['confidence'] = confidence
            data['target_premium'] = target_premium
            data['max_loss'] = max_loss

            # Log extracted values for debugging
            logger.debug(f"Validated JSON data: recommendation={recommendation}, confidence={confidence}, "
                        f"target_premium={target_premium}, max_loss={max_loss}")

            return data

        except json.JSONDecodeError as e:
            logger.debug(f"JSON parsing failed: {e}")
            return None
        except Exception as e:
            logger.debug(f"JSON recommendation extraction failed: {e}")
            return None

    def _extract_text_recommendation(self, analysis: str) -> Tuple[str, str, Optional[float], Optional[float]]:
        """Enhanced fallback text parsing for complex LLM responses"""
        import re

        recommendation = "HOLD"
        confidence = "MEDIUM"
        target_premium = None
        max_loss = None

        try:
            # Convert to uppercase for easier pattern matching
            text_upper = analysis.upper()

            # Enhanced recommendation extraction with multiple patterns
            rec_patterns = [
                r'RECOMMENDATION[:\s]+([A-Z]+)',
                r'"RECOMMENDATION"[:\s]*"([A-Z]+)"',
                r'RECOMMEND[:\s]+(BUY|SELL|HOLD)',
                r'ACTION[:\s]+(BUY|SELL|HOLD)',
                r'TRADE[:\s]+(BUY|SELL|HOLD)',
                r'(BUY|SELL|HOLD)\s+RECOMMENDATION',
                r'I\s+RECOMMEND[:\s]*(BUY|SELL|HOLD)',
                r'STRATEGY[:\s]+(BUY|SELL|HOLD)'
            ]

            for pattern in rec_patterns:
                match = re.search(pattern, text_upper)
                if match:
                    rec_value = match.group(1).strip()
                    if rec_value in ['BUY', 'LONG', 'PURCHASE', 'CALL']:
                        recommendation = "BUY"
                        break
                    elif rec_value in ['SELL', 'SHORT', 'AVOID', 'PUT']:
                        recommendation = "SELL"
                        break
                    elif rec_value == 'HOLD':
                        recommendation = "HOLD"
                        break

            # Enhanced confidence extraction
            conf_patterns = [
                r'CONFIDENCE[:\s]+([A-Z]+)',
                r'"CONFIDENCE"[:\s]*"([A-Z]+)"',
                r'CONFIDENCE[:\s]*([0-9]+)%',
                r'(HIGH|MEDIUM|LOW)\s+CONFIDENCE',
                r'PROBABILITY[:\s]*([0-9]+)%'
            ]

            for pattern in conf_patterns:
                match = re.search(pattern, text_upper)
                if match:
                    conf_value = match.group(1).strip()
                    if conf_value in ['HIGH', '70', '80', '90'] or (conf_value.isdigit() and int(conf_value) >= 70):
                        confidence = "HIGH"
                        break
                    elif conf_value in ['LOW', '30', '40'] or (conf_value.isdigit() and int(conf_value) <= 40):
                        confidence = "LOW"
                        break
                    elif conf_value in ['MEDIUM', 'MED'] or (conf_value.isdigit() and 40 < int(conf_value) < 70):
                        confidence = "MEDIUM"
                        break

            # Enhanced target premium extraction with multiple patterns
            premium_patterns = [
                r'TARGET[_\s]*PREMIUM[:\s]*\$?([0-9]+\.?[0-9]*)',
                r'"TARGET_PREMIUM"[:\s]*([0-9]+\.?[0-9]*)',
                r'PROFIT[_\s]*TARGET[_\s]*1?[:\s]*\$?([0-9]+\.?[0-9]*)',
                r'EXIT[:\s]*\$?([0-9]+\.?[0-9]*)',
                r'TAKE[_\s]*PROFIT[:\s]*\$?([0-9]+\.?[0-9]*)',
                r'TARGET[:\s]*\$?([0-9]+\.?[0-9]*)'
            ]

            for pattern in premium_patterns:
                match = re.search(pattern, text_upper)
                if match:
                    try:
                        target_premium = float(match.group(1))
                        break
                    except ValueError:
                        continue

            # Enhanced max loss extraction
            loss_patterns = [
                r'MAX[_\s]*LOSS[:\s]*\$?([0-9]+\.?[0-9]*)',
                r'"MAX_LOSS"[:\s]*([0-9]+\.?[0-9]*)',
                r'STOP[_\s]*LOSS[:\s]*\$?([0-9]+\.?[0-9]*)',
                r'RISK[:\s]*\$?([0-9]+\.?[0-9]*)',
                r'MAXIMUM[_\s]*RISK[:\s]*\$?([0-9]+\.?[0-9]*)'
            ]

            for pattern in loss_patterns:
                match = re.search(pattern, text_upper)
                if match:
                    try:
                        max_loss = float(match.group(1))
                        break
                    except ValueError:
                        continue

            # If still no values found, try line-by-line parsing for simpler formats
            if target_premium is None or max_loss is None:
                lines = analysis.split('\n')
                for line in lines:
                    line = line.strip()

                    # Look for dollar amounts that might be targets or losses
                    dollar_matches = re.findall(r'\$([0-9]+\.?[0-9]*)', line)
                    if dollar_matches and target_premium is None:
                        if any(word in line.upper() for word in ['TARGET', 'PROFIT', 'EXIT']):
                            try:
                                target_premium = float(dollar_matches[0])
                            except ValueError:
                                pass

                    if dollar_matches and max_loss is None:
                        if any(word in line.upper() for word in ['LOSS', 'STOP', 'RISK']):
                            try:
                                max_loss = float(dollar_matches[0])
                            except ValueError:
                                pass

            logger.debug(f"Text extraction result: recommendation={recommendation}, confidence={confidence}, "
                        f"target_premium={target_premium}, max_loss={max_loss}")

        except Exception as e:
            logger.warning(f"Error in enhanced text extraction: {e}")

        return recommendation, confidence, target_premium, max_loss

    def analyze_ticker_options(self, ticker: str) -> List[OptionsPredictionRecord]:
        """
        Complete options analysis workflow for a single ticker

        Args:
            ticker: Stock symbol to analyze

        Returns:
            List of options prediction records
        """
        try:
            logger.info(f"Starting options analysis for {ticker}")

            # Get stock data and technical analysis
            df, _, _ = self.get_stock_data(ticker, '1y')
            if df is None or df.empty:
                logger.warning(f"No stock data available for {ticker}")
                return []

            # Calculate technical indicators (reuse existing method)
            from .llmAnalysis import AdvancedTechnicalAnalysis
            df = AdvancedTechnicalAnalysis.calculate_comprehensive_indicators(df)
            current_price = df.iloc[-1]['Close']

            # Get technical indicators dictionary
            technical_indicators = {
                'RSI_14': df.iloc[-1].get('RSI_14', 50),
                'MACD_signal': 'NEUTRAL',  # Simplified for now
                'bb_position': 0.5,  # Simplified for now
                'current_price': current_price,
                'volume_sma_ratio': df.iloc[-1].get('volume_sma_ratio', 1.0),
                'price_change_1d': (current_price - df.iloc[-2]['Close']) / df.iloc[-2]['Close'],
                'volatility_20d': df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
            }

            # Get news sentiment
            articles = self.fetch_yahoo_news(ticker)[:5]
            sentiment_data = self.analyze_sentiment(ticker, articles)

            # Fetch options data
            logger.debug(f"Fetching options chain for {ticker}")
            options_data = self.data_fetcher.get_options_chain(ticker)

            if not options_data:
                logger.warning(f"No options data available for {ticker}")
                return []

            logger.info(f"Found options data for {ticker}: {len(options_data)} expiration dates")
            for exp_date, exp_df in options_data.items():
                logger.debug(f"  {exp_date}: {len(exp_df)} total options ({len(exp_df[exp_df['option_type']=='CALL'])} calls, {len(exp_df[exp_df['option_type']=='PUT'])} puts)")

            # Get best options candidates using enhanced method with LLM fallback
            candidates = self.get_best_options_candidates(
                options_data, current_price, technical_indicators,
                sentiment_data.get('sentiment_score', 0.0), ticker, sentiment_data
            )

            if not candidates:
                logger.warning(f"No suitable options candidates found for {ticker}")
                return []

            # Generate LLM analysis and predictions
            predictions = self.get_options_llm_analysis(
                ticker, candidates, technical_indicators, sentiment_data
            )

            # Record predictions in database
            for prediction in predictions:
                self.options_tracker.record_options_prediction(prediction)

            logger.info(f"Generated {len(predictions)} options predictions for {ticker}")
            return predictions

        except Exception as e:
            logger.error(f"Options analysis failed for {ticker}: {e}")
            return []


class OptionsTracker(PredictionTracker):
    """Extended tracker for options predictions"""

    def _init_database(self):
        """Initialize SQLite database with options tables"""
        # Call parent initialization first
        super()._init_database()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create options predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS options_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                option_type TEXT NOT NULL,
                strike_price REAL NOT NULL,
                expiration_date TEXT NOT NULL,
                days_to_expiration INTEGER NOT NULL,
                recommendation TEXT NOT NULL,
                confidence TEXT NOT NULL,
                entry_premium REAL NOT NULL,
                target_premium REAL,
                max_loss REAL,
                underlying_price REAL NOT NULL,
                implied_volatility REAL NOT NULL,
                volume INTEGER NOT NULL,
                open_interest INTEGER NOT NULL,
                greeks TEXT NOT NULL,
                technical_indicators TEXT NOT NULL,
                llm_analysis TEXT NOT NULL,
                sentiment_data TEXT NOT NULL,
                status TEXT DEFAULT 'OPEN'
            )
        ''')

        # Create options performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS options_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                option_type TEXT NOT NULL,
                outcome_date TEXT NOT NULL,
                exit_premium REAL NOT NULL,
                actual_return REAL NOT NULL,
                predicted_return REAL NOT NULL,
                days_held INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                max_profit_achieved REAL NOT NULL,
                underlying_move REAL NOT NULL,
                iv_change REAL NOT NULL,
                market_condition TEXT NOT NULL,
                FOREIGN KEY (prediction_id) REFERENCES options_predictions(id)
            )
        ''')

        # Create options LLM feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS options_llm_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                total_predictions INTEGER NOT NULL,
                success_rate_by_type TEXT NOT NULL,
                success_rate_by_expiration TEXT NOT NULL,
                avg_return_by_type TEXT NOT NULL,
                avg_return_by_expiration TEXT NOT NULL,
                reflection TEXT NOT NULL,
                improvements TEXT NOT NULL,
                market_regime_analysis TEXT NOT NULL
            )
        ''')

        conn.commit()
        conn.close()

    def record_options_prediction(self, prediction: OptionsPredictionRecord) -> int:
        """Record a new options prediction"""
        import json

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO options_predictions (
                ticker, prediction_date, option_type, strike_price, expiration_date,
                days_to_expiration, recommendation, confidence, entry_premium,
                target_premium, max_loss, underlying_price, implied_volatility,
                volume, open_interest, greeks, technical_indicators,
                llm_analysis, sentiment_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction.ticker,
            prediction.prediction_date,
            prediction.option_type,
            prediction.strike_price,
            prediction.expiration_date,
            prediction.days_to_expiration,
            prediction.recommendation,
            prediction.confidence,
            prediction.entry_premium,
            prediction.target_premium,
            prediction.max_loss,
            prediction.underlying_price,
            prediction.implied_volatility,
            prediction.volume,
            prediction.open_interest,
            json.dumps(prediction.greeks),
            json.dumps(prediction.technical_indicators),
            prediction.llm_analysis,
            json.dumps(prediction.sentiment_data)
        ))

        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"Recorded options prediction {prediction_id} for {prediction.ticker}")
        return prediction_id