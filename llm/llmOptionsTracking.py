"""
Options Performance Evaluation and Reflection System

This module provides comprehensive analysis of options predictions performance,
generates insights, and creates improvement recommendations using LLM reflection.
"""

import logging
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .llmTracking import PredictionTracker, PerformanceEvaluator, LLMReflectionEngine
from .llm_models import OptionsPredictionRecord, OptionsPerformanceRecord, OptionsLLMFeedback

logger = logging.getLogger(__name__)


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


class OptionsPerformanceEvaluator(PerformanceEvaluator):
    """Enhanced performance evaluator specifically for options predictions"""

    def __init__(self, options_tracker: OptionsTracker):
        # Use parent constructor but with options tracker
        self.tracker = options_tracker

    def check_and_update_options_predictions(self) -> List[Dict[str, Any]]:
        """
        Check open options predictions and update their performance

        Returns:
            List of updated prediction results
        """
        updated_predictions = []

        try:
            conn = sqlite3.connect(self.tracker.db_path)
            cursor = conn.cursor()

            # Get all open options predictions that have expired or are close to expiring
            cursor.execute('''
                SELECT * FROM options_predictions
                WHERE status = 'OPEN'
                AND (
                    expiration_date <= date('now') OR
                    date(expiration_date) <= date('now', '+2 days')
                )
            ''')

            open_predictions = cursor.fetchall()
            column_names = [description[0] for description in cursor.description]

            for pred_row in open_predictions:
                prediction = dict(zip(column_names, pred_row))

                try:
                    # Get current option price
                    current_premium, underlying_price = self._get_current_option_price(
                        prediction['ticker'],
                        prediction['option_type'],
                        prediction['strike_price'],
                        prediction['expiration_date']
                    )

                    if current_premium is not None:
                        # Calculate performance metrics
                        performance = self._calculate_options_performance(prediction, current_premium, underlying_price)

                        # Record performance
                        self._record_options_performance(prediction['id'], performance)

                        # Update prediction status
                        cursor.execute('''
                            UPDATE options_predictions
                            SET status = 'CLOSED'
                            WHERE id = ?
                        ''', (prediction['id'],))

                        updated_predictions.append({
                            'prediction_id': prediction['id'],
                            'ticker': prediction['ticker'],
                            'performance': performance
                        })

                        logger.info(f"Updated options prediction {prediction['id']} for {prediction['ticker']}")

                except Exception as e:
                    logger.error(f"Error updating options prediction {prediction['id']}: {e}")
                    continue

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error checking options predictions: {e}")

        return updated_predictions

    def _get_current_option_price(self, ticker: str, option_type: str, strike: float, expiration: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Get current option price and underlying price

        Returns:
            Tuple of (option_premium, underlying_price)
        """
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)

            # Get current stock price
            hist = stock.history(period='1d')
            if hist.empty:
                return None, None

            underlying_price = hist['Close'].iloc[-1]

            # Try to get current option price
            try:
                option_chain = stock.option_chain(expiration)

                if option_type.upper() == 'CALL':
                    options_df = option_chain.calls
                else:
                    options_df = option_chain.puts

                matching_options = options_df[options_df['strike'] == strike]

                if not matching_options.empty:
                    # Use last traded price, or mid price if available
                    last_price = matching_options.iloc[0]['lastPrice']
                    if last_price > 0:
                        return last_price, underlying_price

                    # Fallback to mid price
                    bid = matching_options.iloc[0]['bid']
                    ask = matching_options.iloc[0]['ask']
                    if bid > 0 and ask > 0:
                        return (bid + ask) / 2, underlying_price

            except Exception as e:
                logger.warning(f"Could not fetch current option price for {ticker}: {e}")

            # If expired or no current price available, option is likely worthless
            exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
            if exp_date <= datetime.now().date():
                if option_type.upper() == 'CALL':
                    intrinsic_value = max(0, underlying_price - strike)
                else:
                    intrinsic_value = max(0, strike - underlying_price)

                return intrinsic_value, underlying_price

            return None, underlying_price

        except Exception as e:
            logger.error(f"Error fetching current option price: {e}")
            return None, None

    def _calculate_options_performance(self, prediction: Dict[str, Any], current_premium: float, underlying_price: float) -> OptionsPerformanceRecord:
        """Calculate comprehensive performance metrics for an options prediction"""

        entry_premium = prediction['entry_premium']
        entry_underlying = prediction['underlying_price']

        # Calculate returns
        if prediction['recommendation'] == 'BUY':
            actual_return = (current_premium - entry_premium) / entry_premium
        else:  # SELL recommendation
            actual_return = (entry_premium - current_premium) / entry_premium

        # Calculate predicted return if target was set
        target_premium = prediction.get('target_premium')
        if target_premium:
            if prediction['recommendation'] == 'BUY':
                predicted_return = (target_premium - entry_premium) / entry_premium
            else:
                predicted_return = (entry_premium - target_premium) / entry_premium
        else:
            predicted_return = 0.0

        # Calculate underlying move
        underlying_move = (underlying_price - entry_underlying) / entry_underlying

        # Calculate IV change (simplified - would need historical IV data for accuracy)
        entry_iv = prediction['implied_volatility']
        current_iv = entry_iv  # Placeholder - would need current IV

        iv_change = current_iv - entry_iv

        # Determine outcome
        if actual_return >= predicted_return * 0.8:  # Achieved 80% of predicted return
            outcome = 'SUCCESS'
        elif actual_return > 0:
            outcome = 'PARTIAL'
        else:
            outcome = 'FAILURE'

        # Calculate days held
        prediction_date = datetime.strptime(prediction['prediction_date'], '%Y-%m-%d').date()
        days_held = (datetime.now().date() - prediction_date).days

        # Get market condition (simplified)
        market_condition = {
            'underlying_move': underlying_move,
            'iv_change': iv_change,
            'days_held': days_held
        }

        return OptionsPerformanceRecord(
            prediction_id=prediction['id'],
            ticker=prediction['ticker'],
            option_type=prediction['option_type'],
            outcome_date=datetime.now().strftime('%Y-%m-%d'),
            exit_premium=current_premium,
            actual_return=actual_return,
            predicted_return=predicted_return,
            days_held=days_held,
            outcome=outcome,
            max_profit_achieved=max(actual_return, 0),  # Simplified
            underlying_move=underlying_move,
            iv_change=iv_change,
            market_condition=market_condition
        )

    def _record_options_performance(self, prediction_id: int, performance: OptionsPerformanceRecord):
        """Record options performance in database"""

        conn = sqlite3.connect(self.tracker.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO options_performance (
                prediction_id, ticker, option_type, outcome_date, exit_premium,
                actual_return, predicted_return, days_held, outcome,
                max_profit_achieved, underlying_move, iv_change, market_condition
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            performance.prediction_id,
            performance.ticker,
            performance.option_type,
            performance.outcome_date,
            performance.exit_premium,
            performance.actual_return,
            performance.predicted_return,
            performance.days_held,
            performance.outcome,
            performance.max_profit_achieved,
            performance.underlying_move,
            performance.iv_change,
            json.dumps(performance.market_condition)
        ))

        conn.commit()
        conn.close()

    def get_options_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Get comprehensive options performance summary

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with performance metrics
        """
        try:
            conn = sqlite3.connect(self.tracker.db_path)

            # Get performance data - explicitly select columns to avoid duplicates
            perf_df = pd.read_sql_query('''
                SELECT
                    p.id, p.prediction_id, p.ticker, p.option_type, p.outcome_date,
                    p.exit_premium, p.actual_return, p.predicted_return, p.days_held,
                    p.outcome, p.max_profit_achieved, p.underlying_move, p.iv_change,
                    p.market_condition,
                    op.strike_price, op.days_to_expiration, op.confidence
                FROM options_performance p
                JOIN options_predictions op ON p.prediction_id = op.id
                WHERE p.outcome_date >= date('now', '-{} days')
            '''.format(days), conn)

            if perf_df.empty:
                conn.close()
                return {'message': 'No options performance data available'}

            # Validate required columns exist
            required_columns = ['option_type', 'actual_return', 'outcome', 'days_to_expiration', 'confidence']
            missing_columns = [col for col in required_columns if col not in perf_df.columns]
            if missing_columns:
                conn.close()
                logger.error(f"Missing required columns: {missing_columns}")
                return {'error': f'Missing required columns: {missing_columns}'}

            # Overall metrics
            total_predictions = len(perf_df)
            success_rate = len(perf_df[perf_df['outcome'] == 'SUCCESS']) / total_predictions if total_predictions > 0 else 0
            avg_return = perf_df['actual_return'].mean()

            # Performance by option type
            type_performance = {}
            if 'option_type' in perf_df.columns and len(perf_df) > 0:
                type_performance = perf_df.groupby('option_type', as_index=True).agg({
                    'actual_return': ['mean', 'count'],
                    'outcome': lambda x: (x == 'SUCCESS').mean()
                }).round(3)

            # Performance by expiration
            exp_performance = {}
            if 'days_to_expiration' in perf_df.columns and len(perf_df) > 0:
                exp_performance = perf_df.groupby('days_to_expiration', as_index=True).agg({
                    'actual_return': ['mean', 'count'],
                    'outcome': lambda x: (x == 'SUCCESS').mean()
                }).round(3)

            # Performance by confidence
            conf_performance = {}
            if 'confidence' in perf_df.columns and len(perf_df) > 0:
                conf_performance = perf_df.groupby('confidence', as_index=True).agg({
                    'actual_return': ['mean', 'count'],
                    'outcome': lambda x: (x == 'SUCCESS').mean()
                }).round(3)

            conn.close()

            # Helper function to convert grouped stats to JSON-serializable dict
            def grouped_df_to_dict(grouped_df):
                """Convert MultiIndex grouped DataFrame to dict with string keys"""
                if not isinstance(grouped_df, pd.DataFrame) or grouped_df.empty:
                    return {}

                result = {}
                for idx in grouped_df.index:
                    result[str(idx)] = {
                        'actual_return_mean': float(grouped_df.loc[idx, ('actual_return', 'mean')]),
                        'actual_return_count': int(grouped_df.loc[idx, ('actual_return', 'count')]),
                        'success_rate': float(grouped_df.loc[idx, ('outcome', '<lambda>')])
                    }
                return result

            # Safely extract best performing metrics
            best_performing_expiration = None
            if isinstance(exp_performance, pd.DataFrame) and not exp_performance.empty:
                try:
                    best_performing_expiration = exp_performance['actual_return']['mean'].idxmax()
                except:
                    pass

            best_performing_type = None
            if isinstance(type_performance, pd.DataFrame) and not type_performance.empty:
                try:
                    best_performing_type = type_performance['actual_return']['mean'].idxmax()
                except:
                    pass

            return {
                'total_predictions': total_predictions,
                'success_rate': round(success_rate, 3),
                'average_return': round(avg_return, 3),
                'performance_by_type': grouped_df_to_dict(type_performance),
                'performance_by_expiration': grouped_df_to_dict(exp_performance),
                'performance_by_confidence': grouped_df_to_dict(conf_performance),
                'best_performing_expiration': best_performing_expiration,
                'best_performing_type': best_performing_type
            }

        except Exception as e:
            logger.error(f"Error generating options performance summary: {e}")
            return {'error': str(e)}


class OptionsLLMReflectionEngine(LLMReflectionEngine):
    """Enhanced reflection engine for options-specific analysis"""

    def __init__(self, llm_provider, options_tracker: OptionsTracker):
        # Initialize with options tracker
        self.llm = llm_provider
        self.tracker = options_tracker

    def generate_options_reflection(self, ticker: str, days: int = 30) -> OptionsLLMFeedback:
        """
        Generate comprehensive reflection on options trading performance

        Args:
            ticker: Stock ticker to analyze (or 'ALL' for overall performance)
            days: Number of days to analyze

        Returns:
            OptionsLLMFeedback with reflection and improvements
        """
        try:
            # Get performance data
            evaluator = OptionsPerformanceEvaluator(self.tracker)
            performance_summary = evaluator.get_options_performance_summary(days)

            if 'error' in performance_summary or 'message' in performance_summary:
                logger.warning(f"Limited performance data available for options reflection")
                return self._create_default_options_feedback(ticker, days)

            # Create reflection prompt
            reflection_prompt = self._create_options_reflection_prompt(ticker, performance_summary, days)

            # Generate reflection using LLM
            reflection_text = self.llm.generate(reflection_prompt)

            # Extract improvements from reflection
            improvements = self._extract_improvements_from_reflection(reflection_text)

            # Analyze market regime impact
            market_analysis = self._analyze_options_market_regime(performance_summary)

            # Create feedback record
            feedback = OptionsLLMFeedback(
                feedback_date=datetime.now().strftime('%Y-%m-%d'),
                ticker=ticker,
                period_start=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                period_end=datetime.now().strftime('%Y-%m-%d'),
                total_predictions=performance_summary.get('total_predictions', 0),
                success_rate_by_type=self._extract_success_rates_by_type(performance_summary),
                success_rate_by_expiration=self._extract_success_rates_by_expiration(performance_summary),
                avg_return_by_type=self._extract_avg_returns_by_type(performance_summary),
                avg_return_by_expiration=self._extract_avg_returns_by_expiration(performance_summary),
                reflection=reflection_text,
                improvements=improvements,
                market_regime_analysis=market_analysis
            )

            # Save feedback to database
            self._save_options_feedback(feedback)

            return feedback

        except Exception as e:
            logger.error(f"Error generating options reflection: {e}")
            return self._create_default_options_feedback(ticker, days)

    def _create_options_reflection_prompt(self, ticker: str, performance_summary: Dict[str, Any], days: int) -> str:
        """Create comprehensive options reflection prompt"""

        prompt = f"""You are a senior options trading strategist analyzing the performance of LLM-generated options predictions.

PERFORMANCE ANALYSIS PERIOD: {days} days
TICKER: {ticker if ticker != 'ALL' else 'Portfolio-wide analysis'}

PERFORMANCE METRICS:
- Total Predictions: {performance_summary.get('total_predictions', 0)}
- Overall Success Rate: {performance_summary.get('success_rate', 0):.1%}
- Average Return: {performance_summary.get('average_return', 0):.2%}

PERFORMANCE BY OPTION TYPE:
{self._format_performance_by_type(performance_summary)}

PERFORMANCE BY EXPIRATION:
{self._format_performance_by_expiration(performance_summary)}

PERFORMANCE BY CONFIDENCE:
{self._format_performance_by_confidence(performance_summary)}

ANALYSIS REQUIREMENTS:
1. Identify patterns in successful vs unsuccessful predictions
2. Analyze the effectiveness of different expiration timeframes
3. Evaluate the impact of option type (calls vs puts) on performance
4. Assess confidence level accuracy - are HIGH confidence predictions actually performing better?
5. Identify market conditions that favor or hurt options strategies
6. Recommend specific improvements to the options selection process

FOCUS AREAS:
- Strike selection optimization
- Expiration timing analysis
- Market regime considerations
- Risk management effectiveness
- Entry and exit timing improvements

Provide detailed, actionable insights that can improve future options prediction accuracy.
"""

        return prompt

    def _format_performance_by_type(self, performance_summary: Dict[str, Any]) -> str:
        """Format performance by option type for prompt"""
        type_perf = performance_summary.get('performance_by_type', {})
        if not type_perf:
            return "No data available"

        output = ""
        for option_type in ['CALL', 'PUT']:
            if option_type in type_perf.get('actual_return', {}).get('mean', {}):
                avg_return = type_perf['actual_return']['mean'][option_type]
                count = type_perf['actual_return']['count'][option_type]
                success_rate = type_perf['outcome']['<lambda>'][option_type]
                output += f"- {option_type}s: {avg_return:.2%} avg return, {success_rate:.1%} success rate ({count} trades)\n"

        return output or "No type data available"

    def _format_performance_by_expiration(self, performance_summary: Dict[str, Any]) -> str:
        """Format performance by expiration for prompt"""
        exp_perf = performance_summary.get('performance_by_expiration', {})
        if not exp_perf:
            return "No data available"

        output = ""
        for days_to_exp, metrics in exp_perf.get('actual_return', {}).get('mean', {}).items():
            avg_return = metrics
            count = exp_perf['actual_return']['count'][days_to_exp]
            success_rate = exp_perf['outcome']['<lambda>'][days_to_exp]
            output += f"- {days_to_exp} days: {avg_return:.2%} avg return, {success_rate:.1%} success rate ({count} trades)\n"

        return output or "No expiration data available"

    def _format_performance_by_confidence(self, performance_summary: Dict[str, Any]) -> str:
        """Format performance by confidence for prompt"""
        conf_perf = performance_summary.get('performance_by_confidence', {})
        if not conf_perf:
            return "No data available"

        output = ""
        for confidence, metrics in conf_perf.get('actual_return', {}).get('mean', {}).items():
            avg_return = metrics
            count = conf_perf['actual_return']['count'][confidence]
            success_rate = conf_perf['outcome']['<lambda>'][confidence]
            output += f"- {confidence} confidence: {avg_return:.2%} avg return, {success_rate:.1%} success rate ({count} trades)\n"

        return output or "No confidence data available"

    def _analyze_options_market_regime(self, performance_summary: Dict[str, Any]) -> str:
        """Analyze how market conditions affected options performance"""

        # This would ideally analyze VIX, market direction, etc.
        # Simplified version for now

        best_type = performance_summary.get('best_performing_type', 'Unknown')
        best_expiration = performance_summary.get('best_performing_expiration', 'Unknown')

        analysis = f"""
Market Regime Analysis:
- Best performing option type: {best_type}
- Optimal expiration timeframe: {best_expiration} days
- Market conditions appear to favor {'bullish strategies' if best_type == 'CALL' else 'bearish strategies' if best_type == 'PUT' else 'mixed strategies'}
"""

        return analysis

    def _extract_success_rates_by_type(self, performance_summary: Dict[str, Any]) -> Dict[str, float]:
        """Extract success rates by option type"""
        type_perf = performance_summary.get('performance_by_type', {})
        success_rates = {}

        for option_type in ['CALL', 'PUT']:
            if option_type in type_perf.get('outcome', {}).get('<lambda>', {}):
                success_rates[option_type] = type_perf['outcome']['<lambda>'][option_type]

        return success_rates

    def _extract_success_rates_by_expiration(self, performance_summary: Dict[str, Any]) -> Dict[int, float]:
        """Extract success rates by expiration"""
        exp_perf = performance_summary.get('performance_by_expiration', {})
        success_rates = {}

        for days_to_exp, rate in exp_perf.get('outcome', {}).get('<lambda>', {}).items():
            success_rates[days_to_exp] = rate

        return success_rates

    def _extract_avg_returns_by_type(self, performance_summary: Dict[str, Any]) -> Dict[str, float]:
        """Extract average returns by option type"""
        type_perf = performance_summary.get('performance_by_type', {})
        avg_returns = {}

        for option_type in ['CALL', 'PUT']:
            if option_type in type_perf.get('actual_return', {}).get('mean', {}):
                avg_returns[option_type] = type_perf['actual_return']['mean'][option_type]

        return avg_returns

    def _extract_avg_returns_by_expiration(self, performance_summary: Dict[str, Any]) -> Dict[int, float]:
        """Extract average returns by expiration"""
        exp_perf = performance_summary.get('performance_by_expiration', {})
        avg_returns = {}

        for days_to_exp, return_val in exp_perf.get('actual_return', {}).get('mean', {}).items():
            avg_returns[days_to_exp] = return_val

        return avg_returns

    def _save_options_feedback(self, feedback: OptionsLLMFeedback):
        """Save options feedback to database"""
        conn = sqlite3.connect(self.tracker.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO options_llm_feedback (
                feedback_date, ticker, period_start, period_end, total_predictions,
                success_rate_by_type, success_rate_by_expiration, avg_return_by_type,
                avg_return_by_expiration, reflection, improvements, market_regime_analysis
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback.feedback_date,
            feedback.ticker,
            feedback.period_start,
            feedback.period_end,
            feedback.total_predictions,
            json.dumps(feedback.success_rate_by_type),
            json.dumps(feedback.success_rate_by_expiration),
            json.dumps(feedback.avg_return_by_type),
            json.dumps(feedback.avg_return_by_expiration),
            feedback.reflection,
            feedback.improvements,
            feedback.market_regime_analysis
        ))

        conn.commit()
        conn.close()

    def _extract_improvements_from_reflection(self, reflection_text: str) -> str:
        """
        Extract actionable improvements from LLM reflection text.

        Tries multiple extraction strategies:
        1. Parse as JSON and extract 'improvements' key
        2. Use regex to find IMPROVEMENTS/RECOMMENDATIONS sections
        3. Fallback to returning the full text

        Args:
            reflection_text: The raw reflection text from the LLM

        Returns:
            str: Extracted improvements text
        """
        if not reflection_text or not reflection_text.strip():
            return "No specific improvements identified from reflection."

        import json
        import re

        # Strategy 1: Try to parse as JSON
        try:
            # Clean potential markdown code blocks
            cleaned_text = reflection_text.strip()
            if cleaned_text.startswith('```'):
                # Remove markdown code fence
                cleaned_text = re.sub(r'^```(?:json)?\s*', '', cleaned_text)
                cleaned_text = re.sub(r'```\s*$', '', cleaned_text)

            reflection_data = json.loads(cleaned_text)

            # Extract improvements from various possible keys
            if isinstance(reflection_data, dict):
                improvements = (
                    reflection_data.get('improvements') or
                    reflection_data.get('IMPROVEMENTS') or
                    reflection_data.get('recommendations') or
                    reflection_data.get('RECOMMENDATIONS') or
                    reflection_data.get('action_items') or
                    reflection_data.get('key_learnings')
                )

                if improvements:
                    # If it's a list, join into string
                    if isinstance(improvements, list):
                        return '\n- ' + '\n- '.join(str(item) for item in improvements)
                    return str(improvements)

        except (json.JSONDecodeError, ValueError):
            pass  # Not JSON, try other strategies

        # Strategy 2: Regex extraction of IMPROVEMENTS section
        patterns = [
            r'(?:IMPROVEMENTS?|RECOMMENDATIONS?):\s*\n(.*?)(?:\n\n|$)',
            r'(?:## IMPROVEMENTS?|## RECOMMENDATIONS?)\s*\n(.*?)(?:\n##|$)',
            r'(?:2\.\s*IMPROVEMENTS?|2\.\s*RECOMMENDATIONS?):\s*\n(.*?)(?:\n\d+\.|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, reflection_text, re.IGNORECASE | re.DOTALL)
            if match:
                improvements = match.group(1).strip()
                if improvements and len(improvements) > 20:  # Minimum length check
                    return improvements

        # Strategy 3: Look for bullet points or numbered lists after keywords
        improvement_keywords = r'(?:improve|recommend|suggest|should|consider|optimize|enhance|adjust)'
        lines = reflection_text.split('\n')

        improvement_lines = []
        in_improvement_section = False

        for line in lines:
            line = line.strip()
            # Check if line starts improvement section
            if re.search(r'(?:IMPROVEMENT|RECOMMENDATION|ACTION)', line, re.IGNORECASE):
                in_improvement_section = True
                continue

            # Stop if we hit another major section
            if in_improvement_section and re.match(r'^(?:\d+\.|\w+:|\#\#)', line):
                if not re.search(improvement_keywords, line, re.IGNORECASE):
                    break

            # Collect improvement lines
            if in_improvement_section or re.search(improvement_keywords, line, re.IGNORECASE):
                if line and (line.startswith('-') or line.startswith('•') or
                           line.startswith('*') or re.match(r'^\d+\.', line)):
                    improvement_lines.append(line)

        if improvement_lines:
            return '\n'.join(improvement_lines)

        # Strategy 4: Fallback - return second half of text (improvements usually come after reflection)
        sentences = reflection_text.split('. ')
        if len(sentences) > 3:
            midpoint = len(sentences) // 2
            improvements = '. '.join(sentences[midpoint:])
            if improvements:
                return improvements.strip()

        # Last resort: return full text with a note
        return f"[Full reflection - improvements embedded]\n\n{reflection_text[:500]}..."

    def get_enhanced_options_analysis_prompt(self, base_prompt: str, ticker: str, config=None) -> str:
        """
        Enhance options analysis prompt with historical options performance insights.

        Retrieves past options performance for the ticker and injects key learnings
        and improvements into the analysis prompt, enabling continuous self-improvement.

        Args:
            base_prompt: The base options analysis prompt
            ticker: Stock ticker symbol
            config: Configuration object (optional)

        Returns:
            str: Enhanced prompt with historical performance context
        """
        enhanced_prompt = base_prompt

        # Check if historical context is enabled
        if config and not getattr(config, 'use_historical_context', True):
            return enhanced_prompt

        # Get most recent options feedback from database
        conn = sqlite3.connect(self.tracker.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM options_llm_feedback
            WHERE ticker = ?
            ORDER BY feedback_date DESC
            LIMIT 1
        ''', (ticker,))

        recent_feedback = cursor.fetchone()

        if recent_feedback:
            columns = [description[0] for description in cursor.description]
            feedback_dict = dict(zip(columns, recent_feedback))

            # Parse JSON fields
            try:
                success_rate_by_type = json.loads(feedback_dict['success_rate_by_type']) if isinstance(feedback_dict['success_rate_by_type'], str) else feedback_dict['success_rate_by_type']
                success_rate_by_expiration = json.loads(feedback_dict['success_rate_by_expiration']) if isinstance(feedback_dict['success_rate_by_expiration'], str) else feedback_dict['success_rate_by_expiration']
                avg_return_by_type = json.loads(feedback_dict['avg_return_by_type']) if isinstance(feedback_dict['avg_return_by_type'], str) else feedback_dict['avg_return_by_type']
                avg_return_by_expiration = json.loads(feedback_dict['avg_return_by_expiration']) if isinstance(feedback_dict['avg_return_by_expiration'], str) else feedback_dict['avg_return_by_expiration']
            except (json.JSONDecodeError, KeyError):
                success_rate_by_type = {}
                success_rate_by_expiration = {}
                avg_return_by_type = {}
                avg_return_by_expiration = {}

            # Format success rates by type
            type_performance = ""
            if success_rate_by_type:
                for opt_type, rate in success_rate_by_type.items():
                    avg_return = avg_return_by_type.get(opt_type, 0)
                    type_performance += f"- {opt_type} Success Rate: {rate:.1%} | Avg Return: {avg_return:+.1%}\n"

            # Format success rates by expiration
            exp_performance = ""
            if success_rate_by_expiration:
                # Sort by expiration days
                sorted_expirations = sorted(success_rate_by_expiration.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)
                for exp_days, rate in sorted_expirations:
                    avg_return = avg_return_by_expiration.get(str(exp_days), 0)
                    exp_performance += f"- {exp_days}-day Options: {rate:.1%} success | {avg_return:+.1%} avg return\n"

            # Build historical context section
            performance_context = f"""

## HISTORICAL OPTIONS PERFORMANCE CONTEXT
Based on {feedback_dict['total_predictions']} recent {ticker} options predictions:

### Performance by Option Type:
{type_performance if type_performance else "- Insufficient data for type-based analysis"}

### Performance by Expiration:
{exp_performance if exp_performance else "- Insufficient data for expiration analysis"}

### Market Regime Analysis:
{feedback_dict['market_regime_analysis']}

### Key Learnings from Past Performance:
{feedback_dict['reflection']}

### Apply These Improvements to Current Analysis:
{feedback_dict['improvements']}

---
IMPORTANT: Consider this historical context when evaluating the current option.
Prioritize strategies that have worked well and avoid patterns that led to losses.
"""
            enhanced_prompt = enhanced_prompt + performance_context

        conn.close()
        return enhanced_prompt

    def _create_default_options_feedback(self, ticker: str, days: int) -> OptionsLLMFeedback:
        """Create default feedback when insufficient data is available"""
        return OptionsLLMFeedback(
            feedback_date=datetime.now().strftime('%Y-%m-%d'),
            ticker=ticker,
            period_start=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            period_end=datetime.now().strftime('%Y-%m-%d'),
            total_predictions=0,
            success_rate_by_type={},
            success_rate_by_expiration={},
            avg_return_by_type={},
            avg_return_by_expiration={},
            reflection="Insufficient data available for meaningful reflection. Continue building prediction history.",
            improvements="Focus on generating more predictions to build a solid performance baseline.",
            market_regime_analysis="Market regime analysis requires more historical data points."
        )