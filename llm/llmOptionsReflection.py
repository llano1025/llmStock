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

from .llmTracking import PerformanceEvaluator, LLMReflectionEngine
from .llm_models import OptionsPerformanceRecord, OptionsLLMFeedback
from .llmOptionsAnalysis import OptionsTracker

logger = logging.getLogger(__name__)


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
                'performance_by_type': type_performance.to_dict() if isinstance(type_performance, pd.DataFrame) else {},
                'performance_by_expiration': exp_performance.to_dict() if isinstance(exp_performance, pd.DataFrame) else {},
                'performance_by_confidence': conf_performance.to_dict() if isinstance(conf_performance, pd.DataFrame) else {},
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