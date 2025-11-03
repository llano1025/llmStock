"""
Earnings-specific prediction tracking and performance evaluation system.
Tracks pre-earnings and post-earnings predictions separately.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import logging
from .llm_models import EarningsPredictionRecord, EarningsPerformanceRecord, EarningsLLMFeedback

logger = logging.getLogger(__name__)


class EarningsPredictionTracker:
    """Track earnings predictions and performance over time"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for tracking earnings predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create earnings predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS earnings_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                earnings_date TEXT NOT NULL,
                days_until_earnings INTEGER NOT NULL,
                pre_earnings_recommendation TEXT NOT NULL,
                post_earnings_direction TEXT NOT NULL,
                confidence TEXT NOT NULL,
                entry_price REAL NOT NULL,
                pre_earnings_target REAL,
                post_earnings_target REAL,
                stop_loss REAL,
                eps_estimate REAL,
                eps_actual REAL,
                revenue_estimate REAL,
                revenue_actual REAL,
                technical_indicators TEXT NOT NULL,
                llm_analysis TEXT NOT NULL,
                sentiment_data TEXT NOT NULL,
                earnings_context TEXT NOT NULL,
                status TEXT DEFAULT 'OPEN'
            )
        ''')

        # Create earnings performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS earnings_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                earnings_date TEXT NOT NULL,
                outcome_date TEXT NOT NULL,
                pre_earnings_exit_price REAL,
                post_earnings_price REAL NOT NULL,
                pre_earnings_return REAL,
                post_earnings_return REAL NOT NULL,
                total_return REAL NOT NULL,
                days_held_pre INTEGER,
                days_held_post INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                earnings_beat_miss TEXT NOT NULL,
                post_earnings_volatility REAL NOT NULL,
                market_condition TEXT NOT NULL,
                FOREIGN KEY (prediction_id) REFERENCES earnings_predictions(id)
            )
        ''')

        # Create earnings LLM feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS earnings_llm_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                total_predictions INTEGER NOT NULL,
                pre_earnings_success_rate REAL NOT NULL,
                post_earnings_success_rate REAL NOT NULL,
                avg_pre_earnings_return REAL NOT NULL,
                avg_post_earnings_return REAL NOT NULL,
                success_rate_by_beat_miss TEXT NOT NULL,
                avg_return_by_beat_miss TEXT NOT NULL,
                reflection TEXT NOT NULL,
                improvements TEXT NOT NULL,
                earnings_pattern_analysis TEXT NOT NULL
            )
        ''')

        conn.commit()
        conn.close()
        logger.info(f"Initialized earnings tracking database at {self.db_path}")

    def record_prediction(self, prediction: EarningsPredictionRecord) -> int:
        """Record a new earnings prediction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO earnings_predictions (
                    ticker, prediction_date, earnings_date, days_until_earnings,
                    pre_earnings_recommendation, post_earnings_direction, confidence,
                    entry_price, pre_earnings_target, post_earnings_target, stop_loss,
                    eps_estimate, eps_actual, revenue_estimate, revenue_actual,
                    technical_indicators, llm_analysis, sentiment_data, earnings_context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction.ticker,
                prediction.prediction_date,
                prediction.earnings_date,
                prediction.days_until_earnings,
                prediction.pre_earnings_recommendation,
                prediction.post_earnings_direction,
                prediction.confidence,
                prediction.entry_price,
                prediction.pre_earnings_target,
                prediction.post_earnings_target,
                prediction.stop_loss,
                prediction.eps_estimate,
                prediction.eps_actual,
                prediction.revenue_estimate,
                prediction.revenue_actual,
                json.dumps(prediction.technical_indicators),
                prediction.llm_analysis,
                json.dumps(prediction.sentiment_data),
                json.dumps(prediction.earnings_context)
            ))

            prediction_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Recorded earnings prediction for {prediction.ticker} (ID: {prediction_id})")
            return prediction_id

        except Exception as e:
            logger.error(f"Error recording earnings prediction: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def update_actual_earnings_data(self, prediction_id: int, eps_actual: Optional[float],
                                    revenue_actual: Optional[float]) -> bool:
        """Update prediction with actual earnings results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                UPDATE earnings_predictions
                SET eps_actual = ?, revenue_actual = ?
                WHERE id = ?
            ''', (eps_actual, revenue_actual, prediction_id))

            conn.commit()
            logger.info(f"Updated actual earnings data for prediction ID: {prediction_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating actual earnings data: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def record_performance(self, performance: EarningsPerformanceRecord) -> int:
        """Record actual performance vs earnings prediction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO earnings_performance (
                    prediction_id, ticker, earnings_date, outcome_date,
                    pre_earnings_exit_price, post_earnings_price,
                    pre_earnings_return, post_earnings_return, total_return,
                    days_held_pre, days_held_post, outcome, earnings_beat_miss,
                    post_earnings_volatility, market_condition
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                performance.prediction_id,
                performance.ticker,
                performance.earnings_date,
                performance.outcome_date,
                performance.pre_earnings_exit_price,
                performance.post_earnings_price,
                performance.pre_earnings_return,
                performance.post_earnings_return,
                performance.total_return,
                performance.days_held_pre,
                performance.days_held_post,
                performance.outcome,
                performance.earnings_beat_miss,
                performance.post_earnings_volatility,
                json.dumps(performance.market_condition)
            ))

            # Update prediction status to CLOSED
            cursor.execute('''
                UPDATE earnings_predictions
                SET status = 'CLOSED'
                WHERE id = ?
            ''', (performance.prediction_id,))

            performance_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Recorded earnings performance for {performance.ticker} (ID: {performance_id})")
            return performance_id

        except Exception as e:
            logger.error(f"Error recording earnings performance: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def record_llm_feedback(self, feedback: EarningsLLMFeedback) -> int:
        """Record LLM reflection feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO earnings_llm_feedback (
                    feedback_date, ticker, period_start, period_end,
                    total_predictions, pre_earnings_success_rate, post_earnings_success_rate,
                    avg_pre_earnings_return, avg_post_earnings_return,
                    success_rate_by_beat_miss, avg_return_by_beat_miss,
                    reflection, improvements, earnings_pattern_analysis
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.feedback_date,
                feedback.ticker,
                feedback.period_start,
                feedback.period_end,
                feedback.total_predictions,
                feedback.pre_earnings_success_rate,
                feedback.post_earnings_success_rate,
                feedback.avg_pre_earnings_return,
                feedback.avg_post_earnings_return,
                json.dumps(feedback.success_rate_by_beat_miss),
                json.dumps(feedback.avg_return_by_beat_miss),
                feedback.reflection,
                feedback.improvements,
                feedback.earnings_pattern_analysis
            ))

            feedback_id = cursor.lastrowid
            conn.commit()
            logger.info(f"Recorded earnings LLM feedback (ID: {feedback_id})")
            return feedback_id

        except Exception as e:
            logger.error(f"Error recording earnings LLM feedback: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_open_predictions(self, days_threshold: int = 30) -> List[Dict]:
        """Get open earnings predictions within threshold days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_date = (datetime.now() - timedelta(days=days_threshold)).strftime('%Y-%m-%d')

        cursor.execute('''
            SELECT * FROM earnings_predictions
            WHERE status = 'OPEN' AND prediction_date >= ?
            ORDER BY earnings_date ASC
        ''', (cutoff_date,))

        columns = [description[0] for description in cursor.description]
        predictions = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return predictions

    def get_predictions_by_ticker(self, ticker: str, limit: int = 10) -> List[Dict]:
        """Get recent predictions for a specific ticker"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM earnings_predictions
            WHERE ticker = ?
            ORDER BY prediction_date DESC
            LIMIT ?
        ''', (ticker, limit))

        columns = [description[0] for description in cursor.description]
        predictions = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return predictions

    def get_performance_statistics(self, days: int = 90) -> Dict[str, Any]:
        """Get performance statistics for the last N days"""
        conn = sqlite3.connect(self.db_path)

        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        query = '''
            SELECT
                COUNT(*) as total_predictions,
                AVG(CASE WHEN outcome = 'SUCCESS' THEN 1 ELSE 0 END) as success_rate,
                AVG(pre_earnings_return) as avg_pre_earnings_return,
                AVG(post_earnings_return) as avg_post_earnings_return,
                AVG(total_return) as avg_total_return,
                AVG(CASE WHEN earnings_beat_miss = 'BEAT' THEN 1 ELSE 0 END) as beat_rate
            FROM earnings_performance
            WHERE outcome_date >= ?
        '''

        df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        conn.close()

        return df.to_dict('records')[0] if not df.empty else {}

    def get_performance_by_beat_miss(self, days: int = 90) -> pd.DataFrame:
        """Get performance statistics grouped by earnings beat/miss"""
        conn = sqlite3.connect(self.db_path)

        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        query = '''
            SELECT
                earnings_beat_miss,
                COUNT(*) as count,
                AVG(CASE WHEN outcome = 'SUCCESS' THEN 1 ELSE 0 END) as success_rate,
                AVG(total_return) as avg_return,
                AVG(post_earnings_return) as avg_post_earnings_return
            FROM earnings_performance
            WHERE outcome_date >= ?
            GROUP BY earnings_beat_miss
        '''

        df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        conn.close()

        return df


class EarningsPerformanceEvaluator:
    """Evaluate earnings predictions against actual outcomes"""

    def __init__(self, tracker: EarningsPredictionTracker):
        self.tracker = tracker

    def check_and_update_predictions(self, fetch_stock_data_func) -> List[Dict[str, Any]]:
        """
        Check open predictions and update performance if earnings have occurred.

        Args:
            fetch_stock_data_func: Function to fetch current stock data

        Returns:
            List of updated predictions with performance data
        """
        open_predictions = self.tracker.get_open_predictions()
        updated_predictions = []

        for pred in open_predictions:
            try:
                earnings_date = datetime.strptime(pred['earnings_date'], '%Y-%m-%d')
                current_date = datetime.now()

                # Check if earnings date has passed (give 1 day buffer)
                if earnings_date < current_date - timedelta(days=1):
                    result = self._evaluate_prediction(pred, fetch_stock_data_func)
                    if result:
                        updated_predictions.append(result)

            except Exception as e:
                logger.error(f"Error evaluating prediction {pred['id']}: {e}")
                continue

        return updated_predictions

    def _evaluate_prediction(self, prediction: Dict, fetch_stock_data_func) -> Optional[Dict[str, Any]]:
        """Evaluate a single prediction after earnings"""
        try:
            ticker = prediction['ticker']
            entry_price = prediction['entry_price']
            earnings_date = datetime.strptime(prediction['earnings_date'], '%Y-%m-%d')

            # Fetch current stock data
            current_data = fetch_stock_data_func(ticker, period='5d')

            if current_data is None or current_data.empty:
                logger.warning(f"Could not fetch data for {ticker}")
                return None

            current_price = current_data['Close'].iloc[-1]

            # Get price just before earnings (if available)
            pre_earnings_price = None
            try:
                # Try to get price from day before earnings
                pre_earnings_data = current_data[current_data.index.date < earnings_date.date()]
                if not pre_earnings_data.empty:
                    pre_earnings_price = pre_earnings_data['Close'].iloc[-1]
            except Exception as e:
                logger.debug(f"Could not get pre-earnings price: {e}")

            # Calculate returns
            if pre_earnings_price:
                pre_earnings_return = ((pre_earnings_price - entry_price) / entry_price) * 100
                post_earnings_return = ((current_price - pre_earnings_price) / pre_earnings_price) * 100
                days_held_pre = (earnings_date - datetime.strptime(prediction['prediction_date'], '%Y-%m-%d')).days
            else:
                pre_earnings_return = None
                post_earnings_return = None
                days_held_pre = None
                pre_earnings_price = entry_price  # Use entry price as fallback
                post_earnings_return = ((current_price - entry_price) / entry_price) * 100

            total_return = ((current_price - entry_price) / entry_price) * 100
            days_held_post = (datetime.now() - earnings_date).days

            # Determine earnings beat/miss/inline
            eps_estimate = prediction['eps_estimate']
            eps_actual = prediction['eps_actual']

            if eps_actual is not None and eps_estimate is not None:
                if eps_actual > eps_estimate * 1.02:  # 2% threshold
                    earnings_beat_miss = 'BEAT'
                elif eps_actual < eps_estimate * 0.98:
                    earnings_beat_miss = 'MISS'
                else:
                    earnings_beat_miss = 'INLINE'
            else:
                earnings_beat_miss = 'UNKNOWN'

            # Determine outcome
            post_earnings_dir = prediction['post_earnings_direction']
            if post_earnings_dir == 'UP' and post_earnings_return > 0:
                outcome = 'SUCCESS'
            elif post_earnings_dir == 'DOWN' and post_earnings_return < 0:
                outcome = 'SUCCESS'
            elif post_earnings_dir == 'NEUTRAL' and abs(post_earnings_return) < 2:
                outcome = 'SUCCESS'
            elif abs(post_earnings_return) > 1:  # Some positive movement
                outcome = 'PARTIAL'
            else:
                outcome = 'FAILURE'

            # Calculate post-earnings volatility
            post_earnings_volatility = current_data['Close'].pct_change().std() * 100

            # Create performance record
            performance = EarningsPerformanceRecord(
                prediction_id=prediction['id'],
                ticker=ticker,
                earnings_date=prediction['earnings_date'],
                outcome_date=datetime.now().strftime('%Y-%m-%d'),
                pre_earnings_exit_price=pre_earnings_price,
                post_earnings_price=current_price,
                pre_earnings_return=pre_earnings_return,
                post_earnings_return=post_earnings_return if post_earnings_return else total_return,
                total_return=total_return,
                days_held_pre=days_held_pre,
                days_held_post=days_held_post,
                outcome=outcome,
                earnings_beat_miss=earnings_beat_miss,
                post_earnings_volatility=post_earnings_volatility,
                market_condition={}  # Can be enhanced with market data
            )

            # Record performance
            self.tracker.record_performance(performance)

            return {
                'ticker': ticker,
                'outcome': outcome,
                'total_return': total_return,
                'post_earnings_return': post_earnings_return if post_earnings_return else total_return,
                'earnings_beat_miss': earnings_beat_miss
            }

        except Exception as e:
            logger.error(f"Error in _evaluate_prediction: {e}")
            return None


class EarningsLLMReflectionEngine:
    """Generate LLM-powered insights on earnings prediction performance"""

    def __init__(self, tracker: EarningsPredictionTracker, llm_provider):
        self.tracker = tracker
        self.llm_provider = llm_provider

    def generate_reflection(self, ticker: Optional[str] = None, days: int = 90) -> str:
        """Generate reflection on earnings prediction performance"""

        # Get performance statistics
        stats = self.tracker.get_performance_statistics(days)
        beat_miss_stats = self.tracker.get_performance_by_beat_miss(days)

        if not stats or stats.get('total_predictions', 0) == 0:
            return "Insufficient data for earnings reflection. No completed predictions in the specified period."

        # Prepare context for LLM
        context = f"""
        Earnings Prediction Performance Analysis ({days} days)

        Overall Statistics:
        - Total Predictions: {stats.get('total_predictions', 0)}
        - Success Rate: {stats.get('success_rate', 0)*100:.1f}%
        - Average Pre-Earnings Return: {stats.get('avg_pre_earnings_return', 0):.2f}%
        - Average Post-Earnings Return: {stats.get('avg_post_earnings_return', 0):.2f}%
        - Average Total Return: {stats.get('avg_total_return', 0):.2f}%
        - Earnings Beat Rate: {stats.get('beat_rate', 0)*100:.1f}%

        Performance by Earnings Outcome:
        {beat_miss_stats.to_string() if not beat_miss_stats.empty else 'No data available'}

        Please analyze this earnings prediction performance and provide:
        1. Key insights on what's working well
        2. Patterns in earnings beat/miss vs prediction accuracy
        3. Specific areas for improvement
        4. Recommendations for better earnings predictions
        """

        try:
            reflection = self.llm_provider.generate(context)
            return reflection
        except Exception as e:
            logger.error(f"Error generating earnings reflection: {e}")
            return "Error generating reflection. Please check the LLM provider."
