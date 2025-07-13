import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from .llm_models import PredictionRecord, PerformanceRecord

logger = logging.getLogger(__name__)
    
class PredictionTracker:
    """Track predictions and performance over time"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for tracking predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                confidence TEXT NOT NULL,
                entry_price REAL NOT NULL,
                target_price REAL,
                stop_loss REAL,
                predicted_timeframe TEXT NOT NULL,
                technical_indicators TEXT NOT NULL,
                llm_analysis TEXT NOT NULL,
                sentiment_data TEXT NOT NULL,
                status TEXT DEFAULT 'OPEN'
            )
        ''')
        
        # Create performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                outcome_date TEXT NOT NULL,
                exit_price REAL NOT NULL,
                actual_return REAL NOT NULL,
                predicted_return REAL NOT NULL,
                days_held INTEGER NOT NULL,
                outcome TEXT NOT NULL,
                market_condition TEXT NOT NULL,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id)
            )
        ''')
        
        # Create LLM feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS llm_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_date TEXT NOT NULL,
                ticker TEXT NOT NULL,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                total_predictions INTEGER NOT NULL,
                success_rate REAL NOT NULL,
                avg_return REAL NOT NULL,
                reflection TEXT NOT NULL,
                improvements TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def record_prediction(self, prediction: PredictionRecord) -> int:
        """Record a new prediction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (
                ticker, prediction_date, recommendation, confidence,
                entry_price, target_price, stop_loss, predicted_timeframe,
                technical_indicators, llm_analysis, sentiment_data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction.ticker,
            prediction.prediction_date,
            prediction.recommendation,
            prediction.confidence,
            prediction.entry_price,
            prediction.target_price,
            prediction.stop_loss,
            prediction.predicted_timeframe,
            json.dumps(prediction.technical_indicators),
            prediction.llm_analysis,
            json.dumps(prediction.sentiment_data)
        ))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return prediction_id
        
    def update_performance(self, performance: PerformanceRecord):
        """Update performance for a prediction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert performance record
        cursor.execute('''
            INSERT INTO performance (
                prediction_id, ticker, outcome_date, exit_price,
                actual_return, predicted_return, days_held, outcome,
                market_condition
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            performance.prediction_id,
            performance.ticker,
            performance.outcome_date,
            performance.exit_price,
            performance.actual_return,
            performance.predicted_return,
            performance.days_held,
            performance.outcome,
            json.dumps(performance.market_condition)
        ))
        
        # Update prediction status
        cursor.execute('''
            UPDATE predictions SET status = 'CLOSED' WHERE id = ?
        ''', (performance.prediction_id,))
        
        conn.commit()
        conn.close()
        
    def get_open_predictions(self) -> List[Dict[str, Any]]:
        """Get all open predictions that need to be tracked"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions WHERE status = 'OPEN'
        ''')
        
        columns = [description[0] for description in cursor.description]
        predictions = []
        
        for row in cursor.fetchall():
            pred_dict = dict(zip(columns, row))
            # Parse JSON fields
            pred_dict['technical_indicators'] = json.loads(pred_dict['technical_indicators'])
            pred_dict['sentiment_data'] = json.loads(pred_dict['sentiment_data'])
            predictions.append(pred_dict)
            
        conn.close()
        return predictions
        
    def get_performance_stats(self, ticker: Optional[str] = None, 
                            days: int = 30) -> Dict[str, Any]:
        """Get performance statistics for predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Base query
        query = '''
            SELECT p.*, perf.*
            FROM predictions p
            JOIN performance perf ON p.id = perf.prediction_id
            WHERE perf.outcome_date >= date('now', '-{} days')
        '''.format(days)
        
        if ticker:
            query += f" AND p.ticker = '{ticker}'"
            
        cursor.execute(query)
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        
        if not results:
            return {
                'total_predictions': 0,
                'success_rate': 0,
                'avg_return': 0,
                'by_confidence': {},
                'by_recommendation': {}
            }
            
        # Calculate statistics
        df = pd.DataFrame(results)
        
        stats = {
            'total_predictions': len(df),
            'success_rate': (df['outcome'] == 'SUCCESS').mean(),
            'avg_return': df['actual_return'].mean(),
            'by_confidence': df.groupby('confidence').agg({
                'actual_return': ['mean', 'std', 'count'],
                'outcome': lambda x: (x == 'SUCCESS').mean()
            }).to_dict(),
            'by_recommendation': df.groupby('recommendation').agg({
                'actual_return': ['mean', 'std', 'count'],
                'outcome': lambda x: (x == 'SUCCESS').mean()
            }).to_dict()
        }
        
        return stats


class PerformanceEvaluator:
    """Evaluate prediction performance against actual market data"""
    
    def __init__(self, tracker: PredictionTracker):
        self.tracker = tracker
        
    def evaluate_prediction(self, prediction: Dict[str, Any], 
                          current_price: float, 
                          market_data: pd.DataFrame) -> Optional[PerformanceRecord]:
        """Evaluate a single prediction against current market data"""
        
        prediction_date = datetime.strptime(prediction['prediction_date'], '%Y-%m-%d')
        days_held = (datetime.now() - prediction_date).days
        
        # Check if we should close the position
        should_close = False
        outcome = 'OPEN'
        
        # Determine timeframe in days
        timeframe_days = {
            'short-term': 7,
            'medium-term': 30,
            'long-term': 90
        }
        
        expected_days = timeframe_days.get(prediction['predicted_timeframe'], 30)
        
        # Check if we've reached the timeframe
        if days_held >= expected_days:
            should_close = True
            
        # Check stop loss
        if prediction['stop_loss'] and current_price <= prediction['stop_loss']:
            should_close = True
            outcome = 'FAILURE'
            
        # Check target price
        if prediction['target_price'] and current_price >= prediction['target_price']:
            should_close = True
            outcome = 'SUCCESS'
            
        if not should_close and days_held < 3:
            return None  # Too early to evaluate
            
        # Calculate returns
        actual_return = (current_price - prediction['entry_price']) / prediction['entry_price'] * 100
        
        # Calculate predicted return
        predicted_return = 0
        if prediction['target_price']:
            predicted_return = (prediction['target_price'] - prediction['entry_price']) / prediction['entry_price'] * 100
            
        # Determine outcome if not already set
        if outcome == 'OPEN':
            if prediction['recommendation'] == 'BUY':
                if actual_return > 2:
                    outcome = 'SUCCESS'
                elif actual_return < -2:
                    outcome = 'FAILURE'
                else:
                    outcome = 'PARTIAL'
            elif prediction['recommendation'] == 'SELL':
                if actual_return < -2:
                    outcome = 'SUCCESS'
                elif actual_return > 2:
                    outcome = 'FAILURE'
                else:
                    outcome = 'PARTIAL'
                    
        # Get current market conditions
        market_condition = {
            'spy_return': self._get_market_return(market_data, days_held),
            'volatility': market_data['ATR'].iloc[-1] if 'ATR' in market_data else 0,
            'volume_ratio': market_data['Volume'].iloc[-1] / market_data['Volume'].mean() if 'Volume' in market_data else 1
        }
        
        return PerformanceRecord(
            prediction_id=prediction['id'],
            ticker=prediction['ticker'],
            outcome_date=datetime.now().strftime('%Y-%m-%d'),
            exit_price=current_price,
            actual_return=actual_return,
            predicted_return=predicted_return,
            days_held=days_held,
            outcome=outcome,
            market_condition=market_condition
        )
        
    def _get_market_return(self, market_data: pd.DataFrame, days: int) -> float:
        """Calculate market return over the period"""
        try:
            if len(market_data) >= days:
                return (market_data['Close'].iloc[-1] - market_data['Close'].iloc[-days]) / market_data['Close'].iloc[-days] * 100
        except:
            pass
        return 0


class LLMReflectionEngine:
    """Generate reflections and improvements based on performance"""
    
    def __init__(self, llm_provider, tracker: PredictionTracker):
        self.llm = llm_provider
        self.tracker = tracker
        
    def generate_reflection(self, ticker: str, days: int = 30) -> Dict[str, str]:
        """Generate reflection on past predictions"""
        
        # Get performance statistics
        stats = self.tracker.get_performance_stats(ticker, days)
        
        if stats['total_predictions'] == 0:
            return {
                'reflection': 'No predictions to analyze',
                'improvements': 'Start making predictions to track performance'
            }
            
        # Get detailed prediction history
        conn = sqlite3.connect(self.tracker.db_path)
        query = '''
            SELECT p.*, perf.*
            FROM predictions p
            JOIN performance perf ON p.id = perf.prediction_id
            WHERE p.ticker = ? AND perf.outcome_date >= date('now', '-{} days')
            ORDER BY p.prediction_date DESC
        '''.format(days)
        
        cursor = conn.cursor()
        cursor.execute(query, (ticker,))
        
        columns = [description[0] for description in cursor.description]
        predictions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        
        # Create reflection prompt
        prompt = f"""Analyze the prediction performance for {ticker} over the past {days} days:

PERFORMANCE SUMMARY:
- Total Predictions: {stats['total_predictions']}
- Success Rate: {stats['success_rate']:.2%}
- Average Return: {stats['avg_return']:.2f}%

PERFORMANCE BY CONFIDENCE LEVEL:
{json.dumps(stats['by_confidence'], indent=2)}

PERFORMANCE BY RECOMMENDATION TYPE:
{json.dumps(stats['by_recommendation'], indent=2)}

RECENT PREDICTIONS AND OUTCOMES:
"""
        
        # Add recent predictions
        for pred in predictions[:5]:  # Last 5 predictions
            pred_indicators = json.loads(pred['technical_indicators']) if isinstance(pred['technical_indicators'], str) else pred['technical_indicators']
            
            prompt += f"""
- Date: {pred['prediction_date']}
  Recommendation: {pred['recommendation']} (Confidence: {pred['confidence']})
  Entry: ${pred['entry_price']:.2f}, Exit: ${pred['exit_price']:.2f}
  Return: {pred['actual_return']:.2f}% (Predicted: {pred['predicted_return']:.2f}%)
  Outcome: {pred['outcome']}
  Key Indicators: RSI={pred_indicators.get('RSI', 'N/A')}, MACD={pred_indicators.get('MACD_Signal', 'N/A')}
"""

        prompt += """
Based on this performance data:

1. REFLECTION: What patterns do you observe in successful vs failed predictions? Consider:
   - Which technical indicators were most reliable?
   - Were certain market conditions more favorable?
   - How accurate were confidence levels?
   
2. IMPROVEMENTS: Provide specific recommendations to improve future predictions:
   - Which indicators should be weighted more/less?
   - What additional factors should be considered?
   - How should confidence thresholds be adjusted?
   
3. RISK MANAGEMENT: Suggest improvements to risk management based on the outcomes.

Provide actionable insights in JSON format:
{
    "reflection": "detailed reflection on patterns and performance",
    "improvements": "specific actionable improvements",
    "key_learnings": ["learning1", "learning2", "learning3"],
    "adjusted_parameters": {
        "rsi_threshold": "suggested value",
        "confidence_mapping": "adjustments needed",
        "risk_metrics": "improvements"
    }
}"""

        # Get reflection from LLM
        response = self.llm.generate(prompt)
        
        try:
            reflection_data = json.loads(response)
        except:
            # Fallback parsing
            reflection_data = {
                'reflection': response,
                'improvements': 'See reflection above',
                'key_learnings': [],
                'adjusted_parameters': {}
            }
            
        # Save reflection to database
        conn = sqlite3.connect(self.tracker.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO llm_feedback (
                feedback_date, ticker, period_start, period_end,
                total_predictions, success_rate, avg_return,
                reflection, improvements
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().strftime('%Y-%m-%d'),
            ticker,
            (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d'),
            stats['total_predictions'],
            stats['success_rate'],
            stats['avg_return'],
            reflection_data.get('reflection', ''),
            reflection_data.get('improvements', '')
        ))
        
        conn.commit()
        conn.close()
        
        return reflection_data
        
    def get_enhanced_analysis_prompt(self, base_prompt: str, ticker: str) -> str:
        """Enhance analysis prompt with historical performance insights"""
        
        # Get recent reflections
        conn = sqlite3.connect(self.tracker.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM llm_feedback
            WHERE ticker = ?
            ORDER BY feedback_date DESC
            LIMIT 1
        ''', (ticker,))
        
        recent_feedback = cursor.fetchone()
        conn.close()
        
        if recent_feedback:
            columns = [description[0] for description in cursor.description]
            feedback_dict = dict(zip(columns, recent_feedback))
            
            enhancement = f"""

## HISTORICAL PERFORMANCE CONTEXT
Based on analysis of {feedback_dict['total_predictions']} recent predictions:
- Success Rate: {feedback_dict['success_rate']:.2%}
- Average Return: {feedback_dict['avg_return']:.2f}%

Key Learnings:
{feedback_dict['reflection']}

Apply these improvements:
{feedback_dict['improvements']}
"""
            return base_prompt + enhancement
            
        return base_prompt
