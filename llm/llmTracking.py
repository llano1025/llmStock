import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
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
        
    def get_recent_analyses(self, ticker: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get recent analysis summaries for consistency reference"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT prediction_date, recommendation, confidence, llm_analysis
            FROM predictions 
            WHERE ticker = ?
            ORDER BY prediction_date DESC 
            LIMIT ?
        ''', (ticker, limit))
        
        columns = [description[0] for description in cursor.description]
        analyses = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return analyses
        
    def get_analysis_style_summary(self, ticker: str, days: int = 90) -> Dict[str, Any]:
        """Get analysis style patterns for consistency"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT recommendation, confidence, llm_analysis, 
                   technical_indicators, sentiment_data
            FROM predictions 
            WHERE ticker = ? AND prediction_date >= date('now', '-{} days')
            ORDER BY prediction_date DESC
        '''.format(days), (ticker,))
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        
        if not results:
            return {'patterns': [], 'style_summary': ''}
            
        # Extract patterns
        recommendations = [r['recommendation'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        style_summary = f"""
        Recent analysis patterns for {ticker}:
        - Most common recommendation: {max(set(recommendations), key=recommendations.count)}
        - Most common confidence level: {max(set(confidences), key=confidences.count)}
        - Total analyses: {len(results)}
        """
        
        return {
            'patterns': results[:3],  # Last 3 for reference
            'style_summary': style_summary.strip()
        }

    def get_top_tickers_by_prediction_count(self, limit: int = 30, days: int = 90,
                                           status: str = 'CLOSED') -> List[Tuple[str, int]]:
        """Get top N tickers by prediction count within a time window

        Args:
            limit: Maximum number of tickers to return (default: 30)
            days: Time window in days to count predictions (default: 90)
            status: Filter by prediction status ('CLOSED', 'OPEN', or None for all)

        Returns:
            List of (ticker, count) tuples ordered by count descending
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if status:
            query = '''
                SELECT ticker, COUNT(*) as prediction_count
                FROM predictions
                WHERE status = ?
                  AND prediction_date >= datetime('now', '-{} days')
                GROUP BY ticker
                ORDER BY prediction_count DESC, ticker ASC
                LIMIT ?
            '''.format(days)
            cursor.execute(query, (status, limit))
        else:
            query = '''
                SELECT ticker, COUNT(*) as prediction_count
                FROM predictions
                WHERE prediction_date >= datetime('now', '-{} days')
                GROUP BY ticker
                ORDER BY prediction_count DESC, ticker ASC
                LIMIT ?
            '''.format(days)
            cursor.execute(query, (limit,))

        results = cursor.fetchall()
        conn.close()

        return results

    def get_performance_stats(self, ticker: Optional[str] = None,
                            days: int = 90) -> Dict[str, Any]:
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

        # Helper function to convert grouped stats to JSON-serializable dict
        def group_stats_to_dict(df, group_column):
            """Convert grouped aggregation to dict with string keys"""
            grouped = df.groupby(group_column).agg({
                'actual_return': ['mean', 'std', 'count'],
                'outcome': lambda x: (x == 'SUCCESS').mean()
            })

            result = {}
            for idx in grouped.index:
                result[str(idx)] = {
                    'actual_return_mean': float(grouped.loc[idx, ('actual_return', 'mean')]),
                    'actual_return_std': float(grouped.loc[idx, ('actual_return', 'std')]),
                    'actual_return_count': int(grouped.loc[idx, ('actual_return', 'count')]),
                    'success_rate': float(grouped.loc[idx, ('outcome', '<lambda>')])
                }
            return result

        stats = {
            'total_predictions': len(df),
            'success_rate': float((df['outcome'] == 'SUCCESS').mean()),
            'avg_return': float(df['actual_return'].mean()),
            'by_confidence': group_stats_to_dict(df, 'confidence'),
            'by_recommendation': group_stats_to_dict(df, 'recommendation')
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
        
        # Check stop loss first (highest priority)
        if prediction['stop_loss'] and current_price <= prediction['stop_loss']:
            should_close = True
            outcome = 'FAILURE'
            
        # Check target price (second priority)
        elif prediction['target_price'] and current_price >= prediction['target_price']:
            should_close = True
            outcome = 'SUCCESS'
            
        # Check if we've reached the timeframe (third priority)
        elif days_held >= expected_days:
            should_close = True
            # outcome remains 'OPEN' - will be set based on performance below
            
        if not should_close and days_held < 3:
            return None  # Too early to evaluate
            
        # If we should close but haven't yet, ensure we process it
        if should_close:
            pass  # Continue processing to close the prediction
            
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
            else:  # HOLD recommendation
                if abs(actual_return) < 1:
                    outcome = 'SUCCESS'
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
            if len(market_data) <= 1:
                return 0
                
            if days > 0 and len(market_data) > days:
                # We have enough data for the full period
                start_price = market_data['Close'].iloc[-days-1]  # Go back 'days' from the end
                end_price = market_data['Close'].iloc[-1]
                return (end_price - start_price) / start_price * 100
            elif len(market_data) > 1:
                # Use all available data if we don't have enough days
                return (market_data['Close'].iloc[-1] - market_data['Close'].iloc[0]) / market_data['Close'].iloc[0] * 100
        except Exception as e:
            logger.warning(f"Error calculating market return: {e}")
        return 0


class LLMReflectionEngine:
    """Generate reflections and improvements based on performance"""
    
    def __init__(self, llm_provider, tracker: PredictionTracker):
        self.llm = llm_provider
        self.tracker = tracker
        
    def generate_reflection(self, ticker: str, days: int = 90) -> Dict[str, str]:
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
        
    def get_enhanced_analysis_prompt(self, base_prompt: str, ticker: str, config=None) -> str:
        """Enhance analysis prompt with historical performance insights and consistency controls"""
        
        enhanced_prompt = base_prompt
        
        # Add historical context if enabled
        if config and config.use_historical_context:
            # Get recent analyses for style consistency
            recent_analyses = self.tracker.get_recent_analyses(ticker, limit=2)
            
            if recent_analyses and config.analysis_style_consistency:
                style_context = "\n\n## CONSISTENCY REFERENCE\n"
                style_context += "Maintain consistency with these recent analyses:\n\n"
                
                for i, analysis in enumerate(recent_analyses[:2], 1):
                    # Extract key summary from previous analysis
                    analysis_summary = analysis['llm_analysis'][:500] + "..." if len(analysis['llm_analysis']) > 500 else analysis['llm_analysis']
                    style_context += f"Analysis {i} ({analysis['prediction_date']}):\n"
                    style_context += f"- Recommendation: {analysis['recommendation']} (Confidence: {analysis['confidence']})\n"
                    style_context += f"- Analysis approach: {analysis_summary}\n\n"
                
                style_context += "CONSISTENCY INSTRUCTIONS:\n"
                style_context += "- Use similar analytical depth and structure\n"
                style_context += "- Maintain consistent confidence calibration\n"
                style_context += "- Apply similar risk assessment frameworks\n"
                style_context += "- Reference previous insights when relevant\n\n"
                
                enhanced_prompt = style_context + enhanced_prompt
        
        # Get recent reflections for performance context
        conn = sqlite3.connect(self.tracker.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM llm_feedback
            WHERE ticker = ?
            ORDER BY feedback_date DESC
            LIMIT 1
        ''', (ticker,))
        
        recent_feedback = cursor.fetchone()
        
        if recent_feedback:
            columns = [description[0] for description in cursor.description]
            feedback_dict = dict(zip(columns, recent_feedback))
            
            performance_context = f"""

## HISTORICAL PERFORMANCE CONTEXT
Based on analysis of {feedback_dict['total_predictions']} recent predictions:
- Success Rate: {feedback_dict['success_rate']:.2%}
- Average Return: {feedback_dict['avg_return']:.2f}%

Key Learnings:
{feedback_dict['reflection']}

Apply these improvements:
{feedback_dict['improvements']}
"""
            enhanced_prompt = enhanced_prompt + performance_context
            
        conn.close()
        return enhanced_prompt


class ConsistencyValidator:
    """Validate analysis consistency and provide metrics"""
    
    def __init__(self, tracker: PredictionTracker):
        self.tracker = tracker
        
    def calculate_analysis_similarity(self, analysis1: str, analysis2: str) -> float:
        """Calculate similarity between two analyses using basic text similarity"""
        try:
            # Simple word overlap similarity
            words1 = set(analysis1.lower().split())
            words2 = set(analysis2.lower().split())
            
            if not words1 or not words2:
                return 0.0
                
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
        except Exception:
            return 0.0
    
    def validate_consistency(self, ticker: str) -> Dict[str, Any]:
        """Validate consistency of recent analyses"""
        recent_analyses = self.tracker.get_recent_analyses(ticker, limit=5)
        
        if len(recent_analyses) < 2:
            return {
                'consistency_score': 1.0,
                'recommendation_consistency': 1.0,
                'confidence_consistency': 1.0,
                'analysis_text_similarity': 1.0,
                'validation_summary': 'Insufficient data for consistency validation'
            }
        
        # Check recommendation consistency
        recommendations = [a['recommendation'] for a in recent_analyses]
        rec_consistency = len(set(recommendations)) / len(recommendations)
        
        # Check confidence consistency
        confidences = [a['confidence'] for a in recent_analyses]
        conf_consistency = len(set(confidences)) / len(confidences)
        
        # Check text similarity (compare consecutive analyses)
        similarities = []
        for i in range(len(recent_analyses) - 1):
            sim = self.calculate_analysis_similarity(
                recent_analyses[i]['llm_analysis'],
                recent_analyses[i + 1]['llm_analysis']
            )
            similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
        
        # Overall consistency score (lower is more consistent)
        consistency_score = (rec_consistency + conf_consistency + (1 - avg_similarity)) / 3
        
        return {
            'consistency_score': 1 - consistency_score,  # Invert so higher is better
            'recommendation_consistency': 1 - rec_consistency,
            'confidence_consistency': 1 - conf_consistency,
            'analysis_text_similarity': avg_similarity,
            'validation_summary': f'Analyzed {len(recent_analyses)} recent predictions'
        }
    
    def generate_consistency_report(self, ticker: str) -> str:
        """Generate a consistency validation report"""
        validation = self.validate_consistency(ticker)
        
        report = f"""
## Consistency Validation Report for {ticker}

**Overall Consistency Score**: {validation['consistency_score']:.2%}

**Metrics:**
- Recommendation Consistency: {validation['recommendation_consistency']:.2%}
- Confidence Level Consistency: {validation['confidence_consistency']:.2%}
- Analysis Text Similarity: {validation['analysis_text_similarity']:.2%}

**Status**: {validation['validation_summary']}

**Interpretation:**
- Higher scores indicate more consistent analysis
- Recommendation consistency shows how often the same action is recommended
- Text similarity indicates consistent analytical approach
"""
        return report
