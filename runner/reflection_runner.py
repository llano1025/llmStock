"""
Reflection and testing functions extracted from main.py.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

from llm.llmAnalysis import Config, LLMProvider, OllamaProvider, EnhancedStockAnalyzer
from llm.llmTracking import PredictionTracker, LLMReflectionEngine
from llm.llm_models import PredictionRecord, PerformanceRecord

logger = logging.getLogger(__name__)


def _create_provider(provider_type: str, config):
    """Create LLM provider with auto-detection logic"""
    if provider_type == 'auto':
        if config.gemini_api_key:
            from llm.llmAnalysis import GeminiProvider
            return GeminiProvider(config=config)
        elif config.openai_api_key:
            from llm.llmAnalysis import OpenAIProvider
            return OpenAIProvider(config=config)
        else:
            return OllamaProvider(config=config)
    
    # Direct provider creation
    if provider_type == 'ollama':
        return OllamaProvider(config=config)
    elif provider_type == 'lmstudio':
        from llm.llmAnalysis import LMStudioProvider
        return LMStudioProvider(config=config)
    elif provider_type == 'gemini':
        from llm.llmAnalysis import GeminiProvider
        return GeminiProvider(config=config)
    elif provider_type == 'openai':
        from llm.llmAnalysis import OpenAIProvider
        return OpenAIProvider(config=config)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")


def run_weekly_reflection(provider_type='auto'):
    """Run weekly reflection analysis on all tracked tickers"""
    try:
        config = Config()
        llm_provider = _create_provider(provider_type, config)
        
        # Initialize tracker
        db_path = Path('predictions.db')
        tracker = PredictionTracker(db_path)
        reflection_engine = LLMReflectionEngine(llm_provider, tracker)
        
        # Get top 30 tickers with most predictions from last 90 days
        top_tickers = tracker.get_top_tickers_by_prediction_count(limit=30, days=90, status='CLOSED')

        if not top_tickers:
            logger.info("No tickers with closed predictions found in the last 90 days")
            return

        logger.info(f"Generating reflections for top {len(top_tickers)} tickers by prediction count (last 90 days)")
        for ticker, count in top_tickers:
            logger.info(f"  {ticker}: {count} predictions")

        reflections = {}
        for ticker, count in top_tickers:
            logger.info(f"Generating reflection for {ticker} ({count} predictions)")
            reflection = reflection_engine.generate_reflection(ticker, days=30)
            reflections[ticker] = reflection
            
        # Save reflections report
        try:
            from utils.utils_path import get_save_path
            save_path = get_save_path()
            os.makedirs(save_path, exist_ok=True)
            report_path = save_path / f"weekly_reflections_{datetime.now().strftime('%Y-%m-%d')}.json"
            with open(report_path, 'w') as f:
                json.dump(reflections, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save reflections: {e}")
            
        logger.info("Weekly reflection report completed")
        
    except Exception as e:
        logger.error(f"Error in weekly reflection: {e}")


def test_prediction_tracking(provider_type='auto'):
    """Test the prediction tracking system"""
    
    # Initialize
    config = Config()
    llm_provider = _create_provider(provider_type, config)
        
    analyzer = EnhancedStockAnalyzer(llm_provider, config)
    
    # Simulate some predictions
    test_prediction = PredictionRecord(
        ticker='AAPL',
        prediction_date='2025-01-01',
        recommendation='BUY',
        confidence='HIGH',
        entry_price=150.00,
        target_price=160.00,
        stop_loss=145.00,
        predicted_timeframe='short-term',
        technical_indicators={'RSI': 45, 'MACD': 0.5},
        llm_analysis='Test analysis',
        sentiment_data={'sentiment': 'positive'}
    )
    
    # Record prediction
    pred_id = analyzer.tracker.record_prediction(test_prediction)
    print(f"Recorded prediction with ID: {pred_id}")
    
    # Simulate performance after 7 days
    test_performance = PerformanceRecord(
        prediction_id=pred_id,
        ticker='AAPL',
        outcome_date='2025-01-08',
        exit_price=158.00,
        actual_return=5.33,
        predicted_return=6.67,
        days_held=7,
        outcome='SUCCESS',
        market_condition={'spy_return': 2.1}
    )
    
    analyzer.tracker.update_performance(test_performance)
    print("Updated performance")
    
    # Get stats
    stats = analyzer.tracker.get_performance_stats('AAPL', days=30)
    print(f"Performance stats: {stats}")
    
    # Generate reflection
    reflection = analyzer.reflection_engine.generate_reflection('AAPL', days=30)
    print(f"Reflection: {reflection}")