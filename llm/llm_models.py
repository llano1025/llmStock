"""
Shared data models for LLM analysis and tracking modules.
This module contains common dataclasses and types to avoid circular imports.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any


@dataclass
class PredictionRecord:
    """Record of a single prediction made by the system"""
    ticker: str
    prediction_date: str
    recommendation: str  # BUY/SELL/HOLD
    confidence: str  # HIGH/MEDIUM/LOW
    entry_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    predicted_timeframe: str  # short-term/medium-term/long-term
    technical_indicators: Dict[str, float]
    llm_analysis: str
    sentiment_data: Dict[str, Any]
    

@dataclass
class PerformanceRecord:
    """Record of actual performance vs prediction"""
    prediction_id: int
    ticker: str
    outcome_date: str
    exit_price: float
    actual_return: float
    predicted_return: float
    days_held: int
    outcome: str  # SUCCESS/PARTIAL/FAILURE
    market_condition: Dict[str, float]


@dataclass
class LLMFeedback:
    """Feedback from LLM reflection on prediction performance"""
    feedback_date: str
    ticker: str
    period_start: str
    period_end: str
    total_predictions: int
    success_rate: float
    avg_return: float
    reflection: str
    improvements: str


@dataclass
class MarketCondition:
    """Market condition data for context"""
    date: str
    spy_price: float
    vix_value: float
    ten_year_yield: float
    sentiment_score: float
    volatility: float