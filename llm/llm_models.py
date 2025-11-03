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


@dataclass
class OptionsPredictionRecord:
    """Record of a single options prediction made by the system"""
    ticker: str
    prediction_date: str
    option_type: str  # CALL/PUT
    strike_price: float
    expiration_date: str
    days_to_expiration: int  # 7, 14, 21, 28, 56, 84, 175
    recommendation: str  # BUY/SELL
    confidence: str  # HIGH/MEDIUM/LOW
    entry_premium: float
    target_premium: Optional[float]
    max_loss: Optional[float]
    underlying_price: float
    implied_volatility: float
    volume: int
    open_interest: int
    greeks: Dict[str, float]  # delta, gamma, theta, vega, rho
    technical_indicators: Dict[str, float]
    llm_analysis: str
    risk_factor: str
    sentiment_data: Dict[str, Any]


@dataclass
class OptionsPerformanceRecord:
    """Record of actual options performance vs prediction"""
    prediction_id: int
    ticker: str
    option_type: str
    outcome_date: str
    exit_premium: float
    actual_return: float
    predicted_return: float
    days_held: int
    outcome: str  # SUCCESS/PARTIAL/FAILURE
    max_profit_achieved: float
    underlying_move: float
    iv_change: float
    market_condition: Dict[str, float]


@dataclass
class OptionsLLMFeedback:
    """Feedback from LLM reflection on options prediction performance"""
    feedback_date: str
    ticker: str
    period_start: str
    period_end: str
    total_predictions: int
    success_rate_by_type: Dict[str, float]  # CALL vs PUT success rates
    success_rate_by_expiration: Dict[int, float]  # by days to expiration
    avg_return_by_type: Dict[str, float]
    avg_return_by_expiration: Dict[int, float]
    reflection: str
    improvements: str
    market_regime_analysis: str


@dataclass
class EarningsPredictionRecord:
    """Record of a single earnings prediction made by the system"""
    ticker: str
    prediction_date: str
    earnings_date: str
    days_until_earnings: int
    pre_earnings_recommendation: str  # BUY/SELL/HOLD for position before earnings
    post_earnings_direction: str  # UP/DOWN/NEUTRAL predicted after earnings
    confidence: str  # HIGH/MEDIUM/LOW
    entry_price: float
    pre_earnings_target: Optional[float]  # Target before earnings
    post_earnings_target: Optional[float]  # Target after earnings
    stop_loss: Optional[float]
    eps_estimate: Optional[float]
    eps_actual: Optional[float]  # Filled after earnings
    revenue_estimate: Optional[float]
    revenue_actual: Optional[float]  # Filled after earnings
    technical_indicators: Dict[str, float]
    llm_analysis: str
    sentiment_data: Dict[str, Any]
    earnings_context: Dict[str, Any]  # Historical earnings performance, volatility patterns


@dataclass
class EarningsPerformanceRecord:
    """Record of actual performance vs earnings prediction"""
    prediction_id: int
    ticker: str
    earnings_date: str
    outcome_date: str
    pre_earnings_exit_price: Optional[float]  # Price just before earnings
    post_earnings_price: float  # Price after earnings
    pre_earnings_return: Optional[float]  # Return from entry to pre-earnings
    post_earnings_return: float  # Return from pre-earnings to post-earnings
    total_return: float  # Total return from entry to outcome
    days_held_pre: Optional[int]  # Days held before earnings
    days_held_post: int  # Days held after earnings
    outcome: str  # SUCCESS/PARTIAL/FAILURE
    earnings_beat_miss: str  # BEAT/MISS/INLINE
    post_earnings_volatility: float
    market_condition: Dict[str, float]


@dataclass
class EarningsLLMFeedback:
    """Feedback from LLM reflection on earnings prediction performance"""
    feedback_date: str
    ticker: str
    period_start: str
    period_end: str
    total_predictions: int
    pre_earnings_success_rate: float
    post_earnings_success_rate: float
    avg_pre_earnings_return: float
    avg_post_earnings_return: float
    success_rate_by_beat_miss: Dict[str, float]  # BEAT/MISS/INLINE success rates
    avg_return_by_beat_miss: Dict[str, float]
    reflection: str
    improvements: str
    earnings_pattern_analysis: str