"""
LLM-powered Stock Analysis Module

This module provides advanced stock analysis capabilities using Large Language Models
for sentiment analysis, technical analysis, and prediction tracking.
"""

# Import main classes for easy access
from .llmAnalysis import (
    Config,
    LLMProvider,
    OllamaProvider,
    LMStudioProvider,
    GeminiProvider,
    OpenAIProvider,
    StockAnalyzer,
    EnhancedStockAnalyzer,
    AdvancedTechnicalAnalysis,
    MultiTimeframeAnalysis,
    OptionsFlowAnalysis
)

from .llmTracking import (
    PredictionTracker,
    PerformanceEvaluator,
    LLMReflectionEngine
)

from .llm_models import (
    PredictionRecord,
    PerformanceRecord,
    LLMFeedback,
    MarketCondition
)

__version__ = "1.0.0"
__author__ = "Stock Analysis System"

__all__ = [
    # Core configuration
    'Config',
    
    # LLM Providers
    'LLMProvider',
    'OllamaProvider', 
    'LMStudioProvider',
    'GeminiProvider',
    'OpenAIProvider',
    
    # Analysis classes
    'StockAnalyzer',
    'EnhancedStockAnalyzer',
    'AdvancedTechnicalAnalysis',
    'MultiTimeframeAnalysis',
    'OptionsFlowAnalysis',
    
    # Tracking classes
    'PredictionTracker',
    'PerformanceEvaluator', 
    'LLMReflectionEngine',
    
    # Data models
    'PredictionRecord',
    'PerformanceRecord',
    'LLMFeedback',
    'MarketCondition'
]