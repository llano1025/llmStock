#!/usr/bin/env python3
"""
Main entry point for the Stock Analysis System with LLM integration.

This script provides command-line interface for:
- Stock analysis with technical indicators and LLM insights
- Performance tracking and reflection
- Testing prediction tracking system
"""

import os
import sys
import argparse
import logging
import sqlite3
from pathlib import Path
from datetime import datetime

def show_providers_help():
    """Show available providers without importing modules"""
    print("Available LLM Providers:")
    print("  auto      - Auto-detect based on configuration (default)")
    print("  ollama    - Local Ollama server (requires Ollama running)")
    print("  lmstudio  - Local LM Studio server (requires LM Studio running)")
    print("  gemini    - Google Gemini API (requires GEMINI_API_KEY)")
    print("  deepseek  - DeepSeek API (requires DEEPSEEK_API_KEY)")
    print("\nConfiguration:")
    print("  Set API keys in .env file for cloud providers")
    print("  Ensure local servers are running for local providers")
    print("\nUsage Examples:")
    print("  python main.py --provider ollama")
    print("  python main.py --provider gemini --mode analyze")
    print("  python main.py --provider auto  # Auto-detect (default)")

# Handle special options before importing modules
if __name__ == "__main__":
    # Quick check for list-providers without full imports
    if '--list-providers' in sys.argv:
        show_providers_help()
        sys.exit(0)
    
    # Also handle help to avoid import errors
    if '--help' in sys.argv or '-h' in sys.argv:
        parser = argparse.ArgumentParser(description='Stock Analysis with Prediction Tracking')
        parser.add_argument('--mode', choices=['analyze', 'reflect', 'test'], 
                           default='analyze', help='Operation mode')
        parser.add_argument('--provider', choices=['auto', 'ollama', 'lmstudio', 'gemini', 'deepseek'],
                           default='auto', help='LLM provider to use (default: auto-detect)')
        parser.add_argument('--list-providers', action='store_true',
                           help='List available LLM providers and exit')
        parser.print_help()
        sys.exit(0)

# Add project directories to path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "llm"))

# Import matplotlib before other imports to set backend
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Core application imports - these must exist
try:
    from llm.llmAnalysis import (
        Config, LLMProvider, OllamaProvider, LMStudioProvider, 
        GeminiProvider, DeepSeekProvider, EnhancedStockAnalyzer
    )
    from data.data_loader import fetch_most_active_stocks
    from utils.utils_report import EmailSender
    from utils.utils_path import get_plot_path as utils_get_plot_path
except ImportError as e:
    print(f"CRITICAL ERROR: Missing required application modules: {e}")
    print("\nThis indicates a problem with the application structure.")
    print("Please check that all files are present:")
    print("- llm/llmAnalysis.py, llm/llmTracking.py, llm/llm_models.py")
    print("- data/data_loader.py")
    print("- utils/utils_report.py, utils/utils_path.py")
    print("\nIf external dependencies are missing, install them with:")
    print("pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_llm_provider(provider_type: str, config: Config):
    """
    Create and return the appropriate LLM provider based on type
    
    Args:
        provider_type: Type of provider ('ollama', 'lmstudio', 'gemini', 'deepseek', 'auto')
        config: Configuration object
        
    Returns:
        LLMProvider instance
    """
    if provider_type == 'auto':
        # Auto-detect based on available configuration
        if config.gemini_api_key:
            provider_type = 'gemini'
            logger.info("Auto-detected: Using Gemini provider")
        elif config.deepseek_api_key:
            provider_type = 'deepseek'
            logger.info("Auto-detected: Using DeepSeek provider")
        else:
            # Default to Ollama for local deployment
            provider_type = 'ollama'
            logger.info("Auto-detected: Using Ollama provider (default)")
    
    try:
        if provider_type == 'ollama':
            return OllamaProvider(config=config)
        elif provider_type == 'lmstudio':
            return LMStudioProvider(config=config)
        elif provider_type == 'gemini':
            if not config.gemini_api_key:
                raise ValueError("Gemini API key not configured. Set GEMINI_API_KEY in .env file.")
            return GeminiProvider(config=config)
        elif provider_type == 'deepseek':
            if not config.deepseek_api_key:
                raise ValueError("DeepSeek API key not configured. Set DEEPSEEK_API_KEY in .env file.")
            return DeepSeekProvider(config=config)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
            
    except Exception as e:
        logger.error(f"Failed to initialize {provider_type} provider: {e}")
        logger.info("Falling back to Ollama provider...")
        return OllamaProvider(config=config)


def main(provider_type='auto'):
    """Enhanced main function with prediction tracking"""
    try:
        # Check if .env exists
        env_file = Path('.env')
        if not env_file.exists():
            try:
                template_path = Config.create_env_template()
                logger.warning(f".env file not found. Created template at {template_path}")
                return
            except Exception as e:
                logger.error(f"Could not create .env template: {e}")
                return

        # Initialize configuration
        config = Config()
        
        # Check credentials
        if not config.sender_email or not config.sender_password or not config.recipient_emails:
            logger.error("Email credentials are not set in .env file")
            return

        # Initialize the enhanced analyzer with tracking
        llm_provider = create_llm_provider(provider_type, config)
        analyzer = EnhancedStockAnalyzer(llm_provider, config)
        logger.info(f"Initialized analyzer with {type(llm_provider).__name__}")
        
        # Email configuration
        email_sender = EmailSender(
            smtp_server=config.smtp_server,
            smtp_port=config.smtp_port,
            sender_email=config.sender_email,
            sender_password=config.sender_password
        )
        
        # Import and use the analysis runner
        from llm.stock_analysis_runner import StockAnalysisRunner
        runner = StockAnalysisRunner(analyzer, config, email_sender)
        
        # Run workflow steps
        runner.check_open_predictions()
        runner.run_weekly_performance_check()
        
        # Get ticker list
        ticker_list = runner.fetch_active_stocks(fetch_most_active_stocks)
        
        # Setup plot directory
        try:
            plt_path = utils_get_plot_path()
            os.makedirs(plt_path, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create plot directory: {e}")
            plt_path = Path.cwd()
        
        # Process each ticker
        analysis_results = []
        plot_files = []
        
        for ticker in ticker_list:
            result = runner.process_single_ticker(ticker, plt_path)
            if result:
                analysis_results.append(result)
                plot_files.append(result['plot_file'])
        
        # Generate performance summary and send email
        performance_summary = runner.generate_performance_summary(ticker_list)
        runner.send_email_report(analysis_results, plot_files, performance_summary)
        
        # Cleanup
        runner.cleanup_plot_files(plot_files)
        
        # Close any remaining matplotlib figures
        import matplotlib.pyplot as plt
        plt.close('all')
        
    except Exception as e:
        logger.critical(f"Critical error in main process: {e}")


# Main execution
if __name__ == "__main__":
    # Parse command line arguments (special cases already handled above)
    parser = argparse.ArgumentParser(description='Stock Analysis with Prediction Tracking')
    parser.add_argument('--mode', choices=['analyze', 'reflect', 'test'], 
                       default='analyze', help='Operation mode')
    parser.add_argument('--provider', choices=['auto', 'ollama', 'lmstudio', 'gemini', 'deepseek'],
                       default='auto', help='LLM provider to use (default: auto-detect)')
    parser.add_argument('--list-providers', action='store_true',
                       help='List available LLM providers and exit')
    args = parser.parse_args()
    
    if args.mode == 'analyze':
        main(provider_type=args.provider)
    elif args.mode == 'reflect':
        from llm.reflection_runner import run_weekly_reflection
        run_weekly_reflection(provider_type=args.provider)
    elif args.mode == 'test':
        from llm.reflection_runner import test_prediction_tracking
        test_prediction_tracking(provider_type=args.provider)