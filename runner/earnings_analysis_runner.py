"""
Earnings Analysis Runner

This module orchestrates the complete earnings analysis workflow,
including prediction generation for pre-earnings and post-earnings movement,
performance tracking, and reporting.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from llm.llmAnalysis import (
    Config,
    AdvancedTechnicalAnalysis
)
from llm.llmEarningsAnalysis import EarningsAnalyzer
from llm.llmEarningsTracking import (
    EarningsPredictionTracker,
    EarningsPerformanceEvaluator,
    EarningsLLMReflectionEngine
)
from llm.llm_models import EarningsPredictionRecord
from data.fmp_data_loader import get_earnings_for_active_stocks

logger = logging.getLogger(__name__)


class EarningsAnalysisRunner:
    """Main runner for earnings analysis workflow - orchestration only"""

    def __init__(self, earnings_analyzer: EarningsAnalyzer, config: Config, email_sender):
        """
        Initialize the earnings analysis runner

        Args:
            earnings_analyzer: EarningsAnalyzer instance for earnings analysis logic
            config: Configuration settings
            email_sender: EmailSender instance for report distribution
        """
        self.analyzer = earnings_analyzer
        self.config = config
        self.email_sender = email_sender

        # Access tracker from analyzer (analyzer now owns the tracker)
        self.earnings_tracker = self.analyzer.earnings_tracker
        self.evaluator = EarningsPerformanceEvaluator(self.earnings_tracker)
        self.reflection_engine = EarningsLLMReflectionEngine(
            self.earnings_tracker,
            self.analyzer.llm
        )

    def check_open_earnings_predictions(self) -> List[Dict[str, Any]]:
        """Check and update open earnings predictions"""
        logger.info("Checking open earnings predictions...")

        try:
            # Import data loader function
            from data.data_loader import fetch_stock_data

            updated_predictions = self.evaluator.check_and_update_predictions(fetch_stock_data)

            if updated_predictions:
                logger.info(f"Updated {len(updated_predictions)} earnings predictions")
                for update in updated_predictions:
                    logger.info(f"  {update['ticker']}: {update['outcome']}, Return: {update['total_return']:.2f}%")
            else:
                logger.info("No earnings predictions needed updating")

            return updated_predictions

        except Exception as e:
            logger.error(f"Error checking open earnings predictions: {e}")
            return []

    def run_weekly_earnings_performance_check(self) -> Dict[str, Any]:
        """Run weekly performance analysis for earnings"""
        logger.info("Running weekly earnings performance check...")

        try:
            performance_summary = self.earnings_tracker.get_performance_statistics(days=7)

            if performance_summary and performance_summary.get('total_predictions', 0) > 0:
                logger.info(f"Weekly earnings performance: "
                          f"{performance_summary.get('success_rate', 0)*100:.1f}% success rate, "
                          f"{performance_summary.get('avg_total_return', 0):.2f}% avg return")
            else:
                logger.info("No recent earnings predictions to analyze")

            return performance_summary

        except Exception as e:
            logger.error(f"Error running weekly earnings performance check: {e}")
            return {}

    def fetch_earnings_calendar_stocks(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """Fetch top N stocks by volume with upcoming earnings from FMP API"""
        logger.info(f"Fetching top {top_n} stocks by volume with upcoming earnings...")

        try:
            # Get top stocks by volume from earnings calendar
            earnings_df = get_earnings_for_active_stocks(api_key=self.config.fmp_api_key, top_n=top_n)

            if earnings_df is None or earnings_df.empty:
                logger.warning("No earnings data available for upcoming week")
                return []

            # Convert to list of dicts
            earnings_stocks = earnings_df.to_dict('records')

            logger.info(f"Found {len(earnings_stocks)} high-volume stocks with earnings in upcoming week")
            for stock in earnings_stocks[:5]:  # Log first 5
                logger.info(f"  {stock['Symbol']}: Volume={stock['Volume']:,.0f}, Earnings={stock['date']} ({stock['days_until_earnings']} days)")

            return earnings_stocks

        except Exception as e:
            logger.error(f"Error fetching earnings calendar stocks: {e}")
            return []

    def process_single_ticker_earnings(self, ticker: str, earnings_info: Dict, plot_path: Path) -> Optional[Dict[str, Any]]:
        """Process earnings analysis for a single ticker"""
        logger.info(f"Processing earnings analysis for {ticker}")

        try:
            # Fetch stock data
            df, _, _ = self.analyzer.get_stock_data(ticker, period='1y')

            if df is None or df.empty:
                logger.warning(f"No stock data available for {ticker}")
                return None

            # Calculate comprehensive technical indicators
            df = self.analyzer.calculate_technical_indicators(df)
            df = AdvancedTechnicalAnalysis.calculate_comprehensive_indicators(df)

            # Get current price
            current_price = df['Close'].iloc[-1]

            # Enhance sentiment analysis with earnings context
            earnings_context = {
                'earnings_date': str(earnings_info.get('date', '')),
                'days_until_earnings': earnings_info.get('days_until_earnings', 0),
                'eps_estimate': earnings_info.get('epsEstimated'),
                'revenue_estimate': earnings_info.get('revenueEstimated'),
                'eps_actual': earnings_info.get('epsActual'),
                'revenue_actual': earnings_info.get('revenueActual')
            }

            # Get LLM analysis with earnings context from analyzer
            llm_result = self.analyzer.get_earnings_llm_analysis(df, ticker, earnings_context)

            if not llm_result:
                logger.warning(f"Could not generate earnings analysis for {ticker}")
                return None

            # Create earnings prediction record
            prediction = EarningsPredictionRecord(
                ticker=ticker,
                prediction_date=datetime.now().strftime('%Y-%m-%d'),
                earnings_date=earnings_context['earnings_date'],
                days_until_earnings=earnings_context['days_until_earnings'],
                pre_earnings_recommendation=llm_result.get('pre_earnings_recommendation', 'HOLD'),
                post_earnings_direction=llm_result.get('post_earnings_direction', 'NEUTRAL'),
                confidence=llm_result.get('confidence', 'MEDIUM'),
                entry_price=current_price,
                pre_earnings_target=llm_result.get('pre_earnings_target'),
                post_earnings_target=llm_result.get('post_earnings_target'),
                stop_loss=llm_result.get('stop_loss'),
                eps_estimate=earnings_info.get('epsEstimated'),
                eps_actual=earnings_info.get('epsActual'),  # Populated if earnings already occurred
                revenue_estimate=earnings_info.get('revenueEstimated'),
                revenue_actual=earnings_info.get('revenueActual'),  # Populated if earnings already occurred
                technical_indicators=self.analyzer.extract_earnings_technical_indicators(df),
                llm_analysis=llm_result.get('analysis', ''),
                sentiment_data=llm_result.get('sentiment_data', {}),
                earnings_context=earnings_context
            )

            # Record prediction
            prediction_id = self.earnings_tracker.record_prediction(prediction)

            # Create plot with earnings date marked
            plot_file = self._create_earnings_plot(ticker, df, earnings_info, plot_path)

            result = {
                'ticker': ticker,
                'current_price': current_price,
                'earnings_date': earnings_context['earnings_date'],
                'days_until_earnings': earnings_context['days_until_earnings'],
                'prediction_id': prediction_id,
                'prediction': prediction,
                'plot_file': plot_file,
                'summary': self._create_ticker_summary(ticker, prediction, earnings_info)
            }

            logger.info(f"Generated earnings analysis for {ticker}")
            return result

        except Exception as e:
            logger.error(f"Error processing earnings for {ticker}: {e}")
            return None


    def _create_ticker_summary(self, ticker: str, prediction: EarningsPredictionRecord,
                              earnings_info: Dict) -> str:
        """Create a summary of earnings prediction for a ticker"""
        summary = f"{ticker} Earnings Analysis:\n"
        summary += f"- Earnings Date: {prediction.earnings_date} ({prediction.days_until_earnings} days)\n"
        summary += f"- Pre-Earnings: {prediction.pre_earnings_recommendation}\n"
        summary += f"- Post-Earnings: {prediction.post_earnings_direction}\n"
        summary += f"- Confidence: {prediction.confidence}\n"

        if prediction.eps_estimate:
            summary += f"- EPS Estimate: ${prediction.eps_estimate:.2f}\n"

        if prediction.pre_earnings_target:
            summary += f"- Pre-Earnings Target: ${prediction.pre_earnings_target:.2f}\n"

        if prediction.post_earnings_target:
            summary += f"- Post-Earnings Target: ${prediction.post_earnings_target:.2f}\n"

        return summary

    def _create_earnings_plot(self, ticker: str, df, earnings_info: Dict, plot_path: Path) -> str:
        """Create visualization for earnings analysis with earnings date marked"""
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

            # Plot 1: Price chart with MAs and earnings date
            ax1.plot(df.index, df['Close'], label='Close Price', color='black', linewidth=2)
            ax1.plot(df.index, df['EMA_20'], label='20-day MA', color='blue', alpha=0.7)
            ax1.plot(df.index, df['EMA_50'], label='50-day MA', color='red', alpha=0.7)

            # Mark earnings date
            earnings_date = earnings_info.get('date')
            if earnings_date:
                ax1.axvline(x=earnings_date, color='green', linestyle='--', linewidth=2,
                          label=f'Earnings: {earnings_date}', alpha=0.7)

            ax1.set_title(f'{ticker} - Earnings Analysis ({df.index[0].date()} to {df.index[-1].date()})')
            ax1.set_ylabel('Price ($)')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)

            # Plot 2: Volume
            colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red'
                     for i in range(len(df))]
            ax2.bar(df.index, df['Volume'], color=colors, alpha=0.5)
            ax2.set_ylabel('Volume')
            ax2.grid(True, alpha=0.3)

            # Plot 3: RSI
            ax3.plot(df.index, df['RSI'], label='RSI', color='purple', linewidth=2)
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought (70)')
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold (30)')
            ax3.set_ylabel('RSI')
            ax3.set_xlabel('Date')
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            plot_file = plot_path / f"{ticker}_earnings_{datetime.now().strftime('%Y%m%d')}.png"
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            plt.close(fig)

            return str(plot_file)

        except Exception as e:
            logger.error(f"Error creating earnings plot for {ticker}: {e}")
            return None

    def cleanup_plot_files(self, plot_files: List[str]):
        """Clean up generated plot files"""
        for plot_file in plot_files:
            try:
                if plot_file and os.path.exists(plot_file):
                    os.remove(plot_file)
                    logger.debug(f"Removed plot file: {plot_file}")
            except Exception as e:
                logger.warning(f"Could not remove plot file {plot_file}: {e}")


def run_earnings_analysis(provider_type='auto'):
    """Main function to run complete earnings analysis"""
    try:
        logger.info("Starting comprehensive earnings analysis...")

        # Import necessary components
        from llm.llmAnalysis import Config
        from utils.utils_report import EmailSender
        from utils.utils_path import get_plot_path

        # Initialize configuration
        config = Config()

        # Verify FMP API key
        if not config.fmp_api_key or config.fmp_api_key == 'your_api_key_here':
            logger.error("FMP_API_KEY not set in .env file. Please add your FMP API key.")
            return

        if not config.sender_email or not config.sender_password or not config.recipient_emails:
            logger.error("Email credentials are not set in .env file")
            return

        # Create LLM provider
        from main import create_llm_provider, create_secondary_llm_provider
        llm_provider = create_llm_provider(config)
        secondary_llm_provider = create_secondary_llm_provider(config)

        # Create earnings analyzer (handles earnings-specific analysis logic)
        analyzer = EarningsAnalyzer(llm_provider, config, secondary_llm_provider)

        # Initialize email sender
        email_sender = EmailSender(
            smtp_server=config.smtp_server,
            smtp_port=config.smtp_port,
            sender_email=config.sender_email,
            sender_password=config.sender_password
        )

        # Create runner
        runner = EarningsAnalysisRunner(analyzer, config, email_sender)

        # Execute workflow
        runner.check_open_earnings_predictions()
        performance_summary = runner.run_weekly_earnings_performance_check()

        # Fetch top 20 stocks by volume from earnings calendar
        earnings_stocks = runner.fetch_earnings_calendar_stocks(top_n=20)

        if not earnings_stocks:
            logger.warning("No stocks with upcoming earnings found")
            return

        # Setup plot directory
        plt_path = get_plot_path()
        os.makedirs(plt_path, exist_ok=True)

        # Process each ticker with earnings (already filtered to top 20 by volume)
        analysis_results = []
        plot_files = []

        for earnings_info in earnings_stocks:
            ticker = earnings_info['Symbol']
            result = runner.process_single_ticker_earnings(ticker, earnings_info, plt_path)
            if result:
                analysis_results.append(result)
                if result['plot_file']:
                    plot_files.append(result['plot_file'])

        # Send report
        if analysis_results:
            email_sender.send_earnings_report(
                recipient_emails=config.recipient_emails,
                subject=f"Earnings Analysis Report - {len(analysis_results)} Stocks with Upcoming Earnings",
                analysis_results=analysis_results,
                plot_files=plot_files,
                performance_summary=performance_summary
            )
        else:
            logger.warning("No earnings analysis results generated")

        # Cleanup
        runner.cleanup_plot_files(plot_files)
        plt.close('all')

        logger.info("Earnings analysis completed successfully")

    except Exception as e:
        logger.critical(f"Critical error in earnings analysis: {e}")
        import traceback
        traceback.print_exc()


def run_earnings_reflection():
    """Generate reflection on earnings prediction performance"""
    try:
        logger.info("Starting earnings reflection analysis...")

        # Import necessary components
        from llm.llmAnalysis import Config
        from utils.utils_report import EmailSender

        # Initialize configuration
        config = Config()

        if not config.sender_email or not config.sender_password or not config.recipient_emails:
            logger.error("Email credentials are not set in .env file")
            return

        # Create LLM provider
        from main import create_llm_provider
        llm_provider = create_llm_provider(config)

        # Initialize tracker and reflection engine
        db_path = Path(config.project_root) / "earnings_predictions.db"
        tracker = EarningsPredictionTracker(db_path)
        reflection_engine = EarningsLLMReflectionEngine(tracker, llm_provider)

        # Generate reflection
        reflection = reflection_engine.generate_reflection(days=90)

        # Get performance statistics
        stats = tracker.get_performance_statistics(days=90)
        beat_miss_stats = tracker.get_performance_by_beat_miss(days=90)

        # Create email sender and send reflection report
        email_sender = EmailSender(
            smtp_server=config.smtp_server,
            smtp_port=config.smtp_port,
            sender_email=config.sender_email,
            sender_password=config.sender_password
        )

        email_sender.send_earnings_reflection_report(
            recipient_emails=config.recipient_emails,
            subject="Earnings Prediction Performance Reflection",
            reflection_text=reflection,
            stats=stats,
            beat_miss_stats=beat_miss_stats,
            days=90
        )

        logger.info("Earnings reflection report sent successfully")

    except Exception as e:
        logger.critical(f"Critical error in earnings reflection: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run earnings analysis
    run_earnings_analysis()
