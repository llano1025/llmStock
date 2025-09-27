"""
Options Analysis Runner

This module orchestrates the complete options analysis workflow,
including prediction generation, performance tracking, and reporting.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from llm.llmOptionsAnalysis import OptionsAnalyzer, OptionsTracker
from llm.llmOptionsReflection import OptionsPerformanceEvaluator, OptionsLLMReflectionEngine
from llm.llm_models import OptionsPredictionRecord

logger = logging.getLogger(__name__)


class OptionsAnalysisRunner:
    """Main runner for options analysis workflow"""

    def __init__(self, options_analyzer: OptionsAnalyzer, config, email_sender):
        self.analyzer = options_analyzer
        self.config = config
        self.email_sender = email_sender

        # Initialize performance components
        self.evaluator = OptionsPerformanceEvaluator(self.analyzer.options_tracker)
        self.reflection_engine = OptionsLLMReflectionEngine(
            self.analyzer.llm, self.analyzer.options_tracker
        )

    def check_open_options_predictions(self) -> List[Dict[str, Any]]:
        """Check and update open options predictions"""
        logger.info("Checking open options predictions...")

        try:
            updated_predictions = self.evaluator.check_and_update_options_predictions()

            if updated_predictions:
                logger.info(f"Updated {len(updated_predictions)} options predictions")
                for update in updated_predictions:
                    logger.info(f"  {update['ticker']}: {update['performance'].outcome}")
            else:
                logger.info("No options predictions needed updating")

            return updated_predictions

        except Exception as e:
            logger.error(f"Error checking open options predictions: {e}")
            return []

    def run_weekly_options_performance_check(self) -> Dict[str, Any]:
        """Run weekly performance analysis for options"""
        logger.info("Running weekly options performance check...")

        try:
            # Generate performance summary
            performance_summary = self.evaluator.get_options_performance_summary(days=7)

            if 'total_predictions' in performance_summary and performance_summary['total_predictions'] > 0:
                logger.info(f"Weekly options performance: {performance_summary['success_rate']:.1%} success rate, "
                           f"{performance_summary['average_return']:.2%} avg return")
            else:
                logger.info("No recent options predictions to analyze")

            return performance_summary

        except Exception as e:
            logger.error(f"Error running weekly options performance check: {e}")
            return {}

    def fetch_active_stocks(self, fetch_function) -> List[str]:
        """Fetch active stocks for analysis (reuse existing logic)"""
        logger.info("Fetching most active stocks for options analysis...")

        try:
            # Get DataFrame from fetch function
            df = fetch_function()

            if df is None or df.empty:
                logger.warning("No stock data returned from fetch function")
                return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Fallback

            logger.debug(f"Received DataFrame with columns: {list(df.columns)}")
            logger.debug(f"DataFrame shape: {df.shape}")

            # Extract actual ticker symbols from 'Symbol' column
            if 'Symbol' not in df.columns:
                logger.error("'Symbol' column not found in DataFrame")
                return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Fallback

            # Get clean ticker symbols
            raw_symbols = df['Symbol'].astype(str).tolist()

            # Validate and clean symbols
            valid_symbols = []
            for symbol in raw_symbols:
                # Remove any extra formatting and validate
                clean_symbol = symbol.strip().upper()
                # Check if it looks like a valid ticker (1-5 letters, optional numbers)
                if clean_symbol.replace('.', '').replace('-', '').isalnum() and len(clean_symbol) <= 6:
                    valid_symbols.append(clean_symbol)
                else:
                    logger.debug(f"Skipping invalid symbol: {symbol}")

            # Limit to top 10 for options analysis (options data is more resource intensive)
            options_tickers = valid_symbols[:10]

            if not options_tickers:
                logger.warning("No valid ticker symbols found, using fallback")
                return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Fallback

            logger.info(f"Selected {len(options_tickers)} tickers for options analysis: {', '.join(options_tickers)}")
            return options_tickers

        except Exception as e:
            logger.error(f"Error fetching active stocks: {e}")
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Fallback list

    def process_single_ticker_options(self, ticker: str, plot_path: Path) -> Optional[Dict[str, Any]]:
        """Process options analysis for a single ticker"""
        logger.info(f"Processing options analysis for {ticker}")

        try:
            # Generate options predictions
            predictions = self.analyzer.analyze_ticker_options(ticker)

            if not predictions:
                logger.warning(f"No options predictions generated for {ticker}")
                return None

            # Create options analysis plot
            plot_file = self._create_options_plot(ticker, predictions, plot_path)

            # Get current price from predictions (they all have underlying_price)
            current_price = predictions[0].underlying_price if predictions else 0.0

            # Get LLM analysis from the first prediction or create a summary
            llm_analysis = ""
            if predictions:
                # Collect unique LLM analyses from predictions
                analyses = list(set(pred.llm_analysis for pred in predictions if pred.llm_analysis))
                if analyses:
                    llm_analysis = "\n\n".join(analyses[:3])  # Top 3 unique analyses
                else:
                    llm_analysis = f"Generated {len(predictions)} options recommendations based on technical analysis and market conditions."

            # Prepare results summary
            result = {
                'ticker': ticker,
                'current_price': current_price,
                'predictions_count': len(predictions),
                'predictions': predictions,
                'plot_file': plot_file,
                'best_call_prediction': self._get_best_prediction(predictions, 'CALL'),
                'best_put_prediction': self._get_best_prediction(predictions, 'PUT'),
                'summary': self._create_ticker_summary(ticker, predictions),
                'llm_analysis': llm_analysis
            }

            logger.info(f"Generated {len(predictions)} options predictions for {ticker}")
            return result

        except Exception as e:
            logger.error(f"Error processing options for {ticker}: {e}")
            return None

    def _get_best_prediction(self, predictions: List[OptionsPredictionRecord], option_type: str) -> Optional[OptionsPredictionRecord]:
        """Get the best prediction for a given option type"""
        type_predictions = [p for p in predictions if p.option_type == option_type]

        if not type_predictions:
            return None

        # Sort by confidence and expected return potential
        def prediction_score(pred):
            confidence_score = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(pred.confidence, 1)
            target_return = 0
            if pred.target_premium and pred.entry_premium:
                target_return = (pred.target_premium - pred.entry_premium) / pred.entry_premium
            return confidence_score + target_return

        return max(type_predictions, key=prediction_score)

    def _create_ticker_summary(self, ticker: str, predictions: List[OptionsPredictionRecord]) -> str:
        """Create a summary of options predictions for a ticker"""
        if not predictions:
            return f"No options predictions generated for {ticker}"

        calls = [p for p in predictions if p.option_type == 'CALL']
        puts = [p for p in predictions if p.option_type == 'PUT']

        summary = f"{ticker} Options Analysis:\n"
        summary += f"- {len(calls)} CALL predictions, {len(puts)} PUT predictions\n"

        if calls:
            high_conf_calls = [p for p in calls if p.confidence == 'HIGH']
            summary += f"- {len(high_conf_calls)} high-confidence CALL opportunities\n"

        if puts:
            high_conf_puts = [p for p in puts if p.confidence == 'HIGH']
            summary += f"- {len(high_conf_puts)} high-confidence PUT opportunities\n"

        # Add expiration distribution
        expirations = {}
        for pred in predictions:
            days = pred.days_to_expiration
            expirations[days] = expirations.get(days, 0) + 1

        summary += f"- Expiration distribution: {dict(sorted(expirations.items()))}\n"

        return summary

    def _create_options_plot(self, ticker: str, predictions: List[OptionsPredictionRecord], plot_path: Path) -> str:
        """Create visualization for options predictions"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Plot 1: Strike prices vs current price
            if predictions:
                current_price = predictions[0].underlying_price

                calls = [p for p in predictions if p.option_type == 'CALL']
                puts = [p for p in predictions if p.option_type == 'PUT']

                if calls:
                    call_strikes = [p.strike_price for p in calls]
                    call_premiums = [p.entry_premium for p in calls]
                    call_colors = ['red' if p.confidence == 'HIGH' else 'orange' if p.confidence == 'MEDIUM' else 'yellow' for p in calls]
                    ax1.scatter(call_strikes, call_premiums, c=call_colors, marker='^', s=100, label='CALLs', alpha=0.7)

                if puts:
                    put_strikes = [p.strike_price for p in puts]
                    put_premiums = [p.entry_premium for p in puts]
                    put_colors = ['red' if p.confidence == 'HIGH' else 'orange' if p.confidence == 'MEDIUM' else 'yellow' for p in puts]
                    ax1.scatter(put_strikes, put_premiums, c=put_colors, marker='v', s=100, label='PUTs', alpha=0.7)

                # Add current price line
                ax1.axvline(x=current_price, color='black', linestyle='--', linewidth=2, label=f'Current Price: ${current_price:.2f}')

                ax1.set_xlabel('Strike Price ($)')
                ax1.set_ylabel('Option Premium ($)')
                ax1.set_title(f'{ticker} Options Predictions - Strike vs Premium')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Plot 2: Days to expiration vs expected returns
                all_days = [p.days_to_expiration for p in predictions]
                all_returns = []

                for p in predictions:
                    if p.target_premium and p.entry_premium:
                        expected_return = (p.target_premium - p.entry_premium) / p.entry_premium
                    else:
                        expected_return = 0.0
                    all_returns.append(expected_return)

                colors = ['red' if p.confidence == 'HIGH' else 'orange' if p.confidence == 'MEDIUM' else 'yellow' for p in predictions]
                markers = ['^' if p.option_type == 'CALL' else 'v' for p in predictions]

                for i, (days, ret, color, marker) in enumerate(zip(all_days, all_returns, colors, markers)):
                    ax2.scatter(days, ret, c=color, marker=marker, s=100, alpha=0.7)

                ax2.set_xlabel('Days to Expiration')
                ax2.set_ylabel('Expected Return')
                ax2.set_title(f'{ticker} Options Predictions - Time vs Expected Return')
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            plt.tight_layout()

            plot_filename = f"{ticker}_options_analysis.png"
            plot_file_path = plot_path / plot_filename
            plt.savefig(plot_file_path, dpi=150, bbox_inches='tight')
            plt.close()

            return str(plot_file_path)

        except Exception as e:
            logger.error(f"Error creating options plot for {ticker}: {e}")
            plt.close()
            return ""

    def generate_options_performance_summary(self, ticker_list: List[str]) -> str:
        """Generate comprehensive performance summary for options"""
        logger.info("Generating options performance summary...")

        try:
            # Get overall performance
            overall_performance = self.evaluator.get_options_performance_summary(days=30)

            # Generate reflection
            reflection = self.reflection_engine.generate_options_reflection('ALL', days=30)

            summary = "OPTIONS TRADING PERFORMANCE SUMMARY\n"
            summary += "=" * 50 + "\n\n"

            if 'total_predictions' in overall_performance and overall_performance['total_predictions'] > 0:
                summary += f"Total Predictions (30 days): {overall_performance['total_predictions']}\n"
                summary += f"Success Rate: {overall_performance['success_rate']:.1%}\n"
                summary += f"Average Return: {overall_performance['average_return']:.2%}\n\n"

                # Add performance by type
                if 'best_performing_type' in overall_performance:
                    summary += f"Best Performing Option Type: {overall_performance['best_performing_type']}\n"

                if 'best_performing_expiration' in overall_performance:
                    summary += f"Optimal Expiration: {overall_performance['best_performing_expiration']} days\n\n"

                # Add key insights from reflection
                summary += "KEY INSIGHTS:\n"
                summary += reflection.improvements + "\n\n"

            else:
                summary += "No recent options predictions for performance analysis.\n"
                summary += "Building prediction history...\n\n"

            summary += f"Analyzed Tickers: {', '.join(ticker_list)}\n"
            summary += f"Analysis Date: {reflection.feedback_date}\n"

            return summary

        except Exception as e:
            logger.error(f"Error generating options performance summary: {e}")
            return "Error generating performance summary."


    def send_options_email_report(self, analysis_results: List[Dict[str, Any]],
                                plot_files: List[str], performance_summary: str):
        """Send dedicated options analysis email report"""
        logger.info("Sending options analysis email report...")

        try:
            # Create email subject
            subject = f"Options Trading Analysis Report - {len(analysis_results)} Tickers Analyzed"

            # Filter valid plot files
            valid_plots = [f for f in plot_files if f and os.path.exists(f)]

            # Use dedicated options email method
            self.email_sender.send_options_report(
                recipient_emails=self.config.recipient_emails,
                subject=subject,
                options_results=analysis_results,
                plot_files=valid_plots
            )

            logger.info("Options analysis email sent successfully")

        except Exception as e:
            logger.error(f"Error sending options email report: {e}")

    def send_empty_results_email_report(self, ticker_list: List[str], performance_summary: str):
        """Send email report when no options analysis results are found"""
        logger.info("Sending empty results email report...")

        try:
            # Create diagnostic information
            diagnostic_info = self._create_diagnostic_analysis(ticker_list)

            # Create empty results structure with diagnostic content
            empty_results = [{
                'ticker': 'MARKET_ANALYSIS',
                'current_price': 0.0,
                'predictions_count': 0,
                'predictions': [],
                'plot_file': None,
                'best_call_prediction': None,
                'best_put_prediction': None,
                'summary': diagnostic_info,
                'llm_analysis': 'No options analysis could be performed due to market conditions or data availability. See diagnostic information for details.'
            }]

            subject = f"Options Analysis Report - Market Conditions Prevented Analysis ({len(ticker_list)} tickers attempted)"

            # Use the dedicated options email method
            self.email_sender.send_options_report(
                recipient_emails=self.config.recipient_emails,
                subject=subject,
                options_results=empty_results,
                plot_files=[]
            )

            logger.info("Empty results email sent successfully")

        except Exception as e:
            logger.error(f"Error sending empty results email: {e}")

    def send_partial_results_email_report(self, analysis_results: List[Dict[str, Any]],
                                        plot_files: List[str], performance_summary: str, ticker_list: List[str]):
        """Send email report for partial results with diagnostic information"""
        logger.info("Sending partial results email report...")

        try:
            # Find failed tickers
            successful_tickers = {result['ticker'] for result in analysis_results}
            failed_tickers = [ticker for ticker in ticker_list if ticker not in successful_tickers]

            # Create diagnostic entry for failed tickers
            if failed_tickers:
                diagnostic_info = self._create_diagnostic_analysis(failed_tickers)

                diagnostic_result = {
                    'ticker': 'DIAGNOSTIC_INFO',
                    'current_price': 0.0,
                    'predictions_count': 0,
                    'predictions': [],
                    'plot_file': None,
                    'best_call_prediction': None,
                    'best_put_prediction': None,
                    'summary': f"Failed Analysis for: {', '.join(failed_tickers)}",
                    'llm_analysis': diagnostic_info
                }

                # Add diagnostic info to results
                analysis_results.append(diagnostic_result)

            subject = f"Options Analysis Report - Partial Results ({len(analysis_results)-1}/{len(ticker_list)} tickers successful)"

            # Filter valid plot files
            valid_plots = [f for f in plot_files if f and os.path.exists(f)]

            # Use the dedicated options email method
            self.email_sender.send_options_report(
                recipient_emails=self.config.recipient_emails,
                subject=subject,
                options_results=analysis_results,
                plot_files=valid_plots
            )

            logger.info("Partial results email sent successfully")

        except Exception as e:
            logger.error(f"Error sending partial results email: {e}")

    def _create_diagnostic_analysis(self, failed_tickers: List[str]) -> str:
        """Create diagnostic information for failed options analysis"""
        try:
            diagnostic = f"OPTIONS ANALYSIS DIAGNOSTIC REPORT\n"
            diagnostic += "=" * 50 + "\n\n"

            diagnostic += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            diagnostic += f"Failed Tickers: {', '.join(failed_tickers)}\n\n"

            diagnostic += "POSSIBLE CAUSES:\n"
            diagnostic += "1. Low Options Liquidity: Stocks may have limited options trading activity\n"
            diagnostic += "2. Market Hours: Analysis performed outside trading hours may have stale data\n"
            diagnostic += "3. Data Provider Issues: Temporary issues with options data feed\n"
            diagnostic += "4. Technical Filtering: Stocks failed liquidity and technical criteria\n\n"

            diagnostic += "MARKET CONDITIONS:\n"
            diagnostic += "- Options markets may be experiencing low volatility\n"
            diagnostic += "- Recent market events may have affected options pricing\n"
            diagnostic += "- Consider checking individual tickers manually\n\n"

            diagnostic += "RECOMMENDATIONS:\n"
            diagnostic += "1. Try analysis again during regular market hours\n"
            diagnostic += "2. Check individual tickers on your broker platform\n"
            diagnostic += "3. Consider more liquid alternatives (SPY, QQQ, AAPL, TSLA)\n"
            diagnostic += "4. Review overall market volatility conditions\n\n"

            diagnostic += "TECHNICAL NOTES:\n"
            diagnostic += "- Emergency fallback mechanisms were attempted\n"
            diagnostic += "- Multiple liquidity filtering levels were applied\n"
            diagnostic += "- LLM-assisted selection was attempted if applicable\n"

            return diagnostic

        except Exception as e:
            logger.error(f"Error creating diagnostic analysis: {e}")
            return f"Diagnostic analysis failed for tickers: {', '.join(failed_tickers)}"

    def cleanup_plot_files(self, plot_files: List[str]):
        """Clean up generated plot files"""
        for plot_file in plot_files:
            try:
                if plot_file and os.path.exists(plot_file):
                    os.remove(plot_file)
                    logger.debug(f"Removed plot file: {plot_file}")
            except Exception as e:
                logger.warning(f"Could not remove plot file {plot_file}: {e}")


def run_options_analysis(provider_type='auto'):
    """Main function to run complete options analysis"""
    try:
        logger.info("Starting comprehensive options analysis...")

        # Import necessary components
        from llm.llmAnalysis import Config
        from data.data_loader import fetch_most_active_stocks
        from utils.utils_report import EmailSender
        from utils.utils_path import get_plot_path

        # Initialize configuration
        config = Config()

        if not config.sender_email or not config.sender_password or not config.recipient_emails:
            logger.error("Email credentials are not set in .env file")
            return

        # Create LLM provider and analyzer
        from main import create_llm_provider
        llm_provider = create_llm_provider(provider_type, config)
        analyzer = OptionsAnalyzer(llm_provider, config)

        # Initialize email sender
        email_sender = EmailSender(
            smtp_server=config.smtp_server,
            smtp_port=config.smtp_port,
            sender_email=config.sender_email,
            sender_password=config.sender_password
        )

        # Create runner
        runner = OptionsAnalysisRunner(analyzer, config, email_sender)

        # Execute workflow
        runner.check_open_options_predictions()
        runner.run_weekly_options_performance_check()

        # Get ticker list
        # ticker_list = runner.fetch_active_stocks(fetch_most_active_stocks)
        ticker_list = ['NVDA', 'HOOD']

        # Setup plot directory
        plt_path = get_plot_path()
        os.makedirs(plt_path, exist_ok=True)

        # Process each ticker
        analysis_results = []
        plot_files = []

        for ticker in ticker_list:
            result = runner.process_single_ticker_options(ticker, plt_path)
            if result:
                analysis_results.append(result)
                if result['plot_file']:
                    plot_files.append(result['plot_file'])

        # Generate performance summary and send report
        performance_summary = runner.generate_options_performance_summary(ticker_list)

        # Handle empty or partial results intelligently
        if not analysis_results:
            logger.warning("No options analysis results generated for any ticker")
            runner.send_empty_results_email_report(ticker_list, performance_summary)
        elif len(analysis_results) < len(ticker_list) * 0.5:  # Less than 50% success
            logger.warning(f"Low success rate: {len(analysis_results)}/{len(ticker_list)} tickers analyzed")
            runner.send_partial_results_email_report(analysis_results, plot_files, performance_summary, ticker_list)
        else:
            runner.send_options_email_report(analysis_results, plot_files, performance_summary)

        # Cleanup
        runner.cleanup_plot_files(plot_files)
        plt.close('all')

        logger.info("Options analysis completed successfully")

    except Exception as e:
        logger.critical(f"Critical error in options analysis: {e}")
        plt.close('all')