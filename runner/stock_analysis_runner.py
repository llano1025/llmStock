"""
Business logic for running stock analysis operations.
Extracted from main.py to improve code organization.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class StockAnalysisRunner:
    """Handles the main stock analysis workflow"""
    
    def __init__(self, analyzer, config, email_sender):
        self.analyzer = analyzer
        self.config = config
        self.email_sender = email_sender
        
    def get_default_ticker_list(self) -> List[str]:
        """Get default list of tickers to analyze"""
        return ['META', 'TSLA', 'NVDA', 'AAPL', 'AMZN', 'MSFT', 'GOOGL', 'AMD', 'BABA', 'OKLO']
    
    def fetch_active_stocks(self, fetch_most_active_stocks_func) -> List[str]:
        """Fetch most active stocks and combine with default list"""
        ticker_list = self.get_default_ticker_list()
        
        try:
            active_stock = fetch_most_active_stocks_func()
            if active_stock is not None:
                ticker_list.extend(active_stock['Symbol'].tolist())
                ticker_list = list(set(ticker_list))
        except Exception as e:
            logger.error(f"Error fetching most active stocks: {e}")
            
        return ticker_list
    
    def process_single_ticker(self, ticker: str, plt_path: Path) -> Dict[str, Any]:
        """Process analysis for a single ticker"""
        logger.info(f"Processing {ticker}...")
        
        try:
            # Get and analyze stock data
            df, adv_df, earning_delta = self.analyzer.get_stock_data(ticker)
            
            if df is None or df.empty:
                logger.warning(f"No data available for {ticker}, skipping")
                return None
                
            df = self.analyzer.calculate_technical_indicators(df)
            
            # Create and save plot
            date = datetime.now().strftime("%Y-%m-%d")
            plot_file = f'{ticker}_technical_analysis_{date}.png'
            fig = self.analyzer.plot_technical_analysis(
                ticker, df, adv_df, self.config.predict_window, 
                [], [], earning_delta  # Empty predictions arrays
            )
            plot_path_full = plt_path / plot_file
            fig.savefig(str(plot_path_full))
            
            # Close figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)
            
            # Get LLM analysis
            analysis, action, confidence = self.analyzer.get_llm_analysis(df, ticker)
            logger.info(f"Analysis for {ticker}: {action} (Confidence: {confidence})")
            
            # Prepare results
            latest = df.iloc[-1]
            result = {
                'symbol': ticker,
                'price': latest['Close'],
                'change_pct': (latest['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100,
                'volume': latest['Volume'],
                'rsi': latest['RSI'],
                'macd_signal': 'Bullish' if latest['MACD'] > latest['MACD_Signal'] else 'Bearish',
                'recommendation': action,
                'confidence': confidence,
                'summary': analysis,
                'plot_file': str(plot_path_full)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            return None
    
    def generate_performance_summary(self, ticker_list: List[str]) -> str:
        """Generate performance summary for top tickers"""
        performance_summary = ""
        
        for ticker in ticker_list[:5]:  # Top 5 tickers
            try:
                if hasattr(self.analyzer, 'tracker'):
                    stats = self.analyzer.tracker.get_performance_stats(ticker, days=7)
                    if stats['total_predictions'] > 0:
                        performance_summary += f"\\n{ticker}: {stats['success_rate']:.0%} success rate ({stats['total_predictions']} predictions)"
            except Exception as e:
                logger.warning(f"Could not get performance stats for {ticker}: {e}")
                
        return performance_summary
    
    def send_email_report(self, analysis_results: List[Dict], plot_files: List[str], 
                         performance_summary: str = ""):
        """Send email report with analysis results"""
        if not analysis_results:
            logger.warning("No analysis results to send")
            return
            
        # Prepare email subject
        subject = f"Stock Analysis Report - {datetime.now().strftime('%Y-%m-%d')}"
        if performance_summary:
            subject += " | Recent Performance"
            
        # Add performance summary to first result if available
        if performance_summary and analysis_results:
            analysis_results[0]['performance_summary'] = performance_summary
            
        try:
            self.email_sender.send_report(
                recipient_emails=self.config.recipient_emails,
                subject=subject,
                analysis_results=analysis_results,
                plot_files=plot_files
            )
            logger.info("Email report sent successfully")
        except Exception as e:
            logger.error(f"Failed to send email report: {e}")
    
    def cleanup_plot_files(self, plot_files: List[str]):
        """Clean up temporary plot files"""
        for plot_file in plot_files:
            try:
                os.remove(plot_file)
            except Exception as e:
                logger.warning(f"Failed to remove plot file {plot_file}: {e}")
    
    def run_weekly_performance_check(self):
        """Run weekly performance check if it's Monday"""
        if datetime.now().weekday() == 0 and hasattr(self.analyzer, 'generate_performance_report'):
            try:
                performance_report = self.analyzer.generate_performance_report(days=30)
                logger.info("Generated performance report")
                
                # Save report
                from utils.utils_path import get_save_path
                report_path = get_save_path() / f"performance_report_{datetime.now().strftime('%Y-%m-%d')}.md"
                os.makedirs(report_path.parent, exist_ok=True)
                with open(report_path, 'w') as f:
                    f.write(performance_report)
            except Exception as e:
                logger.warning(f"Could not generate/save performance report: {e}")
    
    def check_open_predictions(self):
        """Check and update any open predictions"""
        if hasattr(self.analyzer, 'check_and_update_predictions'):
            try:
                logger.info("Checking open predictions...")
                self.analyzer.check_and_update_predictions()
            except Exception as e:
                logger.warning(f"Could not check open predictions: {e}")