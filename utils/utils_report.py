import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.text import MIMEText
from dotenv import load_dotenv
from markdown_it import MarkdownIt

class EmailSender:
    def __init__(self, smtp_server, smtp_port, sender_email, sender_password):
        """
        Initialize email sender with SMTP credentials
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.md = MarkdownIt('commonmark', {'breaks': True, 'html': True})

    def create_html_report(self, analysis_results, include_full_analysis=True):
        """
        Create detailed HTML report with option to include or exclude full analysis
        
        Args:
            analysis_results: List of analysis results
            include_full_analysis: Boolean to include full LLM analysis text (default: True)
        """
        html = """
        <html>
        <head>
            <style>
                /* Previous styles remain the same */
                .recommendation {
                    font-weight: bold;
                    padding: 5px 10px;
                    border-radius: 3px;
                }
                .BUY {
                    background-color: #d4edda;
                    color: #155724;
                }
                .SELL {
                    background-color: #f8d7da;
                    color: #721c24;
                }
                .HOLD {
                    background-color: #fff3cd;
                    color: #856404;
                }
                .HIGH {
                    border-left: 4px solid #28a745;
                }
                .MEDIUM {
                    border-left: 4px solid #ffc107;
                }
                .LOW {
                    border-left: 4px solid #6c757d;
                }
                /* Add styles for clickable symbols */
                .symbol-link {
                    color: #0366d6;
                    text-decoration: none;
                }
                .symbol-link:hover {
                    text-decoration: underline;
                }
                /* Add smooth scrolling for better user experience */
                html {
                    scroll-behavior: smooth;
                }
                /* Add some padding to prevent header overlap when jumping to sections */
                .stock-analysis {
                    scroll-margin-top: 20px;
                    margin-bottom: 30px;
                }
                /* Style for Jump to Top link */
                .jump-to-top {
                    display: inline-block;
                    margin-top: 15px;
                    padding: 8px 15px;
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    color: #0366d6;
                    text-decoration: none;
                    font-size: 14px;
                }
                .jump-to-top:hover {
                    background-color: #e9ecef;
                    text-decoration: none;
                }
                /* Markdown content styles */
                .analysis-text h1, .analysis-text h2, .analysis-text h3, .analysis-text h4 {
                    color: #0366d6;
                    margin-top: 20px;
                    margin-bottom: 10px;
                }
                .analysis-text h1 { font-size: 1.8em; }
                .analysis-text h2 { font-size: 1.5em; }
                .analysis-text h3 { font-size: 1.3em; }
                .analysis-text h4 { font-size: 1.1em; }
                .analysis-text p {
                    margin-bottom: 10px;
                    line-height: 1.6;
                }
                .analysis-text ul, .analysis-text ol {
                    margin-left: 20px;
                    margin-bottom: 10px;
                }
                .analysis-text li {
                    margin-bottom: 5px;
                    line-height: 1.5;
                }
                .analysis-text strong {
                    font-weight: bold;
                    color: #333;
                }
                .analysis-text em {
                    font-style: italic;
                }
                .analysis-text code {
                    background-color: #f6f8fa;
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-family: monospace;
                }
                .analysis-text blockquote {
                    border-left: 4px solid #dfe2e5;
                    padding-left: 16px;
                    margin: 16px 0;
                    color: #6a737d;
                }
                .analysis-text hr {
                    border: none;
                    border-top: 1px solid #e1e4e8;
                    margin: 20px 0;
                }
            </style>
        </head>
        <body>
            <div id="top"></div>
            <h1>Stock Analysis Report</h1>
            <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            
            <h2>Market Overview</h2>
            <div class="metrics">
                <div class="metric">
                    <strong>Total Stocks Analyzed:</strong> """ + str(len(analysis_results)) + """
                </div>
                <div class="metric">
                    <strong>Buy Recommendations:</strong> """ + str(sum(1 for x in analysis_results if x['recommendation'] == 'BUY')) + """
                </div>
                <div class="metric">
                    <strong>Sell Recommendations:</strong> """ + str(sum(1 for x in analysis_results if x['recommendation'] == 'SELL')) + """
                </div>
                <div class="metric">
                    <strong>Hold Recommendations:</strong> """ + str(sum(1 for x in analysis_results if x['recommendation'] == 'HOLD')) + """
                </div>
            </div>

            <h2>Summary Table</h2>
            <table class="summary-table">
                <tr>
                    <th>Symbol</th>
                    <th>Price</th>
                    <th>Change %</th>
                    <th>Volume</th>
                    <th>RSI</th>
                    <th>MACD Signal</th>
                    <th>Recommendation</th>
                    <th>Confidence</th>
                </tr>
        """
        
        # Add summary rows with clickable symbols
        for result in analysis_results:
            change_class = 'positive' if result['change_pct'] > 0 else 'negative'
            change_symbol = '+' if result['change_pct'] > 0 else ''
            symbol_id = f"analysis-{result['symbol']}"  # Create unique ID for the detailed section
            
            html += f"""
                <tr>
                    <td><strong><a href="#{symbol_id}" class="symbol-link">{result['symbol']}</a></strong></td>
                    <td>${result['price']:.2f}</td>
                    <td class="{change_class}">{change_symbol}{result['change_pct']:.2f}%</td>
                    <td>{result['volume']:,.0f}</td>
                    <td>{result['rsi']:.2f}</td>
                    <td>{result['macd_signal']}</td>
                    <td><span class="recommendation {result['recommendation']}">{result['recommendation']}</span></td>
                    <td><span class="recommendation {result['confidence']}">{result['confidence']}</span></td>
                </tr>
            """
        
        html += "</table>"
        
        # Detailed analysis section with added IDs for anchor links
        html += "<h2>Detailed Analysis</h2>"
        
        for result in analysis_results:
            change_class = 'positive' if result['change_pct'] > 0 else 'negative'
            change_symbol = '+' if result['change_pct'] > 0 else ''
            symbol_id = f"analysis-{result['symbol']}"  # Same ID as used in the summary table

            # Create condensed summary or full analysis based on flag
            if include_full_analysis:
                analysis_html = self.md.render(result['summary'])
                analysis_section = f"""
                    <div>
                        <h4>Full Technical Analysis:</h4>
                        <div class="analysis-text">
                            {analysis_html}
                        </div>
                    </div>
                """
            else:
                # Create condensed summary with key information only
                analysis_section = f"""
                    <div>
                        <h4>Key Analysis Points:</h4>
                        <div class="analysis-text">
                            <p><strong>Action:</strong> {result['recommendation']} with {result['confidence']} confidence</p>
                            <p><strong>Current Price:</strong> ${result['price']:.2f}</p>
                            <p><strong>Technical Status:</strong> RSI: {result['rsi']:.1f}, MACD: {result['macd_signal']}</p>
                            <p><em>Full technical analysis charts are attached to this email.</em></p>
                        </div>
                    </div>
                """

            html += f"""
                <div id="{symbol_id}" class="stock-analysis">
                    <div class="stock-header">
                        <h3>{result['symbol']} Analysis</h3>
                    </div>
                    <div class="metrics">
                        <div class="metric">
                            <strong>Current Price:</strong> ${result['price']:.2f}
                        </div>
                        <div class="metric">
                            <strong>24h Change:</strong> <span class="{change_class}">{change_symbol}{result['change_pct']:.2f}%</span>
                        </div>
                        <div class="metric">
                            <strong>RSI:</strong> {result['rsi']:.2f}
                        </div>
                        <div class="metric">
                            <strong>MACD Signal:</strong> {result['macd_signal']}
                        </div>
                        <div class="metric">
                            <strong>Recommendation:</strong> 
                            <span class="recommendation {result['recommendation']} {result['confidence']}">
                                {result['recommendation']} (Confidence: {result['confidence']})
                            </span>
                        </div>
                    </div>
                    {analysis_section}
                    <a href="#top" class="jump-to-top">↑ Jump to Top</a>
                </div>
            """
        
        html += """
            <p>Note: Technical analysis charts are attached to this email.</p>
        </body>
        </html>
        """
        return html

    def send_report(self, recipient_emails, subject, analysis_results, plot_files, include_full_analysis=True):
        """
        Send email report to multiple recipients
        Args:
            recipient_emails: str or list of str - single email or multiple emails
            subject: str - email subject
            analysis_results: list of dict - analysis results
            plot_files: list - paths to plot files
            include_full_analysis: bool - whether to include full LLM analysis text (default: True)
        """
        # Convert single email to list for consistent handling
        if isinstance(recipient_emails, str):
            recipient_emails = [recipient_emails]

        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.sender_email
        msg['To'] = ', '.join(recipient_emails)  # Join multiple emails with comma

        # Create HTML content
        html_content = self.create_html_report(analysis_results, include_full_analysis)
        msg.attach(MIMEText(html_content, 'html'))

        # Attach plots
        for plot_file in plot_files:
            with open(plot_file, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-ID', f'<{os.path.basename(plot_file)}>')
                img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(plot_file))
                msg.attach(img)

        # Send email
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(
                    self.sender_email,
                    recipient_emails,
                    msg.as_string()
                )
            print(f"Email sent successfully to {', '.join(recipient_emails)}")
        except Exception as e:
            print(f"Error sending email: {str(e)}")

    def _split_multi_option_analysis(self, reasoning_text):
        """
        Split combined analysis text into individual option analyses

        Args:
            reasoning_text: Combined analysis text for multiple options

        Returns:
            List of individual option analysis texts
        """
        import re

        # Patterns to identify option analysis starts
        option_patterns = [
            r'This\s+\w+\s+(CALL|PUT)\s+option\s+(?:with\s+a\s+strike\s+of\s+)?\$(\d+(?:\.\d{2})?)',
            r'This\s+analysis\s+focuses\s+on\s+a\s+\w+\s+(CALL|PUT)\s+option\s+(?:with\s+a\s+strike\s+of\s+)?\$(\d+(?:\.\d{2})?)',
            r'(\w+)\s+(CALL|PUT)\s+option\s+(?:at\s+a\s+)?\$(\d+(?:\.\d{2})?)\s+strike'
        ]

        # Find all potential option starts
        option_starts = []
        for pattern in option_patterns:
            matches = re.finditer(pattern, reasoning_text, re.IGNORECASE)
            for match in matches:
                option_starts.append({
                    'start': match.start(),
                    'end': match.end(),
                    'match_text': match.group(0)
                })

        # Sort by position
        option_starts = sorted(option_starts, key=lambda x: x['start'])

        # If we found multiple option starts, split the text
        if len(option_starts) > 1:
            analyses = []
            for i, start_info in enumerate(option_starts):
                start_pos = start_info['start']
                end_pos = option_starts[i + 1]['start'] if i + 1 < len(option_starts) else len(reasoning_text)

                analysis_text = reasoning_text[start_pos:end_pos].strip()
                analyses.append(analysis_text)

            return analyses
        else:
            # Single option analysis
            return [reasoning_text]

    def _format_single_option_analysis(self, reasoning_text, action='N/A', confidence='N/A', risk_factor=None, option_number=None):
        """
        Format a single option analysis with extracted details

        Args:
            reasoning_text: Analysis text for one option
            action: Recommendation (BUY/SELL/HOLD/SKIP)
            confidence: Confidence level
            risk_factor: Risk factors (array or string)
            option_number: Option number for multi-option scenarios

        Returns:
            Formatted HTML string for single option
        """
        import re
        from datetime import date, timedelta, datetime

        # Extract option details from reasoning text using regex
        ticker = 'N/A'
        option_type = 'N/A'
        strike = 'N/A'
        days = 'N/A'
        expiration = 'N/A'

        if reasoning_text:
            # Extract ticker (e.g., "TSLA CALL option")
            ticker_match = re.search(r'(\b[A-Z]{2,5}\b)\s+(CALL|PUT)', reasoning_text, re.IGNORECASE)
            if ticker_match:
                ticker = ticker_match.group(1)
                option_type = ticker_match.group(2).upper()

            # Extract strike price (e.g., "strike of $450", "$450 strike")
            strike_match = re.search(r'strike\s+(?:of\s+)?\$(\d+(?:\.\d{2})?)|(\$\d+(?:\.\d{2})?)\s+strike', reasoning_text)
            if strike_match:
                strike_price = strike_match.group(1) or strike_match.group(2).replace('$', '')
                strike = f"${float(strike_price):.2f}"

            # Extract days to expiration (e.g., "expiring in 20 days", "20 days to expiration", "(6 days)")
            days_matches = [
                re.search(r'(?:expiring\s+(?:in\s+|on\s+)?)(\d+)\s+days?|(\d+)\s+days?\s+(?:to\s+)?expiration', reasoning_text),
                re.search(r'\((\d+)\s+days?\)', reasoning_text),
                re.search(r'with\s+only\s+(\d+)\s+days?\s+(?:to\s+expiration|left)', reasoning_text)
            ]

            for days_match in days_matches:
                if days_match:
                    days_num = days_match.group(1) or days_match.group(2)
                    days = f"{days_num} Days"

                    # Calculate approximate expiration date
                    try:
                        exp_date = date.today() + timedelta(days=int(days_num))
                        expiration = exp_date.strftime('%Y-%m-%d')
                    except:
                        expiration = 'N/A'
                    break

            # Extract specific expiration dates (e.g., "2025-10-03", "2025-12-19")
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', reasoning_text)
            if date_match:
                expiration = date_match.group(1)
                # Calculate days if we have the date
                try:
                    exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
                    today = date.today()
                    days_diff = (exp_date - today).days
                    days = f"{days_diff} Days"
                except:
                    pass

        # Create formatted heading with option number if provided
        option_prefix = f"Option {option_number}: " if option_number else ""
        heading = f"{option_prefix}{ticker} {option_type} | {strike} | {expiration} | {days} | {action} | {confidence}"

        # Format reasoning section
        reasoning_html = f"""
        <h4>Reasoning</h4>
        <div class="analysis-content">
            {self.md.render(reasoning_text)}
        </div>
        """ if reasoning_text else ""

        return f"""
        <h3>{heading}</h3>
        {reasoning_html}
        """

    def _format_multi_option_analysis(self, predictions):
        """
        Format options analysis using the llm_analysis from each prediction directly

        Args:
            predictions: List of prediction objects with option details and llm_analysis

        Returns:
            Formatted HTML string with properly separated option analyses
        """
        # Map each prediction's analysis to its details
        formatted_sections = []
        for i, prediction in enumerate(predictions):
            # Get analysis directly from prediction
            analysis_text = prediction.llm_analysis if hasattr(prediction, 'llm_analysis') and prediction.llm_analysis else ""

            # Create heading using prediction data
            option_prefix = f"Option {i + 1}: " if len(predictions) > 1 else ""
            heading = f"{option_prefix}{prediction.ticker} {prediction.option_type} | ${prediction.strike_price:.2f} | {prediction.expiration_date} | {prediction.days_to_expiration} Days | {prediction.recommendation} | {prediction.confidence}"

            # Format reasoning section
            reasoning_html = f"""
            <h4>Reasoning</h4>
            <div class="analysis-content">
                {self.md.render(analysis_text)}
            </div>
            """ if analysis_text else ""

            section_html = f"""
            <h3>{heading}</h3>
            {reasoning_html}
            """
            formatted_sections.append(section_html)

        return '\n'.join(formatted_sections)

    def _format_options_analysis(self, llm_analysis, prediction=None):
        """
        Format options analysis data with structured layout

        Args:
            llm_analysis: Either JSON object or plain text analysis
            prediction: Optional prediction object with options details

        Returns:
            Formatted HTML string
        """
        import json
        import re
        from datetime import datetime, date, timedelta

        # Try to parse as JSON first
        try:
            if isinstance(llm_analysis, str):
                analysis_data = json.loads(llm_analysis)
            else:
                analysis_data = llm_analysis

            # Get top-level fields first
            action = analysis_data.get('recommendation', 'N/A')
            confidence = analysis_data.get('confidence', 'N/A')
            reasoning = analysis_data.get('reasoning', '')
            risk_factor = analysis_data.get('risk_factors', '')

            # Check if this is a multi-option analysis
            individual_analyses = self._split_multi_option_analysis(reasoning)

            if len(individual_analyses) > 1:
                # Multiple options detected - format each separately
                formatted_sections = []
                for i, analysis_text in enumerate(individual_analyses):
                    section_html = self._format_single_option_analysis(analysis_text, action, confidence, risk_factor, i + 1)
                    formatted_sections.append(section_html)

                return '\n'.join(formatted_sections)
            else:
                # Single option - use existing logic
                return self._format_single_option_analysis(reasoning, action, confidence, risk_factor)

        except (json.JSONDecodeError, TypeError, AttributeError):
            # Fallback to original markdown rendering for plain text
            return self.md.render(str(llm_analysis))

    def create_options_html_report(self, options_results):
        """
        Create dedicated HTML report for options trading analysis

        Args:
            options_results: List of options analysis results
        """
        # Count statistics
        total_tickers = len(options_results)
        total_recommendations = sum(len(result.get('predictions', [])) for result in options_results)
        high_confidence_count = sum(1 for result in options_results
                                  for pred in result.get('predictions', [])
                                  if pred.confidence == 'HIGH')
        call_count = sum(1 for result in options_results
                        for pred in result.get('predictions', [])
                        if pred.option_type == 'CALL')
        put_count = sum(1 for result in options_results
                       for pred in result.get('predictions', [])
                       if pred.option_type == 'PUT')

        html = f"""
        <html>
        <head>
            <style>
                .recommendation {{
                    font-weight: bold;
                    padding: 5px 10px;
                    border-radius: 3px;
                }}
                .BUY {{
                    background-color: #d4edda;
                    color: #155724;
                }}
                .SELL {{
                    background-color: #f8d7da;
                    color: #721c24;
                }}
                .HOLD {{
                    background-color: #fff3cd;
                    color: #856404;
                }}
                .symbol-link {{
                    color: #0366d6;
                    text-decoration: none;
                }}
                .symbol-link:hover {{
                    text-decoration: underline;
                }}
                html {{
                    scroll-behavior: smooth;
                }}
                .stock-analysis {{
                    scroll-margin-top: 20px;
                    margin-bottom: 30px;
                }}
                table {{
                    border-collapse: collapse;
                    margin-bottom: 20px;
                    width: 100%;
                }}
                th, td {{
                    border: 1px solid #dee2e6;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f8f9fa;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                .option-type {{
                    font-weight: bold;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-size: 12px;
                }}
                .CALL {{
                    background-color: #d4edda;
                    color: #155724;
                }}
                .PUT {{
                    background-color: #f8d7da;
                    color: #721c24;
                }}
                .HIGH {{
                    border-left: 4px solid #28a745;
                }}
                .MEDIUM {{
                    border-left: 4px solid #ffc107;
                }}
                .LOW {{
                    border-left: 4px solid #6c757d;
                }}
                .ticker-analysis {{
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: white;
                    border-radius: 6px;
                    border-left: 4px solid #667eea;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .ticker-header {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 6px;
                    margin-bottom: 15px;
                }}
                .ticker-header h3 {{
                    margin: 0;
                    color: #333;
                    font-size: 22px;
                }}
                .options-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                    font-size: 14px;
                }}
                .options-table th {{
                    background-color: #667eea;
                    color: white;
                    padding: 8px;
                    text-align: left;
                    font-size: 12px;
                }}
                .options-table td {{
                    padding: 8px;
                    border-bottom: 1px solid #ddd;
                }}
                .options-table tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .analysis-text {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 6px;
                    margin-top: 15px;
                    line-height: 1.6;
                }}
                .analysis-text h4 {{
                    color: #667eea;
                    margin-top: 0;
                }}
                .analysis-text h3 {{
                    color: #333;
                    font-size: 18px;
                    font-weight: bold;
                    margin: 20px 0 15px 0;
                    padding: 10px;
                    background-color: #e9ecef;
                    border-radius: 4px;
                    border-left: 4px solid #667eea;
                }}
                .analysis-content {{
                    margin: 10px 0;
                    padding: 0 10px;
                }}
                .analysis-content h4 {{
                    color: #495057;
                    font-size: 16px;
                    margin: 15px 0 10px 0;
                    font-weight: bold;
                }}
                .analysis-content p {{
                    margin-bottom: 10px;
                    line-height: 1.6;
                }}
                .analysis-content ul, .analysis-content ol {{
                    margin: 10px 0 10px 20px;
                }}
                .analysis-content li {{
                    margin-bottom: 5px;
                    line-height: 1.5;
                }}
                .analysis-content strong {{
                    font-weight: bold;
                    color: #333;
                }}
                .analysis-content em {{
                    font-style: italic;
                }}
                .symbol-link {{
                    color: #667eea;
                    text-decoration: none;
                    font-weight: bold;
                }}
                .symbol-link:hover {{
                    text-decoration: underline;
                }}
                .jump-to-top {{
                    display: inline-block;
                    margin-top: 15px;
                    padding: 8px 15px;
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    color: #0366d6;
                    text-decoration: none;
                    font-size: 14px;
                }}
                .jump-to-top:hover {{
                    background-color: #e9ecef;
                    text-decoration: none;
                }}
            </style>
        </head>
        <body>
            <div id="top"></div>
            <h1>Options Trading Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Market Overview</h2>
            <div class="metrics">
                <div class="metric">
                    <strong>Tickers Analyzed:</strong> {total_tickers}
                </div>
                <div class="metric">
                    <strong>Total Recommendations:</strong> {total_recommendations}
                </div>
                <div class="metric">
                    <strong>High Confidence:</strong> {high_confidence_count}
                </div>
                <div class="metric">
                    <strong>CALL Options:</strong> {call_count}
                </div>
                <div class="metric">
                    <strong>PUT Options:</strong> {put_count}
                </div>
            </div>

                <h2>Summary Table</h2>
                <table class="summary-table">
                    <tr>
                        <th>Ticker</th>
                        <th>Current Price</th>
                        <th>Best Call</th>
                        <th>Best Put</th>
                        <th>Top Confidence</th>
                        <th>Options Count</th>
                    </tr>
        """

        # Add summary rows
        for result in options_results:
            ticker = result['ticker']
            current_price = result.get('current_price', 0)
            predictions = result.get('predictions', [])
            best_call = result.get('best_call_prediction')
            best_put = result.get('best_put_prediction')

            best_call_str = f"${best_call.strike_price:.2f} ({best_call.days_to_expiration}d)" if best_call else "N/A"
            best_put_str = f"${best_put.strike_price:.2f} ({best_put.days_to_expiration}d)" if best_put else "N/A"

            top_confidence = "LOW"
            if predictions:
                confidences = [p.confidence for p in predictions]
                if "HIGH" in confidences:
                    top_confidence = "HIGH"
                elif "MEDIUM" in confidences:
                    top_confidence = "MEDIUM"

            symbol_id = f"analysis-{ticker}"

            html += f"""
                <tr>
                    <td><strong><a href="#{symbol_id}" class="symbol-link">{ticker}</a></strong></td>
                    <td>${current_price:.2f}</td>
                    <td>{best_call_str}</td>
                    <td>{best_put_str}</td>
                    <td><span class="confidence {top_confidence}">{top_confidence}</span></td>
                    <td>{len(predictions)}</td>
                </tr>
            """

        html += """
                </table>

                <h2>Detailed Analysis</h2>
        """

        # Detailed analysis for each ticker
        for result in options_results:
            ticker = result['ticker']
            current_price = result.get('current_price', 0)
            predictions = result.get('predictions', [])
            summary = result.get('summary', '')
            symbol_id = f"analysis-{ticker}"

            html += f"""
                <div id="{symbol_id}" class="ticker-analysis">
                    <div class="ticker-header">
                        <h3>{ticker} Options Analysis</h3>
                        <p><strong>Current Price:</strong> ${current_price:.2f} | <strong>Recommendations:</strong> {len(predictions)}</p>
                    </div>
            """

            if predictions:
                html += """
                    <h4>Options Recommendations</h4>
                    <table class="options-table">
                        <tr>
                            <th>Type</th>
                            <th>Strike</th>
                            <th>Expiration</th>
                            <th>Days</th>
                            <th>Action</th>
                            <th>Confidence</th>
                            <th>Premium</th>
                            <th>Target</th>
                        </tr>
                """

                for pred in predictions:
                    target_str = f"${pred.target_premium:.2f}" if pred.target_premium else "N/A"
                    html += f"""
                        <tr>
                            <td><span class="option-type {pred.option_type}">{pred.option_type}</span></td>
                            <td>${pred.strike_price:.2f}</td>
                            <td>{pred.expiration_date}</td>
                            <td>{pred.days_to_expiration}</td>
                            <td>{pred.recommendation}</td>
                            <td><span class="confidence {pred.confidence}">{pred.confidence}</span></td>
                            <td>${pred.entry_premium:.2f}</td>
                            <td>{target_str}</td>
                        </tr>
                    """

                html += "</table>"

            # Add LLM analysis with structured formatting
            if predictions:

                analysis_html = self._format_multi_option_analysis(predictions)

                html += f"""
                    <div class="analysis-text">
                        <h4>AI Analysis & Market Insights</h4>
                        {analysis_html}
                    </div>
                """

            # Add summary if available
            if summary:
                html += f"""
                    <div class="analysis-text">
                        <h4>Technical Summary</h4>
                        <pre>{summary}</pre>
                    </div>
                """

            html += '<a href="#top" class="jump-to-top">↑ Jump to Top</a>'
            html += "</div>"

        html += """
            <p>Note: Technical analysis charts are attached to this email.</p>
            <p style="color: #666; font-size: 12px;">This report is generated by AI-powered options analysis. Please consult with a financial advisor before making trading decisions.</p>
        </body>
        </html>
        """

        return html

    def send_options_report(self, recipient_emails, subject, options_results, plot_files=None):
        """
        Send dedicated options trading analysis email report

        Args:
            recipient_emails: List of recipient email addresses
            subject: Email subject line
            options_results: List of options analysis results
            plot_files: List of plot file paths to attach (optional)
        """
        try:
            # Create HTML content
            html_content = self.create_options_html_report(options_results)

            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(recipient_emails)

            # Add HTML content
            msg.attach(MIMEText(html_content, 'html'))

            # Add plot attachments if provided
            if plot_files:
                for plot_file in plot_files:
                    if plot_file and os.path.exists(plot_file):
                        with open(plot_file, 'rb') as f:
                            img_data = f.read()

                        img = MIMEImage(img_data)
                        img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(plot_file))
                        msg.attach(img)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(
                    self.sender_email,
                    recipient_emails,
                    msg.as_string()
                )
            print(f"Options analysis email sent successfully to {', '.join(recipient_emails)}")

        except Exception as e:
            print(f"Error sending options email: {str(e)}")

    def send_earnings_report(self, recipient_emails, subject, analysis_results, plot_files, performance_summary):
        """
        Send earnings analysis email report with enhanced formatting

        Args:
            recipient_emails: str or list of str - email addresses
            subject: str - email subject
            analysis_results: list of dict - earnings analysis results with keys:
                - ticker, earnings_date, days_until_earnings, current_price, prediction
            plot_files: list of str - paths to plot files
            performance_summary: dict - performance statistics with keys:
                - total_predictions, success_rate, avg_pre_earnings_return,
                  avg_post_earnings_return, avg_total_return
        """
        try:
            if not analysis_results:
                print("No analysis results to send")
                return

            # Convert single email to list
            if isinstance(recipient_emails, str):
                recipient_emails = [recipient_emails]

            # Build HTML with matching structure to create_html_report
            body = """
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; }
        h1, h2, h3 { color: #2c3e50; }

        /* Recommendation and confidence badges */
        .recommendation {
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 3px;
            display: inline-block;
        }
        .BUY {
            background-color: #d4edda;
            color: #155724;
        }
        .SELL {
            background-color: #f8d7da;
            color: #721c24;
        }
        .HOLD {
            background-color: #fff3cd;
            color: #856404;
        }
        .UP {
            background-color: #d4edda;
            color: #155724;
        }
        .DOWN {
            background-color: #f8d7da;
            color: #721c24;
        }
        .NEUTRAL {
            background-color: #fff3cd;
            color: #856404;
        }
        .HIGH {
            border-left: 4px solid #28a745;
            padding-left: 10px;
        }
        .MEDIUM {
            border-left: 4px solid #ffc107;
            padding-left: 10px;
        }
        .LOW {
            border-left: 4px solid #6c757d;
            padding-left: 10px;
        }

        /* Clickable ticker links */
        .symbol-link {
            color: #0366d6;
            text-decoration: none;
            font-weight: bold;
        }
        .symbol-link:hover {
            text-decoration: underline;
        }

        /* Smooth scrolling */
        html {
            scroll-behavior: smooth;
        }

        /* Stock analysis sections */
        .stock-analysis {
            scroll-margin-top: 20px;
            margin-bottom: 30px;
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }

        /* Jump to top link */
        .jump-to-top {
            display: inline-block;
            margin-top: 15px;
            padding: 8px 15px;
            background-color: #e9ecef;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            color: #0366d6;
            text-decoration: none;
            font-size: 14px;
        }
        .jump-to-top:hover {
            background-color: #dee2e6;
            text-decoration: none;
        }

        /* Metrics display */
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        .metric {
            background-color: #ffffff;
            padding: 12px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
        }

        /* Performance summary */
        .performance-summary {
            background-color: #ecf0f1;
            padding: 20px;
            margin: 15px 0;
            border-radius: 5px;
            border: 1px solid #bdc3c7;
        }

        /* Summary table */
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            background-color: #ffffff;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        tr:hover {
            background-color: #e9ecef;
        }

        /* Analysis text formatting */
        .analysis-text {
            margin-top: 15px;
            padding: 15px;
            background-color: #ffffff;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .earnings-date {
            color: #e74c3c;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div id="top"></div>
    <h1>Earnings Analysis Report</h1>
    <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>

    <h2>Performance Summary (Last 7 Days)</h2>
    <div class="performance-summary">
"""

            # Add performance metrics in card format
            if performance_summary and performance_summary.get('total_predictions', 0) > 0:
                body += """
        <div class="metrics">
            <div class="metric">
                <strong>Total Predictions:</strong> """ + str(performance_summary.get('total_predictions', 0)) + """
            </div>
            <div class="metric">
                <strong>Success Rate:</strong> """ + f"{performance_summary.get('success_rate', 0)*100:.1f}%" + """
            </div>
            <div class="metric">
                <strong>Avg Pre-Earnings Return:</strong> """ + f"{performance_summary.get('avg_pre_earnings_return', 0):.2f}%" + """
            </div>
            <div class="metric">
                <strong>Avg Post-Earnings Return:</strong> """ + f"{performance_summary.get('avg_post_earnings_return', 0):.2f}%" + """
            </div>
            <div class="metric">
                <strong>Avg Total Return:</strong> """ + f"{performance_summary.get('avg_total_return', 0):.2f}%" + """
            </div>
        </div>
"""
            else:
                body += "<p>No recent earnings predictions to report.</p>"

            body += """
    </div>

    <h2>Summary Table - Stocks with Upcoming Earnings</h2>
    <table>
        <tr>
            <th>Ticker</th>
            <th>Earnings Date</th>
            <th>Days Until</th>
            <th>Current Price</th>
            <th>Pre-Earnings</th>
            <th>Post-Earnings</th>
            <th>Confidence</th>
        </tr>
"""

            # Add summary table rows with clickable tickers
            for result in analysis_results:
                ticker = result['ticker']
                prediction = result['prediction']
                ticker_id = f"analysis-{ticker}"

                body += f"""
        <tr>
            <td><a href="#{ticker_id}" class="symbol-link">{ticker}</a></td>
            <td class="earnings-date">{result['earnings_date']}</td>
            <td>{result['days_until_earnings']} days</td>
            <td>${result['current_price']:.2f}</td>
            <td><span class="recommendation {prediction.pre_earnings_recommendation}">{prediction.pre_earnings_recommendation}</span></td>
            <td><span class="recommendation {prediction.post_earnings_direction}">{prediction.post_earnings_direction}</span></td>
            <td><span class="recommendation {prediction.confidence}">{prediction.confidence}</span></td>
        </tr>
"""

            body += """
    </table>

    <h2>Detailed Analysis</h2>
"""

            # Add detailed analysis sections with anchor IDs
            for result in analysis_results:
                ticker = result['ticker']
                prediction = result['prediction']
                ticker_id = f"analysis-{ticker}"

                body += f"""
    <div id="{ticker_id}" class="stock-analysis">
        <h3>{ticker} - Earnings Analysis</h3>

        <div class="metrics">
            <div class="metric">
                <strong>Earnings Date:</strong> <span class="earnings-date">{result['earnings_date']}</span>
            </div>
            <div class="metric">
                <strong>Days Until Earnings:</strong> {result['days_until_earnings']} days
            </div>
            <div class="metric">
                <strong>Current Price:</strong> ${result['current_price']:.2f}
            </div>
"""

                if prediction.pre_earnings_target:
                    body += f"""
            <div class="metric">
                <strong>Pre-Earnings Target:</strong> ${prediction.pre_earnings_target:.2f}
            </div>
"""

                if prediction.post_earnings_target:
                    body += f"""
            <div class="metric">
                <strong>Post-Earnings Target:</strong> ${prediction.post_earnings_target:.2f}
            </div>
"""

                if prediction.stop_loss:
                    body += f"""
            <div class="metric">
                <strong>Stop Loss:</strong> ${prediction.stop_loss:.2f}
            </div>
"""

                body += f"""
            <div class="metric {prediction.confidence}">
                <strong>Pre-Earnings Recommendation:</strong><br>
                <span class="recommendation {prediction.pre_earnings_recommendation}">{prediction.pre_earnings_recommendation}</span>
            </div>
            <div class="metric {prediction.confidence}">
                <strong>Post-Earnings Direction:</strong><br>
                <span class="recommendation {prediction.post_earnings_direction}">{prediction.post_earnings_direction}</span>
            </div>
        </div>
"""

                # Add LLM analysis if available
                if prediction.llm_analysis:
                    body += f"""
        <h4>Detailed Analysis:</h4>
        <div class="analysis-text">{prediction.llm_analysis}</div>
"""

                body += """
        <a href="#top" class="jump-to-top">↑ Jump to Top</a>
    </div>
"""

            body += """
    <p><em>Disclaimer: This analysis is for informational purposes only and should not be considered as financial advice.</em></p>
</body>
</html>
"""

            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(recipient_emails)

            # Add HTML content
            msg.attach(MIMEText(body, 'html'))

            # Add plot attachments if provided
            if plot_files:
                for plot_file in plot_files:
                    if plot_file and os.path.exists(plot_file):
                        with open(plot_file, 'rb') as f:
                            img_data = f.read()

                        img = MIMEImage(img_data)
                        img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(plot_file))
                        msg.attach(img)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(
                    self.sender_email,
                    recipient_emails,
                    msg.as_string()
                )
            print(f"Earnings analysis email sent successfully to {', '.join(recipient_emails)}")

        except Exception as e:
            print(f"Error sending earnings email: {str(e)}")
            raise

    def send_earnings_reflection_report(self, recipient_emails, subject, reflection_text, stats, beat_miss_stats, days=90):
        """
        Send earnings reflection email report with performance analysis

        Args:
            recipient_emails: str or list of str - email addresses
            subject: str - email subject
            reflection_text: str - LLM-generated reflection analysis
            stats: dict - performance statistics with keys:
                - total_predictions, success_rate, avg_total_return, beat_rate
            beat_miss_stats: DataFrame - performance breakdown by earnings outcome
            days: int - time period analyzed (default: 90)
        """
        try:
            # Convert single email to list
            if isinstance(recipient_emails, str):
                recipient_emails = [recipient_emails]

            # Build HTML with enhanced styling
            body = """
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; }
        h1, h2, h3 { color: #2c3e50; }

        /* Metrics display */
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        .metric {
            background-color: #ffffff;
            padding: 12px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
        }

        /* Performance section */
        .performance-section {
            background-color: #f8f9fa;
            padding: 20px;
            margin: 15px 0;
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }

        /* Beat/Miss table */
        .beat-miss-table {
            background-color: #ffffff;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            overflow-x: auto;
            white-space: pre;
        }

        /* Reflection text */
        .reflection-text {
            background-color: #ffffff;
            padding: 20px;
            margin: 15px 0;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        /* Color indicators */
        .positive { color: #28a745; font-weight: bold; }
        .negative { color: #dc3545; font-weight: bold; }
        .neutral { color: #6c757d; font-weight: bold; }
    </style>
</head>
<body>
    <h1>Earnings Prediction Performance Reflection</h1>
    <p>Generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + f"""</p>
    <p><strong>Analysis Period:</strong> Last {days} Days</p>

    <h2>Overall Statistics</h2>
    <div class="performance-section">
        <div class="metrics">
            <div class="metric">
                <strong>Total Predictions:</strong><br>
                """ + str(stats.get('total_predictions', 0)) + """
            </div>
            <div class="metric">
                <strong>Success Rate:</strong><br>
                <span class=\"""" + ('positive' if stats.get('success_rate', 0) >= 0.6 else 'negative' if stats.get('success_rate', 0) < 0.5 else 'neutral') + """">
                    """ + f"{stats.get('success_rate', 0)*100:.1f}%" + """
                </span>
            </div>
            <div class="metric">
                <strong>Avg Total Return:</strong><br>
                <span class=\"""" + ('positive' if stats.get('avg_total_return', 0) > 0 else 'negative' if stats.get('avg_total_return', 0) < 0 else 'neutral') + """">
                    """ + f"{stats.get('avg_total_return', 0):.2f}%" + """
                </span>
            </div>
            <div class="metric">
                <strong>Earnings Beat Rate:</strong><br>
                <span class=\"""" + ('positive' if stats.get('beat_rate', 0) >= 0.5 else 'negative') + """">
                    """ + f"{stats.get('beat_rate', 0)*100:.1f}%" + """
                </span>
            </div>
        </div>
    </div>

    <h2>Performance by Earnings Outcome</h2>
    <div class="beat-miss-table">
"""

            # Add beat/miss statistics
            if beat_miss_stats is not None and not beat_miss_stats.empty:
                body += beat_miss_stats.to_string()
            else:
                body += "No data available"

            body += """
    </div>

    <h2>LLM Reflection and Insights</h2>
    <div class="reflection-text">""" + reflection_text + """</div>

    <p><em>Disclaimer: This analysis is for informational purposes only and should not be considered as financial advice.</em></p>
</body>
</html>
"""

            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(recipient_emails)

            # Add HTML content
            msg.attach(MIMEText(body, 'html'))

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(
                    self.sender_email,
                    recipient_emails,
                    msg.as_string()
                )
            print(f"Earnings reflection email sent successfully to {', '.join(recipient_emails)}")

        except Exception as e:
            print(f"Error sending earnings reflection email: {str(e)}")
            raise

    def send_email(self, recipient_emails, subject, body, attachments=None, is_html=True):
        """
        Send a generic email with optional attachments

        Args:
            recipient_emails: str or list of str - email addresses
            subject: str - email subject
            body: str - email body content (HTML or plain text)
            attachments: list of str - paths to attachment files (optional)
            is_html: bool - whether body is HTML (default: True)
        """
        try:
            # Convert single email to list
            if isinstance(recipient_emails, str):
                recipient_emails = [recipient_emails]

            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = ', '.join(recipient_emails)

            # Attach body content
            content_type = 'html' if is_html else 'plain'
            msg.attach(MIMEText(body, content_type))

            # Attach files if provided
            if attachments:
                for file_path in attachments:
                    if file_path and os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            img_data = f.read()

                        # Try to attach as image first, otherwise as generic file
                        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                            img = MIMEImage(img_data)
                        else:
                            img = MIMEApplication(img_data)

                        img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(file_path))
                        msg.attach(img)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.sendmail(
                    self.sender_email,
                    recipient_emails,
                    msg.as_string()
                )
            print(f"Email sent successfully to {', '.join(recipient_emails)}")

        except Exception as e:
            print(f"Error sending email: {str(e)}")
            raise