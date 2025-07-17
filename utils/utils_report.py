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