"""
Advanced Reporting System Module for Manus Clone

This module provides comprehensive reporting capabilities including:
- Detailed session reports
- Data visualization
- Export functionality
- Custom report templates
- Scheduled reporting
"""

import os
import re
import json
import time
import uuid
import logging
import datetime
import tempfile
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import csv
import markdown
import pdfkit
from jinja2 import Template
import arabic_reshaper
from bidi.algorithm import get_display

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedReporting:
    """Advanced Reporting System for comprehensive report generation and visualization"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Advanced Reporting system
        
        Args:
            config: Configuration parameters
        """
        self.config = config
        self.reports_dir = config.get('REPORTS_DIR', os.path.join(os.path.expanduser('~'), '.manus_clone', 'reports'))
        self.templates_dir = config.get('TEMPLATES_DIR', os.path.join(os.path.expanduser('~'), '.manus_clone', 'report_templates'))
        self.data_dir = config.get('DATA_DIR', os.path.join(os.path.expanduser('~'), '.manus_clone', 'data'))
        
        # Create directories if they don't exist
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize default templates
        self._initialize_default_templates()
        
        # Configure matplotlib for Arabic text
        self._setup_arabic_fonts()
        
        logger.info("Advanced Reporting system initialized")
    
    def _setup_arabic_fonts(self):
        """
        Configure matplotlib to properly display Arabic text
        """
        try:
            # Set font properties for Arabic text
            plt.rcParams['font.family'] = 'Arial'
            logger.info("Arabic font configuration set up for matplotlib")
        except Exception as e:
            logger.error(f"Error setting up Arabic fonts: {str(e)}")
    
    def _initialize_default_templates(self):
        """
        Initialize default report templates
        """
        try:
            # Define default templates
            default_templates = {
                'session_summary': {
                    'name': 'Session Summary',
                    'description': 'Summary of a single session with the AI assistant',
                    'format': 'html',
                    'template': '''
                    <!DOCTYPE html>
                    <html dir="{{ text_direction }}">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>{{ title }}</title>
                        <style>
                            body {
                                font-family: Arial, sans-serif;
                                line-height: 1.6;
                                color: #333;
                                max-width: 1200px;
                                margin: 0 auto;
                                padding: 20px;
                                direction: {{ text_direction }};
                            }
                            h1, h2, h3 {
                                color: #2c3e50;
                            }
                            .header {
                                border-bottom: 2px solid #3498db;
                                margin-bottom: 20px;
                                padding-bottom: 10px;
                            }
                            .section {
                                margin-bottom: 30px;
                            }
                            .metrics {
                                display: flex;
                                flex-wrap: wrap;
                                gap: 20px;
                                margin-bottom: 20px;
                            }
                            .metric-card {
                                background-color: #f8f9fa;
                                border-radius: 8px;
                                padding: 15px;
                                min-width: 200px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            }
                            .metric-title {
                                font-size: 14px;
                                color: #7f8c8d;
                                margin-bottom: 5px;
                            }
                            .metric-value {
                                font-size: 24px;
                                font-weight: bold;
                                color: #2980b9;
                            }
                            .conversation {
                                border: 1px solid #ddd;
                                border-radius: 8px;
                                padding: 15px;
                                margin-bottom: 20px;
                            }
                            .message {
                                padding: 10px;
                                margin-bottom: 10px;
                                border-radius: 8px;
                            }
                            .user-message {
                                background-color: #f1f8ff;
                                border-left: 4px solid #3498db;
                            }
                            .assistant-message {
                                background-color: #f8f9fa;
                                border-left: 4px solid #2ecc71;
                            }
                            .timestamp {
                                font-size: 12px;
                                color: #95a5a6;
                                margin-bottom: 5px;
                            }
                            .chart-container {
                                margin: 20px 0;
                                text-align: center;
                            }
                            .chart-container img {
                                max-width: 100%;
                                height: auto;
                                border-radius: 8px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            }
                            table {
                                width: 100%;
                                border-collapse: collapse;
                                margin-bottom: 20px;
                            }
                            th, td {
                                padding: 12px 15px;
                                text-align: {{ text_align }};
                                border-bottom: 1px solid #ddd;
                            }
                            th {
                                background-color: #f8f9fa;
                                font-weight: bold;
                            }
                            tr:hover {
                                background-color: #f5f5f5;
                            }
                            .footer {
                                margin-top: 30px;
                                padding-top: 10px;
                                border-top: 1px solid #ddd;
                                font-size: 12px;
                                color: #7f8c8d;
                                text-align: center;
                            }
                        </style>
                    </head>
                    <body>
                        <div class="header">
                            <h1>{{ title }}</h1>
                            <p>{{ description }}</p>
                        </div>
                        
                        <div class="section">
                            <h2>{{ session_info_title }}</h2>
                            <div class="metrics">
                                <div class="metric-card">
                                    <div class="metric-title">{{ date_title }}</div>
                                    <div class="metric-value">{{ session_date }}</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-title">{{ duration_title }}</div>
                                    <div class="metric-value">{{ session_duration }}</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-title">{{ messages_title }}</div>
                                    <div class="metric-value">{{ message_count }}</div>
                                </div>
                                <div class="metric-card">
                                    <div class="metric-title">{{ tasks_title }}</div>
                                    <div class="metric-value">{{ task_count }}</div>
                                </div>
                            </div>
                        </div>
                        
                        {% if charts %}
                        <div class="section">
                            <h2>{{ charts_title }}</h2>
                            {% for chart in charts %}
                            <div class="chart-container">
                                <h3>{{ chart.title }}</h3>
                                <img src="data:image/png;base64,{{ chart.data }}" alt="{{ chart.title }}">
                            </div>
                            {% endfor %}
                        </div>
                        {% endif %}
                        
                        {% if tasks %}
                        <div class="section">
                            <h2>{{ tasks_summary_title }}</h2>
                            <table>
                                <thead>
                                    <tr>
                                        <th>{{ task_name_title }}</th>
                                        <th>{{ task_status_title }}</th>
                                        <th>{{ task_duration_title }}</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for task in tasks %}
                                    <tr>
                                        <td>{{ task.name }}</td>
                                        <td>{{ task.status }}</td>
                                        <td>{{ task.duration }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        {% endif %}
                        
                        {% if conversation %}
                        <div class="section">
                            <h2>{{ conversation_title }}</h2>
                            <div class="conversation">
                                {% for message in conversation %}
                                <div class="message {{ message.type }}-message">
                                    <div class="timestamp">{{ message.timestamp }}</div>
                                    <div class="content">{{ message.content }}</div>
                                </div>
                                {% endfor %}
                            </div>
                        </div>
                        {% endif %}
                        
                        {% if resources %}
                        <div class="section">
                            <h2>{{ resources_title }}</h2>
                            <table>
                                <thead>
                                    <tr>
                                        <th>{{ resource_name_title }}</th>
                                        <th>{{ resource_type_title }}</th>
                                        <th>{{ resource_url_title }}</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for resource in resources %}
                                    <tr>
                                        <td>{{ resource.name }}</td>
                                        <td>{{ resource.type }}</td>
                                        <td><a href="{{ resource.url }}" target="_blank">{{ resource.url }}</a></td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        {% endif %}
                        
                        <div class="footer">
                            {{ footer_text }} | {{ generated_date }}
                        </div>
                    </body>
                    </html>
                    '''
                },
                'activity_report': {
                    'name': 'Activity Report',
                    'description': 'Summary of activities over a period of time',
                    'format': 'html',
                    'template': '''
                    <!DOCTYPE html>
                    <html dir="{{ text_direction }}">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>{{ title }}</title>
                        <style>
                            body {
                                font-family: Arial, sans-serif;
                                line-height: 1.6;
                                color: #333;
                                max-width: 1200px;
                                margin: 0 auto;
                                padding: 20px;
                                direction: {{ text_direction }};
                            }
                            h1, h2, h3 {
                                color: #2c3e50;
                            }
                            .header {
                                border-bottom: 2px solid #3498db;
                                margin-bottom: 20px;
                                padding-bottom: 10px;
                            }
                            .section {
                                margin-bottom: 30px;
                            }
                            .metrics {
                                display: flex;
                                flex-wrap: wrap;
                                gap: 20px;
                                margin-bottom: 20px;
                            }
                            .metric-card {
                                background-color: #f8f9fa;
                                border-radius: 8px;
                                padding: 15px;
                                min-width: 200px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            }
                            .metric-title {
                                font-size: 14px;
                                color: #7f8c8d;
                                margin-bottom: 5px;
                            }
                            .metric-value {
                                font-size: 24px;
                                font-weight: bold;
          
(Content truncated due to size limit. Use line ranges to read in chunks)