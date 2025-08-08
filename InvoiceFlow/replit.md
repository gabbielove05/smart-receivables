# JPMorgan Smart Receivables Navigator

## Overview

The Smart Receivables Navigator is a comprehensive AI-powered financial dashboard built with Streamlit for managing accounts receivable at JPMorgan. The application combines advanced analytics, machine learning-driven insights, and automated action recommendations to optimize receivables management and cash flow.

The system provides executives and financial professionals with real-time insights through multiple specialized dashboards including a one-minute CFO summary, KPI cockpit with DSO tracking, exception heat maps for root-cause analysis, and interactive what-if simulators for scenario modeling. The application features an AI assistant powered by OpenAI GPT-4o for natural language queries and automated communication capabilities through SMTP, SendGrid, and MS Teams integrations.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
The application uses Streamlit as the primary web framework with a single-page application design. The UI is organized into tab-based navigation without sidebars, featuring a four-column configuration panel at the top and full-width dashboard displays. Custom CSS styling implements JPMorgan brand colors (blue #0066CC, dark blue #003366, gold #FFD700) with gradient backgrounds and responsive metric cards.

### Backend Architecture
The codebase follows a modular architecture with separation of concerns:

- **Main Application (`app.py`)**: Entry point handling page configuration, environment validation, and tab navigation
- **Dashboard Components (`dashboards.py`)**: Renders CFO panel, KPI cockpit, heat maps, and what-if simulators using Plotly for interactive visualizations
- **AI Chatbot (`chatbot.py`)**: OpenAI GPT-4o integration with finance FAQ fallback system for natural language Q&A
- **Smart Actions (`smart_actions.py`)**: ML-powered priority queue using Isolation Forest and Gradient Boosting for next-best-action recommendations
- **Data Quality (`data_quality.py`)**: Comprehensive validation system with quality scoring and Great Expectations integration patterns
- **Machine Learning (`ml_models.py`)**: Implements priority scoring, anomaly detection, and SHAP-based model explainability
- **Utilities (`utils.py`)**: Data processing, normalization, and sample data generation functions

### Data Processing Pipeline
The ETL pipeline normalizes invoice and payment data through pandas DataFrames, merging on `invoice_id` as the primary key. The system supports both file upload (CSV) and environment-based data sources (Snowflake, S3). Data validation includes completeness, uniqueness, validity, consistency, timeliness, and accuracy checks with automated quality scoring.

### Machine Learning Components
The ML architecture uses scikit-learn for multiple models:
- Isolation Forest for anomaly detection
- Gradient Boosting Classifier for priority scoring
- Random Forest for risk assessment
- SHAP integration for model explainability and transparency

Models generate priority scores for receivables and power the next-best-action recommendation engine with automated follow-up suggestions.

### Communication Infrastructure
Multi-channel communication system supporting:
- SMTP email with SSL/TLS for payment reminders
- SendGrid API integration for enhanced email delivery
- MS Teams webhook alerts for real-time notifications
- Simulated phone call functionality for urgent collections

### Security and Configuration
Environment variable-based configuration for sensitive data (API keys, SMTP credentials, webhook URLs). The application includes comprehensive error handling, logging, and graceful fallbacks when external services are unavailable.

## External Dependencies

### Core Framework Dependencies
- **Streamlit**: Web application framework for dashboard rendering and user interface
- **Pandas/NumPy**: Data manipulation and numerical computation for financial analytics
- **Plotly**: Interactive charting and visualization library for dashboard components
- **Scikit-learn**: Machine learning library for predictive models and anomaly detection

### AI and ML Services
- **OpenAI API**: GPT-4o integration for natural language processing and executive summary generation
- **SHAP**: Model explainability library for transparent ML decision-making

### Communication Integrations
- **SMTP Services**: Email server integration for payment reminders and notifications
- **SendGrid API**: Enhanced email delivery service with tracking capabilities
- **MS Teams Webhooks**: Real-time alert system for urgent collection issues

### Data Quality and Validation
- **Great Expectations**: Data validation framework for quality assurance and profiling
- **Faker**: Sample data generation for testing and demonstration purposes

### Optional Enhancement Services
- **Snowflake**: Cloud data warehouse connection for enterprise data sources
- **AWS S3**: Cloud storage integration for file-based data ingestion
- **Phone/SMS APIs**: External telephony services for automated calling functionality

The application is designed to gracefully handle missing dependencies and provides fallback functionality when external services are unavailable, ensuring core functionality remains accessible in various deployment environments.