"""
JPMorgan Smart Receivables Navigator - Main Application
A comprehensive financial dashboard for receivables management with AI-powered insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
import json
import time

# Import custom modules
from chatbot import ask_llm
from integrations import call_client, send_teams_alert
from dashboards import render_cfo_panel, render_kpi_dashboard, render_heatmap, render_what_if_simulator
from smart_actions import generate_next_best_actions, render_automation_center
from data_quality import validate_data_quality, get_quality_score
from ml_models import train_models, get_model_explanations
from utils import normalize_data, generate_sample_data
from email_utils import draft_email, send_email, FIXED_LINE



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="JPMorgan Smart Receivables Navigator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': 'JPMorgan Smart Receivables Navigator - AI-Powered Financial Intelligence Platform'
    }
)

# Normal Streamlit mode - no custom styling
# Updated: 2025-08-08 - API key debugging

def check_environment_variables():
    """Check for required environment variables and display warnings."""
    # OpenRouter API is configured by default, no required vars
    required_vars = {}
    
    missing_required = []
    
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_required.append(f"{var}: {description}")
    
    if missing_required:
        st.error(f"❌ Missing required environment variables:\n" + "\n".join([f"• {var}" for var in missing_required]))
        st.info("Please set these environment variables for full functionality.")

def load_data():
    """Load and process invoice and payment data."""
    try:
        # Check for environment-based data sources first
        snowflake_conn = os.getenv('SNOWFLAKE_CONN')
        s3_bucket = os.getenv('S3_BUCKET')
        
        if snowflake_conn or s3_bucket:
            st.info("🔗 Environment data sources detected but not implemented in this demo. Please upload CSV files.")
        
        # User information at the top
        st.markdown("### 👤 User Information")
        col1, col2 = st.columns(2)
        
        with col1:
            user_name = st.text_input(
                "Your name:", 
                placeholder="John Smith",
                key="user_name_input",
                value=st.session_state.get("user_name", "")
            )
        
        with col2:
            user_email = st.text_input(
                "Your email address:", 
                placeholder="john.smith@company.com",
                key="user_email_input",
                value=st.session_state.get("recipient", "")
            )
        
        # Save user info
        if st.button("💾 Save Information", key="save_user_info"):
            if user_name and user_email and "@" in user_email:
                st.session_state["user_name"] = user_name
                st.session_state["recipient"] = user_email
                st.success(f"✅ Information saved: {user_name} ({user_email})")
                st.rerun()
            else:
                st.error("Please enter both name and a valid email address")
        
        # Show current info if set
        if st.session_state.get("user_name") and st.session_state.get("recipient"):
            st.info(f"👤 Logged in as: {st.session_state['user_name']} ({st.session_state['recipient']})")
        
        st.markdown("---")
        
        # File uploaders
        st.markdown("### 📁 Upload Data")
        col1, col2 = st.columns(2)
        
        with col1:
            invoice_file = st.file_uploader(
                "📋 Upload Invoice Data",
                type=['csv'],
                help="CSV file with invoice data including invoice_id, amount, due_date, customer_id"
            )
            
        with col2:
            payment_file = st.file_uploader(
                "💰 Upload Payment Data", 
                type=['csv'],
                help="CSV file with payment data including payment_id, invoice_id, amount, payment_date"
            )
        
        invoices_df = None
        payments_df = None
        
        if invoice_file is not None:
            try:
                invoices_df = pd.read_csv(invoice_file)
                st.success(f"✅ Loaded {len(invoices_df)} invoice records")
                logger.info(f"Loaded invoice data: {len(invoices_df)} records")
            except Exception as e:
                st.error(f"❌ Error loading invoice data: {str(e)}")
                logger.error(f"Invoice data loading error: {e}")
        
        if payment_file is not None:
            try:
                payments_df = pd.read_csv(payment_file)
                st.success(f"✅ Loaded {len(payments_df)} payment records")
                logger.info(f"Loaded payment data: {len(payments_df)} records")
            except Exception as e:
                st.error(f"❌ Error loading payment data: {str(e)}")
                logger.error(f"Payment data loading error: {e}")
        
        # If no files uploaded, show empty state
        if invoices_df is None and payments_df is None:
            st.warning("📤 Please upload invoice and payment CSV files to begin analysis.")
            st.info("💡 **Expected CSV format:**\n"
                   "• **Invoices**: invoice_id, customer_id, amount, issue_date, due_date, status\n"
                   "• **Payments**: payment_id, invoice_id, amount, payment_date, method")
            return None, None, None
            
        # Normalize and merge data if both files are present
        if invoices_df is not None and payments_df is not None:
            try:
                merged_df = normalize_data(invoices_df, payments_df)
                logger.info(f"Successfully normalized and merged data: {len(merged_df)} records")
                return invoices_df, payments_df, merged_df
            except Exception as e:
                st.error(f"❌ Error processing data: {str(e)}")
                logger.error(f"Data processing error: {e}")
                return invoices_df, payments_df, None
        
        return invoices_df, payments_df, None
        
    except Exception as e:
        st.error(f"❌ Critical error in data loading: {str(e)}")
        logger.error(f"Critical data loading error: {e}")
        return None, None, None

def render_configuration_panel():
    """Render the simplified configuration panel."""
    st.markdown("### ⚙️ Configuration Panel")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Data Quality**")
        # This will be populated by the data quality module
        if 'quality_score' in st.session_state:
            score = st.session_state.quality_score
            st.metric("Quality Score", f"{score:.1%}")
        else:
            st.info("No Data")
        
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False, key="auto_refresh")
    
    with col2:
        st.markdown("**⚙️ System Settings**")
        date_range = st.selectbox("Analysis Period", ["Last 30 Days", "Last 90 Days", "Last 12 Months", "All Time"], index=1, key="date_range")
        batch_size = st.number_input("Batch Action Size", min_value=1, max_value=50, value=10, key="batch_size")
    
    return {
        'auto_refresh': auto_refresh,
        'date_range': date_range,
        'batch_size': batch_size,
        # Default values for removed configurations
        'dso_threshold': 45,
        'overdue_threshold': 50000,
        'currency': 'USD',
        'date_format': 'MM/DD/YYYY',
        'sim_mode': True
    }

def main():
    """Main application function."""
    # Header
    st.title("🏦 JPMorgan Smart Receivables Navigator")
    st.markdown("**AI-Powered Financial Intelligence Platform**")
    
    # User information setup - only show if not already set
    # This section is now handled within load_data
    
    # Check environment variables
    check_environment_variables()
    
    # Configuration panel
    config = render_configuration_panel()
    
    st.markdown("---")
    
    # Load data
    invoices_df, payments_df, merged_df = load_data()
    
    # If we have data, perform quality validation
    if merged_df is not None:
        try:
            quality_results = validate_data_quality(merged_df)
            quality_score = get_quality_score(quality_results)
            st.session_state.quality_score = quality_score
            
            if quality_score < 0.7:
                st.warning(f"⚠️ Data quality score is {quality_score:.1%}. Some analyses may be affected.")
        except Exception as e:
            st.warning(f"⚠️ Could not validate data quality: {str(e)}")
            logger.warning(f"Data quality validation error: {e}")
    
    # Tab structure - new order: KPI Cockpit, Smart Actions, AI Assistant, CFO Dashboard, Heat Map, What-If Simulator
    if merged_df is not None:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📈 KPI Cockpit", 
            "⚡ Smart Actions",
            "🤖 AI Assistant",
            "📊 CFO Dashboard", 
            "🔥 Heat Map", 
            "🎯 What-If Simulator",
            "📧 Quick Email"
        ])
        
        with tab1:
            render_kpi_dashboard(merged_df, config)
        
        with tab2:
            render_automation_center(merged_df, config)
        
        with tab3:
            st.markdown("### 🤖 AI Financial Assistant")
            st.markdown("Ask questions about your receivables data in natural language.")
            
            # Chat interface
            query = st.text_input(
                "Ask a question:",
                placeholder="e.g., 'What are our top overdue customers?' or 'Show me payment trends'",
                key="chat_query"
            )
            
            if st.button("Send", type="primary"):
                if query:
                    with st.spinner("🤔 Analyzing your data..."):
                        try:
                            df_dict = {
                                'invoices': invoices_df.to_dict() if invoices_df is not None else {},
                                'payments': payments_df.to_dict() if payments_df is not None else {},
                                'merged': merged_df.to_dict() if merged_df is not None else {}
                            }
                            response = ask_llm(query, df_dict)
                            st.markdown("**🤖 AI Response:**")
                            st.markdown(response)
                        except Exception as e:
                            st.error(f"❌ Error getting AI response: {str(e)}")
                            logger.error(f"AI assistant error: {e}")
        
        with tab4:
            render_cfo_panel(merged_df, config)
        
        with tab5:
            render_heatmap(merged_df, config)
        
        with tab6:
            render_what_if_simulator(merged_df, config)
            
        with tab7:
            st.subheader("📧 Quick AI Email")

            # Check if email is configured
            if not st.session_state.get("recipient"):
                st.warning("⚠️ Please configure your email address in the sidebar first.")
                st.info("Go to the sidebar and enter your email address, then click 'Save Information'.")
                st.stop()

            # Email configuration status - only test once
            if "email_tested" not in st.session_state:
                st.session_state["email_tested"] = False
            
            if not st.session_state["email_tested"]:
                with st.spinner("Testing email configuration..."):
                    from email_utils import test_email_connection
                    if test_email_connection():
                        st.session_state["email_tested"] = True
                        st.success("✅ Email configuration working")
                    else:
                        st.session_state["email_tested"] = True
                        st.error("❌ Email configuration failed. Check Gmail app password.")
                        st.info("The email drafting will still work, but sending may fail.")

            # Email interface
            st.markdown(f"**📧 Sending to:** {st.session_state['recipient']}")
            
            col1, col2 = st.columns([3,1])
            
            with col1:
                user_prompt = st.text_area(
                    "What should the email say?", 
                    placeholder="e.g., 'Send a reminder about overdue invoice INV001'",
                    key="email_prompt"
                )
            with col2:
                draft_button = st.button("💬 Generate Draft", type="primary", key="draft_button")
                
                if draft_button:
                    if not user_prompt.strip():
                        st.error("Please type something for the email.")
                    else:
                        with st.spinner("🤖 Generating email draft..."):
                            try:
                                draft = draft_email(user_prompt)
                                st.session_state["email_draft"] = f"{draft}\n\n{FIXED_LINE}"
                                st.success("✅ Draft generated successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error generating draft: {str(e)}")

            # Show draft if exists
            if "email_draft" in st.session_state:
                st.markdown("### 📝 Email Preview")
                edited_draft = st.text_area(
                    "Edit the draft if needed:", 
                    st.session_state["email_draft"], 
                    height=200,
                    key="email_editor"
                )
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    send_button = st.button("🚀 Send Email", type="primary", key="send_button")
                    if send_button:
                        with st.spinner("📧 Sending email..."):
                            try:
                                success = send_email(
                                    st.session_state["recipient"],
                                    "Your requested information",
                                    edited_draft
                                )
                                if success:
                                    st.success(f"✅ Email sent to {st.session_state['recipient']}")
                                    # Clear the draft after successful send
                                    del st.session_state["email_draft"]
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to send email. Check the logs above.")
                            except Exception as e:
                                st.error(f"❌ Error sending email: {str(e)}")
                
                with col2:
                    if st.button("🗑️ Clear Draft", key="clear_draft"):
                        if "email_draft" in st.session_state:
                            del st.session_state["email_draft"]
                        st.rerun()
                
                with col3:
                    if st.button("📝 New Draft", key="new_draft"):
                        if "email_draft" in st.session_state:
                            del st.session_state["email_draft"]
                        st.rerun()
    
    else:
        # Empty state with helpful information
        st.markdown("""
            <div style='text-align: center; padding: 60px 20px;'>
                <h3 style='color: #cccccc;'>📤 No Data Loaded</h3>
                <p style='color: #aaaaaa; font-size: 1.1rem;'>
                    Upload your invoice and payment CSV files to start analyzing your receivables.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Show sample data format
        with st.expander("📋 Sample Data Format", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Sample Invoice CSV:**")
                st.code("""invoice_id,customer_id,amount,issue_date,due_date,status
INV001,CUST001,15000,2024-01-15,2024-02-14,outstanding
INV002,CUST002,25000,2024-01-20,2024-02-19,paid
INV003,CUST001,8000,2024-02-01,2024-03-03,overdue""")
            
            with col2:
                st.markdown("**Sample Payment CSV:**")
                st.code("""payment_id,invoice_id,amount,payment_date,method
PAY001,INV002,25000,2024-02-15,bank_transfer
PAY002,INV001,5000,2024-02-20,check
PAY003,INV003,8000,2024-03-10,credit_card""")
    
    # Auto-refresh functionality
    if config.get('auto_refresh', False) and merged_df is not None:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
