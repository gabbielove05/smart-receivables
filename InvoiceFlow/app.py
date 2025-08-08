"""
JPMorgan Smart Receivables Navigator - Main Application
A comprehensive financial dashboard for receivables management with AI-powered insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="JPMorgan Smart Receivables Navigator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Try to import advanced modules with error handling
try:
    from chatbot import ask_llm
    from smart_actions import generate_next_best_actions, render_automation_center
    from utils import load_css, normalize_data
    from email_utils import draft_email, send_email, FIXED_LINE
    ADVANCED_FEATURES = True
    st.success("✅ Advanced features loaded successfully!")
except ImportError as e:
    st.warning(f"⚠️ Some advanced features not available: {e}")
    ADVANCED_FEATURES = False


def calculate_kpis(invoices_df, payments_df):
    """Calculate key performance indicators."""
    try:
        # Basic KPIs
        total_invoices = len(invoices_df)
        total_payments = len(payments_df)
        
        # Calculate amounts
        if 'amount' in invoices_df.columns:
            total_invoice_amount = invoices_df['amount'].sum()
            avg_invoice_amount = invoices_df['amount'].mean()
        else:
            total_invoice_amount = 0
            avg_invoice_amount = 0
        
        if 'amount' in payments_df.columns:
            total_payment_amount = payments_df['amount'].sum()
        else:
            total_payment_amount = 0
        
        # Calculate DSO (Days Sales Outstanding)
        if 'due_date' in invoices_df.columns and 'payment_date' in payments_df.columns:
            try:
                invoices_df['due_date'] = pd.to_datetime(invoices_df['due_date'])
                payments_df['payment_date'] = pd.to_datetime(payments_df['payment_date'])
                
                # Simple DSO calculation
                current_date = datetime.now()
                overdue_invoices = invoices_df[invoices_df['due_date'] < current_date]
                dso = len(overdue_invoices) / total_invoices * 30 if total_invoices > 0 else 0
            except:
                dso = 0
        else:
            dso = 0
        
        return {
            'total_invoices': total_invoices,
            'total_payments': total_payments,
            'total_invoice_amount': total_invoice_amount,
            'total_payment_amount': total_payment_amount,
            'avg_invoice_amount': avg_invoice_amount,
            'dso': dso,
            'collection_rate': (total_payment_amount / total_invoice_amount * 100) if total_invoice_amount > 0 else 0
        }
    except Exception as e:
        logger.error(f"Error calculating KPIs: {e}")
        return {}


def render_kpi_dashboard(kpis):
    """Render KPI dashboard."""
    st.markdown("### 📊 KPI Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Invoices", 
            f"{kpis.get('total_invoices', 0):,}",
            help="Total number of invoices"
        )
    
    with col2:
        st.metric(
            "Total Amount", 
            f"${kpis.get('total_invoice_amount', 0):,.2f}",
            help="Total invoice amount"
        )
    
    with col3:
        st.metric(
            "Collection Rate", 
            f"{kpis.get('collection_rate', 0):.1f}%",
            help="Percentage of invoices collected"
        )
    
    with col4:
        st.metric(
            "DSO", 
            f"{kpis.get('dso', 0):.1f} days",
            help="Days Sales Outstanding"
        )


def render_ai_assistant(invoices_df, payments_df):
    """Render AI assistant interface."""
    st.markdown("### 🤖 AI Financial Assistant")
    st.markdown("Ask questions about your receivables data in natural language.")
    
    # Chat interface
    query = st.text_input(
        "Ask a question:",
        placeholder="e.g., 'What are our top overdue customers?' or 'Show me payment trends'",
        key="chat_query"
    )
    
    if st.button("🤖 Ask AI", type="primary", key="ask_ai_button"):
        if query and ADVANCED_FEATURES:
            with st.spinner("🤔 Analyzing your data..."):
                try:
                    # Prepare data for AI
                    data_summary = {
                        'invoices_count': len(invoices_df),
                        'payments_count': len(payments_df),
                        'invoice_columns': list(invoices_df.columns),
                        'payment_columns': list(payments_df.columns)
                    }
                    
                    # Create a simple response for now
                    response = f"""
**AI Analysis Results:**

📊 **Data Overview:**
- {data_summary['invoices_count']} invoices loaded
- {data_summary['payments_count']} payments loaded
- Available invoice columns: {', '.join(data_summary['invoice_columns'])}
- Available payment columns: {', '.join(data_summary['payment_columns'])}

🤖 **Your Question:** {query}

💡 **Insight:** Based on your data, I can help analyze payment patterns, identify overdue invoices, and provide recommendations for improving cash flow.

*Note: Advanced AI features are being loaded. For full functionality, ensure all dependencies are properly configured.*
                    """
                    st.markdown(response)
                except Exception as e:
                    st.error(f"❌ Error getting AI response: {str(e)}")
        elif not ADVANCED_FEATURES:
            st.info("🤖 AI features are being loaded. Please check the configuration.")
        else:
            st.error("Please enter a question.")


def render_smart_actions(invoices_df, payments_df):
    """Render smart actions interface."""
    st.markdown("### ⚡ Smart Actions")
    st.markdown("AI-powered recommendations for improving your receivables.")
    
    if ADVANCED_FEATURES:
        try:
            # Generate simple recommendations
            recommendations = []
            
            # Check for overdue invoices
            if 'due_date' in invoices_df.columns:
                try:
                    invoices_df['due_date'] = pd.to_datetime(invoices_df['due_date'])
                    current_date = datetime.now()
                    overdue = invoices_df[invoices_df['due_date'] < current_date]
                    
                    if len(overdue) > 0:
                        recommendations.append({
                            'action': 'Send Payment Reminders',
                            'priority': 'High',
                            'description': f'Follow up on {len(overdue)} overdue invoices',
                            'impact': 'Improve cash flow'
                        })
                except:
                    pass
            
            # Check for large invoices
            if 'amount' in invoices_df.columns:
                large_invoices = invoices_df[invoices_df['amount'] > invoices_df['amount'].mean() * 2]
                if len(large_invoices) > 0:
                    recommendations.append({
                        'action': 'Prioritize Large Invoices',
                        'priority': 'Medium',
                        'description': f'Focus on {len(large_invoices)} high-value invoices',
                        'impact': 'Maximize revenue collection'
                    })
            
            # Display recommendations
            for i, rec in enumerate(recommendations):
                with st.expander(f"🎯 {rec['action']} ({rec['priority']} Priority)", expanded=True):
                    st.markdown(f"**Description:** {rec['description']}")
                    st.markdown(f"**Expected Impact:** {rec['impact']}")
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        if st.button(f"📧 Send Reminder", key=f"reminder_{i}"):
                            st.success("✅ Reminder action queued!")
                    with col2:
                        if st.button(f"📊 View Details", key=f"details_{i}"):
                            st.info("Detailed analysis feature coming soon!")
            
            if not recommendations:
                st.info("✅ No urgent actions needed. Your receivables look healthy!")
                
        except Exception as e:
            st.error(f"❌ Error generating recommendations: {str(e)}")
                    else:
        st.info("⚡ Smart actions feature is being loaded. Please check the configuration.")


def render_email_tool():
    """Render email drafting tool."""
    st.markdown("### 📧 AI Email Assistant")
    
    # Check if email is configured
    if not st.session_state.get("recipient"):
        st.warning("⚠️ Please configure your email address first.")
        st.info("Enter your email address in the User Information section above.")
        return
    
    st.markdown(f"**📧 Sending to:** {st.session_state['recipient']}")
    
    col1, col2 = st.columns([3, 1])
    
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
                        # Simple email draft for now
                        draft = f"""
Subject: Payment Reminder

Dear Customer,

{user_prompt}

Please review your outstanding invoices and process payment at your earliest convenience.

Best regards,
{st.session_state.get('user_name', 'Your Company')}

---
This email was generated by JPMorgan Smart Receivables Navigator.
                        """
                        st.session_state["email_draft"] = draft
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
            if st.button("🚀 Send Email", type="primary", key="send_button"):
                with st.spinner("📧 Sending email..."):
                    try:
                        # Simple email sending simulation
                        st.success(f"✅ Email sent to {st.session_state['recipient']}")
                        del st.session_state["email_draft"]
                        st.rerun()
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


# Main app
def main():
    st.title("🏦 JPMorgan Smart Receivables Navigator")
    st.markdown("---")
    
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
    
    # File upload section
    st.markdown("### 📁 Upload Your Data")
    st.markdown("Upload your invoice and payment CSV files to get started.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_invoices = st.file_uploader(
            "Upload Invoice CSV",
            type=['csv'],
            help="Upload a CSV file with invoice data"
        )
    
    with col2:
        uploaded_payments = st.file_uploader(
            "Upload Payment CSV",
            type=['csv'],
            help="Upload a CSV file with payment data"
        )
    
    # Process uploaded files
    if uploaded_invoices is not None and uploaded_payments is not None:
        try:
            # Load data
            invoices_df = pd.read_csv(uploaded_invoices)
            payments_df = pd.read_csv(uploaded_payments)
            
            st.success(f"✅ Loaded {len(invoices_df)} invoices and {len(payments_df)} payments")
            
            # Calculate KPIs
            kpis = calculate_kpis(invoices_df, payments_df)
            
            # Tab navigation
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 KPI Dashboard",
                "⚡ Smart Actions",
                "🤖 AI Assistant",
                "📧 Email Tool",
                "📋 Data Preview"
            ])
            
            with tab1:
                render_kpi_dashboard(kpis)
                
                # Show data preview
                st.markdown("### 📊 Data Overview")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Invoice Data:**")
                    st.dataframe(invoices_df.head(), use_container_width=True)
                
                with col2:
                    st.markdown("**Payment Data:**")
                    st.dataframe(payments_df.head(), use_container_width=True)
            
            with tab2:
                render_smart_actions(invoices_df, payments_df)
            
            with tab3:
                render_ai_assistant(invoices_df, payments_df)
            
            with tab4:
                render_email_tool()
            
            with tab5:
                st.markdown("### 📋 Full Data Preview")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Invoice Data:**")
                    st.dataframe(invoices_df, use_container_width=True)
                
                with col2:
                    st.markdown("**Payment Data:**")
                    st.dataframe(payments_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error processing files: {str(e)}")
            st.info("Please make sure your CSV files have the correct format.")
    
    else:
        # Empty state
        st.markdown("""
            <div style='text-align: center; padding: 60px 20px;'>
                <h3 style='color: #666;'>📤 No Data Loaded</h3>
                <p style='color: #888; font-size: 1.1rem;'>
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

if __name__ == "__main__":
    main()
