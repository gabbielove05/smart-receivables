"""
JPMorgan Smart Receivables Navigator - Main Application
A comprehensive financial dashboard for receivables management with AI-powered insights.
"""

import streamlit as st
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="JPMorgan Smart Receivables Navigator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
            
            # Show data preview
            st.markdown("### 📊 Data Preview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Invoice Data:**")
                st.dataframe(invoices_df.head(), use_container_width=True)
            
            with col2:
                st.markdown("**Payment Data:**")
                st.dataframe(payments_df.head(), use_container_width=True)
            
            # Basic analytics
            st.markdown("### 📈 Basic Analytics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Invoices", len(invoices_df))
            
            with col2:
                st.metric("Total Payments", len(payments_df))
            
            with col3:
                if 'amount' in invoices_df.columns:
                    total_amount = invoices_df['amount'].sum()
                    st.metric("Total Invoice Amount", f"${total_amount:,.2f}")
            
            # AI Chat section
            st.markdown("### 🤖 AI Assistant")
            st.markdown("Ask questions about your receivables data:")
            
            user_question = st.text_input(
                "Your question:",
                placeholder="e.g., What are my top overdue invoices?",
                key="ai_question"
            )
            
            if st.button("🤖 Ask AI", key="ask_ai"):
                if user_question:
                    with st.spinner("🤖 AI is analyzing your data..."):
                        # Simple AI response for now
                        st.info("AI analysis feature is being loaded. This will provide insights about your receivables data.")
                else:
                    st.error("Please enter a question.")
            
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
