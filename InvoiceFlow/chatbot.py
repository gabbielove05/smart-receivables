"""
AI-Powered Chatbot for Smart Receivables Navigator
Uses OpenAI GPT-4o for natural language Q&A with finance FAQ fallback.
"""

import json
import os
import logging
from typing import Dict, Any
from openai import OpenAI

# Optional .env support
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)

# Initialize client (supports OpenRouter or OpenAI)
import streamlit as st

# Try to get API keys from Streamlit secrets first, then environment variables
try:
    _router_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    _openai_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
except Exception:
    _router_key = os.getenv("OPENROUTER_API_KEY")
    _openai_key = os.getenv("OPENAI_API_KEY")

api_key = None
base_url = None

if _router_key:
    api_key = _router_key
    base_url = "https://openrouter.ai/api/v1"
elif _openai_key:
    # If the provided OPENAI_API_KEY is actually an OpenRouter key (sk-or-...), route to OpenRouter
    if _openai_key.startswith("sk-or-") or _openai_key.startswith("sk-or-v1-"):
        api_key = _openai_key
        base_url = "https://openrouter.ai/api/v1"
    else:
        api_key = _openai_key
        base_url = None

openai_client = None
if api_key:
    try:
        openai_client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
        logger.info("LLM client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        openai_client = None
else:
    logger.warning("No API key found (OPENROUTER_API_KEY or OPENAI_API_KEY). Falling back to FAQ mode")

# Finance FAQ fallback database
FINANCE_FAQS = {
    "dso": {
        "question": "What is DSO and how is it calculated?",
        "answer": "Days Sales Outstanding (DSO) measures the average number of days it takes to collect receivables. It's calculated as: DSO = (Accounts Receivable / Total Credit Sales) × Number of Days in Period. A lower DSO indicates faster collection."
    },
    "aging": {
        "question": "How do you analyze accounts receivable aging?",
        "answer": "Accounts receivable aging categorizes outstanding invoices by time periods (0-30 days, 31-60 days, 61-90 days, 90+ days). This helps identify collection issues and prioritize follow-up activities."
    },
    "cash_flow": {
        "question": "How does receivables management impact cash flow?",
        "answer": "Effective receivables management directly impacts cash flow by reducing the time between sale and cash collection. Faster collections improve working capital and reduce financing costs."
    },
    "credit_risk": {
        "question": "What are credit risk indicators?",
        "answer": "Key credit risk indicators include: payment history, credit score changes, industry conditions, debt-to-equity ratios, and days payable outstanding (DPO) trends."
    },
    "collection_strategies": {
        "question": "What are effective collection strategies?",
        "answer": "Effective strategies include: automated payment reminders, early payment discounts, payment plans for distressed accounts, professional collection services, and credit limit adjustments."
    },
    "kpis": {
        "question": "What are key receivables KPIs to track?",
        "answer": "Key KPIs include: DSO, collection effectiveness index (CEI), bad debt ratio, average collection period, accounts receivable turnover, and percentage of receivables over 90 days."
    }
}

def get_faq_response(query: str) -> str:
    """Get response from FAQ database based on query keywords."""
    query_lower = query.lower()
    
    # Simple keyword matching for FAQ topics
    if any(word in query_lower for word in ["dso", "days sales outstanding", "collection time"]):
        return FINANCE_FAQS["dso"]["answer"]
    elif any(word in query_lower for word in ["aging", "overdue", "outstanding"]):
        return FINANCE_FAQS["aging"]["answer"]
    elif any(word in query_lower for word in ["cash flow", "working capital"]):
        return FINANCE_FAQS["cash_flow"]["answer"]
    elif any(word in query_lower for word in ["credit risk", "default", "risk assessment"]):
        return FINANCE_FAQS["credit_risk"]["answer"]
    elif any(word in query_lower for word in ["collection", "collect", "follow up"]):
        return FINANCE_FAQS["collection_strategies"]["answer"]
    elif any(word in query_lower for word in ["kpi", "metrics", "performance", "measure"]):
        return FINANCE_FAQS["kpis"]["answer"]
    else:
        return ("I can help you with receivables management topics including DSO calculation, "
                "aging analysis, cash flow impact, credit risk assessment, collection strategies, "
                "and key performance indicators. Please ask a specific question about these topics.")

def create_data_summary(df_dict: Dict[str, Any]) -> str:
    """Create a summary of the data for context in AI responses."""
    try:
        summary_parts = []
        
        if 'merged' in df_dict and df_dict['merged']:
            merged_data = df_dict['merged']
            if 'amount' in merged_data:
                amounts = list(merged_data['amount'].values()) if isinstance(merged_data['amount'], dict) else []
                if amounts:
                    total_amount = sum(amounts)
                    avg_amount = total_amount / len(amounts)
                    summary_parts.append(f"Total receivables: ${total_amount:,.2f}")
                    summary_parts.append(f"Average invoice amount: ${avg_amount:,.2f}")
                    summary_parts.append(f"Number of invoices: {len(amounts)}")
        
        if 'invoices' in df_dict and df_dict['invoices']:
            invoices_data = df_dict['invoices']
            if 'status' in invoices_data:
                statuses = list(invoices_data['status'].values()) if isinstance(invoices_data['status'], dict) else []
                if statuses:
                    outstanding = sum(1 for s in statuses if s in ['outstanding', 'overdue'])
                    paid = sum(1 for s in statuses if s == 'paid')
                    summary_parts.append(f"Outstanding invoices: {outstanding}")
                    summary_parts.append(f"Paid invoices: {paid}")
        
        return ". ".join(summary_parts) if summary_parts else "No data summary available"
        
    except Exception as e:
        logger.error(f"Error creating data summary: {e}")
        return "Data summary unavailable due to processing error"

def ask_llm(query: str, df_dict: Dict[str, Any]) -> str:
    """
    Main function to ask the LLM or fall back to FAQ system.
    
    Args:
        query: User's question
        df_dict: Dictionary containing DataFrames converted to dicts
        
    Returns:
        AI-generated or FAQ response
    """
    try:
        if not openai_client:
            logger.info("LLM client unavailable, using FAQ fallback")
            return (
                f"🔧 **FAQ Mode**: {get_faq_response(query)}\n\n"
                "💡 *Set OPENROUTER_API_KEY or OPENAI_API_KEY to enable AI-powered responses.*"
            )
        
        # Create context from data
        data_summary = create_data_summary(df_dict)
        
        # Construct prompt for financial analysis
        system_prompt = """You are a senior financial analyst specializing in accounts receivable management. 
        You provide clear, actionable insights about receivables data and answer questions about DSO, 
        cash flow, credit risk, collection strategies, and financial KPIs. 
        
        Be concise but thorough, use financial terminology appropriately, and always provide 
        practical recommendations when relevant. If you identify concerning patterns in the data, 
        highlight them and suggest corrective actions."""
        
        user_prompt = f"""Based on the following receivables data summary: {data_summary}

        Please answer this question: {query}
        
        Provide a professional financial analysis with specific insights and recommendations where appropriate."""
        
        # Using OpenRouter with smaller, cheaper models to avoid 402 errors
        model_name = "openai/gpt-4o-mini" if base_url else "gpt-4o-mini"
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=300,  # Reduced from 1000
            temperature=0.3
        )
        
        ai_response = response.choices[0].message.content
        logger.info(f"Generated AI response for query: {query[:50]}...")
        
        return f"🤖 **AI Analysis**: {ai_response}"
        
    except Exception as e:
        logger.error(f"Error in ask_llm: {e}")
        
        # Check for specific 402 error (insufficient credits)
        if "402" in str(e) or "Payment Required" in str(e):
            return (
                f"⚠️ **Token Limit Reached**: The AI request was too large for your current plan.\n\n"
                f"💡 **FAQ Response**: {get_faq_response(query)}\n\n"
                f"🔧 *Try asking a shorter question or contact support to increase your token limit.*"
            )
        
        # Fallback to FAQ system
        faq_response = get_faq_response(query)
        return (
            f"⚠️ **Fallback Mode**: {faq_response}\n\n"
            f"🔧 *AI service temporarily unavailable. Error: {str(e)}*"
        )

def generate_insights(df_dict: Dict[str, Any]) -> str:
    """Generate automated insights about the receivables data."""
    try:
        if not openai_client:
            return "AI insights unavailable. Please set OPENROUTER_API_KEY or OPENAI_API_KEY."
        
        data_summary = create_data_summary(df_dict)
        
        prompt = f"""Analyze this receivables data and provide 3-4 key insights with specific recommendations:

        Data Summary: {data_summary}
        
        Focus on:
        1. Cash flow implications
        2. Collection efficiency
        3. Risk assessment
        4. Actionable next steps
        
        Be specific and quantitative where possible. Format as bullet points."""
        
        # Use smaller, cheaper models to avoid 402 errors
        model_name = "openai/gpt-4o-mini" if base_url else "gpt-4o-mini"
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,  # Reduced from 500
            temperature=0.2
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        
        # Check for specific 402 error
        if "402" in str(e) or "Payment Required" in str(e):
            return (
                "⚠️ **Token Limit Reached**: Unable to generate AI insights due to token limit.\n\n"
                "💡 **Manual Analysis**: Review the KPI dashboard and action queue for key insights.\n\n"
                "🔧 *Contact support to increase your token limit for AI-powered insights.*"
            )
        
        return f"Unable to generate insights: {str(e)}"

def generate_executive_summary(df_dict: Dict[str, Any]) -> str:
    """Generate executive summary for CFO dashboard."""
    try:
        if not openai_client:
            return (
                "**Receivables Summary**: Data loaded successfully. "
                "AI-powered insights require OPENROUTER_API_KEY or OPENAI_API_KEY configuration. "
                "Review KPI dashboard for detailed metrics."
            )
        
        data_summary = create_data_summary(df_dict)
        
        prompt = f"""Create a 2-3 sentence executive summary for a CFO dashboard based on this receivables data:

        {data_summary}
        
        Focus on: overall financial health, immediate concerns, and strategic implications.
        Be concise and executive-level appropriate."""
        
        # Use smaller, cheaper models to avoid 402 errors
        model_name = "openai/gpt-4o-mini" if base_url else "gpt-4o-mini"
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,  # Reduced from 200
            temperature=0.2
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating executive summary: {e}")
        
        # Check for specific 402 error
        if "402" in str(e) or "Payment Required" in str(e):
            return (
                "⚠️ **Token Limit Reached**: Unable to generate AI executive summary.\n\n"
                "💡 **Manual Summary**: Review the KPI dashboard for key metrics and trends.\n\n"
                "🔧 *Contact support to increase your token limit for AI-powered summaries.*"
            )
        
        return "Executive summary unavailable due to AI service error."
