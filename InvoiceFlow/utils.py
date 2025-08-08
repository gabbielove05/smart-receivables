"""
Utility Functions for Smart Receivables Navigator
Contains data processing, normalization, and helper functions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import hashlib
import os

logger = logging.getLogger(__name__)

# CSS loading disabled - using default Streamlit styling
def load_css() -> None:
    """CSS loading disabled - using default Streamlit styling."""
    pass

def load_fallback_css() -> None:
    """CSS loading disabled - using default Streamlit styling."""
    pass

def normalize_data(invoices_df: pd.DataFrame, payments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and merge invoice and payment data.
    
    Args:
        invoices_df: Invoice DataFrame
        payments_df: Payment DataFrame
        
    Returns:
        Merged and normalized DataFrame
    """
    try:
        logger.info("Starting data normalization process")
        
        # Validate required columns
        required_invoice_cols = ['invoice_id']
        required_payment_cols = ['invoice_id'] if 'invoice_id' in payments_df.columns else ['payment_id']
        
        missing_invoice_cols = [col for col in required_invoice_cols if col not in invoices_df.columns]
        if missing_invoice_cols:
            logger.warning(f"Missing required invoice columns: {missing_invoice_cols}")
        
        # Clean and standardize data
        invoices_clean = clean_invoice_data(invoices_df)
        payments_clean = clean_payment_data(payments_df)
        
        # Merge data on invoice_id
        if 'invoice_id' in payments_clean.columns:
            merged_df = merge_invoice_payment_data(invoices_clean, payments_clean)
        else:
            # If no direct relationship, create synthetic merge
            merged_df = create_synthetic_merge(invoices_clean, payments_clean)
        
        # Add derived fields
        merged_df = add_derived_fields(merged_df)
        
        # Final data validation
        merged_df = validate_merged_data(merged_df)
        
        logger.info(f"Data normalization completed: {len(merged_df)} records")
        return merged_df
        
    except Exception as e:
        logger.error(f"Data normalization failed: {e}")
        # Return invoices data as fallback
        return clean_invoice_data(invoices_df) if not invoices_df.empty else pd.DataFrame()

def clean_invoice_data(invoices_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize invoice data."""
    try:
        df = invoices_df.copy()
        
        # Standardize column names
        column_mapping = {
            'Invoice_ID': 'invoice_id',
            'Customer_ID': 'customer_id',
            'Amount': 'amount',
            'Issue_Date': 'issue_date',
            'Due_Date': 'due_date',
            'Status': 'status',
            'invoice_amount': 'amount',
            'customer': 'customer_id',
            'date_issued': 'issue_date',
            'date_due': 'due_date'
        }
        
        # Apply column mapping
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]
                df = df.drop(columns=[old_name])
        
        # Clean amount field
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df['amount'] = df['amount'].fillna(0).abs()  # Ensure positive amounts
        
        # Clean dates
        date_columns = ['issue_date', 'due_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Clean status field
        if 'status' in df.columns:
            df['status'] = df['status'].astype(str).str.lower().str.strip()
            # Standardize status values
            status_mapping = {
                'open': 'outstanding',
                'pending': 'outstanding',
                'closed': 'paid',
                'completed': 'paid',
                'past due': 'overdue',
                'overdue': 'overdue',
                'outstanding': 'outstanding',
                'paid': 'paid'
            }
            df['status'] = df['status'].map(status_mapping).fillna('outstanding')
        
        # Generate invoice_id if missing
        if 'invoice_id' not in df.columns:
            df['invoice_id'] = df.index.map(lambda x: f'INV_{x+1:06d}')
        
        # Generate customer_id if missing
        if 'customer_id' not in df.columns:
            df['customer_id'] = df.index.map(lambda x: f'CUST_{(x % 100)+1:03d}')
        
        # Remove duplicates
        if 'invoice_id' in df.columns:
            df = df.drop_duplicates(subset=['invoice_id'], keep='first')
        
        logger.info(f"Cleaned invoice data: {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning invoice data: {e}")
        return invoices_df

def clean_payment_data(payments_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize payment data."""
    try:
        df = payments_df.copy()
        
        # Standardize column names
        column_mapping = {
            'Payment_ID': 'payment_id',
            'Invoice_ID': 'invoice_id',
            'Amount': 'amount',
            'Payment_Date': 'payment_date',
            'Method': 'payment_method',
            'payment_amount': 'amount',
            'date_paid': 'payment_date',
            'pay_method': 'payment_method'
        }
        
        # Apply column mapping
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]
                df = df.drop(columns=[old_name])
        
        # Clean amount field
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df['amount'] = df['amount'].fillna(0).abs()
        
        # Clean payment date
        if 'payment_date' in df.columns:
            df['payment_date'] = pd.to_datetime(df['payment_date'], errors='coerce')
        
        # Clean payment method
        if 'payment_method' in df.columns:
            df['payment_method'] = df['payment_method'].astype(str).str.lower().str.strip()
        
        # Generate payment_id if missing
        if 'payment_id' not in df.columns:
            df['payment_id'] = df.index.map(lambda x: f'PAY_{x+1:06d}')
        
        # Remove duplicates
        if 'payment_id' in df.columns:
            df = df.drop_duplicates(subset=['payment_id'], keep='first')
        
        logger.info(f"Cleaned payment data: {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning payment data: {e}")
        return payments_df

def merge_invoice_payment_data(invoices_df: pd.DataFrame, payments_df: pd.DataFrame) -> pd.DataFrame:
    """Merge invoice and payment data on invoice_id."""
    try:
        # Aggregate payments by invoice_id
        payment_agg = payments_df.groupby('invoice_id').agg({
            'amount': 'sum',
            'payment_date': 'max',  # Latest payment date
            'payment_method': 'first',
            'payment_id': 'count'  # Number of payments
        }).rename(columns={
            'amount': 'payment_amount',
            'payment_date': 'latest_payment_date',
            'payment_id': 'payment_count'
        })
        
        # Merge with invoices
        merged_df = invoices_df.merge(payment_agg, left_on='invoice_id', right_index=True, how='left')
        
        # Fill missing payment data
        merged_df['payment_amount'] = merged_df['payment_amount'].fillna(0)
        merged_df['payment_count'] = merged_df['payment_count'].fillna(0).astype(int)
        
        # Update status based on payment data
        merged_df = update_status_from_payments(merged_df)
        
        logger.info(f"Merged invoice-payment data: {len(merged_df)} records")
        return merged_df
        
    except Exception as e:
        logger.error(f"Error merging invoice-payment data: {e}")
        return invoices_df

def create_synthetic_merge(invoices_df: pd.DataFrame, payments_df: pd.DataFrame) -> pd.DataFrame:
    """Create synthetic merge when no direct relationship exists."""
    try:
        # Add payment information as aggregated columns
        merged_df = invoices_df.copy()
        
        if not payments_df.empty:
            # Add summary payment information
            total_payments = payments_df['amount'].sum() if 'amount' in payments_df.columns else 0
            avg_payment = payments_df['amount'].mean() if 'amount' in payments_df.columns else 0
            payment_count = len(payments_df)
            
            # Distribute payment info proportionally (simplified approach)
            if 'amount' in merged_df.columns and total_payments > 0:
                invoice_weights = merged_df['amount'] / merged_df['amount'].sum()
                merged_df['payment_amount'] = invoice_weights * total_payments * 0.7  # 70% collection rate assumption
            else:
                merged_df['payment_amount'] = 0
            
            merged_df['payment_count'] = np.random.poisson(payment_count / len(merged_df), len(merged_df))
            merged_df['latest_payment_date'] = pd.NaT
            merged_df['payment_method'] = 'unknown'
        
        logger.info(f"Created synthetic merge: {len(merged_df)} records")
        return merged_df
        
    except Exception as e:
        logger.error(f"Error creating synthetic merge: {e}")
        return invoices_df

def update_status_from_payments(df: pd.DataFrame) -> pd.DataFrame:
    """Update invoice status based on payment information."""
    try:
        if 'amount' in df.columns and 'payment_amount' in df.columns:
            # Calculate payment ratio
            df['payment_ratio'] = df['payment_amount'] / df['amount'].replace(0, 1)
            
            # Update status based on payment ratio and due date
            current_date = datetime.now()
            
            def determine_status(row):
                if row['payment_ratio'] >= 0.95:  # 95% paid
                    return 'paid'
                elif 'due_date' in df.columns and pd.notnull(row.get('due_date')):
                    if row['due_date'] < current_date:
                        return 'overdue'
                    else:
                        return 'outstanding'
                else:
                    return row.get('status', 'outstanding')
            
            df['status'] = df.apply(determine_status, axis=1)
        
        return df
        
    except Exception as e:
        logger.error(f"Error updating status from payments: {e}")
        return df

def add_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived fields for analysis."""
    try:
        current_date = datetime.now()
        
        # Days since issue
        if 'issue_date' in df.columns:
            df['days_since_issue'] = (current_date - df['issue_date']).dt.days
            df['days_since_issue'] = df['days_since_issue'].fillna(30).clip(lower=0)
        
        # Days overdue
        if 'due_date' in df.columns:
            df['days_overdue'] = (current_date - df['due_date']).dt.days
            df['days_overdue'] = df['days_overdue'].fillna(0).clip(lower=0)
        else:
            df['days_overdue'] = 0
        
        # Payment terms (days from issue to due)
        if 'issue_date' in df.columns and 'due_date' in df.columns:
            df['payment_terms'] = (df['due_date'] - df['issue_date']).dt.days
            df['payment_terms'] = df['payment_terms'].fillna(30).clip(lower=0)
        else:
            df['payment_terms'] = 30
        
        # Outstanding amount
        if 'amount' in df.columns and 'payment_amount' in df.columns:
            df['outstanding_amount'] = (df['amount'] - df['payment_amount']).clip(lower=0)
        elif 'amount' in df.columns:
            # Estimate based on status
            status_multiplier = df['status'].map({
                'paid': 0.0,
                'outstanding': 1.0,
                'overdue': 1.0
            }).fillna(1.0)
            df['outstanding_amount'] = df['amount'] * status_multiplier
        
        # Risk score (simplified)
        df['risk_score'] = calculate_risk_scores(df)
        
        # Age bucket
        df['age_bucket'] = pd.cut(
            df.get('days_overdue', 0),
            bins=[-1, 0, 30, 60, 90, float('inf')],
            labels=['Current', '1-30 days', '31-60 days', '61-90 days', '90+ days']
        )
        
        logger.info("Added derived fields successfully")
        return df
        
    except Exception as e:
        logger.error(f"Error adding derived fields: {e}")
        return df

def calculate_risk_scores(df: pd.DataFrame) -> pd.Series:
    """Calculate risk scores for each record."""
    try:
        risk_scores = pd.Series(0.0, index=df.index)
        
        # Amount-based risk (higher amounts = higher risk if unpaid)
        if 'outstanding_amount' in df.columns:
            amount_percentiles = df['outstanding_amount'].quantile([0.5, 0.8, 0.95])
            amount_risk = pd.cut(
                df['outstanding_amount'],
                bins=[-1, amount_percentiles[0.5], amount_percentiles[0.8], amount_percentiles[0.95], float('inf')],
                labels=[0, 1, 2, 3]
            ).astype(float)
            risk_scores += amount_risk * 25
        
        # Age-based risk
        if 'days_overdue' in df.columns:
            age_risk = pd.cut(
                df['days_overdue'],
                bins=[-1, 0, 30, 60, float('inf')],
                labels=[0, 1, 2, 3]
            ).astype(float)
            risk_scores += age_risk * 25
        
        # Status-based risk
        if 'status' in df.columns:
            status_risk = df['status'].map({
                'paid': 0,
                'outstanding': 1,
                'overdue': 3
            }).fillna(1)
            risk_scores += status_risk * 25
        
        # Customer history risk (simplified)
        if 'customer_id' in df.columns:
            customer_counts = df['customer_id'].value_counts()
            customer_risk = df['customer_id'].map(lambda x: min(2, max(0, 3 - customer_counts.get(x, 1))))
            risk_scores += customer_risk * 25
        
        # Normalize to 0-100 scale
        risk_scores = risk_scores.clip(0, 100)
        
        return risk_scores
        
    except Exception as e:
        logger.error(f"Error calculating risk scores: {e}")
        return pd.Series(50.0, index=df.index)

def validate_merged_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform final validation on merged data."""
    try:
        # Remove records with no amount
        if 'amount' in df.columns:
            initial_count = len(df)
            df = df[df['amount'] > 0]
            removed_count = initial_count - len(df)
            if removed_count > 0:
                logger.warning(f"Removed {removed_count} records with zero/negative amounts")
        
        # Ensure required columns exist
        required_columns = ['invoice_id', 'customer_id', 'amount', 'status']
        for col in required_columns:
            if col not in df.columns:
                if col == 'invoice_id':
                    df[col] = df.index.map(lambda x: f'INV_{x+1:06d}')
                elif col == 'customer_id':
                    df[col] = df.index.map(lambda x: f'CUST_{(x % 50)+1:03d}')
                elif col == 'amount':
                    df[col] = 1000.0  # Default amount
                elif col == 'status':
                    df[col] = 'outstanding'
        
        # Sort by amount descending for better analysis
        if 'amount' in df.columns:
            df = df.sort_values('amount', ascending=False).reset_index(drop=True)
        
        logger.info(f"Data validation completed: {len(df)} valid records")
        return df
        
    except Exception as e:
        logger.error(f"Error validating merged data: {e}")
        return df

def generate_sample_data(n_invoices: int = 100, n_payments: int = 80) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate sample data for testing purposes."""
    try:
        from faker import Faker
        fake = Faker()
        np.random.seed(42)  # For reproducible results
        
        # Generate invoice data
        invoices = []
        for i in range(n_invoices):
            issue_date = fake.date_between(start_date='-1y', end_date='today')
            due_date = issue_date + timedelta(days=np.random.choice([15, 30, 45, 60]))
            
            invoice = {
                'invoice_id': f'INV{i+1:06d}',
                'customer_id': f'CUST{np.random.randint(1, 21):03d}',  # 20 unique customers
                'amount': np.random.lognormal(8, 1.5),  # Log-normal distribution for realistic amounts
                'issue_date': issue_date,
                'due_date': due_date,
                'status': np.random.choice(['outstanding', 'paid', 'overdue'], p=[0.4, 0.5, 0.1])
            }
            invoices.append(invoice)
        
        invoices_df = pd.DataFrame(invoices)
        
        # Generate payment data
        payments = []
        paid_invoices = invoices_df[invoices_df['status'] == 'paid']['invoice_id'].tolist()
        
        # Add some payments for outstanding invoices too
        outstanding_invoices = invoices_df[invoices_df['status'] == 'outstanding']['invoice_id'].tolist()
        some_outstanding = np.random.choice(outstanding_invoices, size=min(10, len(outstanding_invoices)), replace=False)
        
        all_paid_invoices = paid_invoices + list(some_outstanding)
        
        for i, invoice_id in enumerate(all_paid_invoices[:n_payments]):
            invoice_amount = invoices_df[invoices_df['invoice_id'] == invoice_id]['amount'].iloc[0]
            payment_amount = invoice_amount * np.random.uniform(0.8, 1.0)  # Partial to full payment
            
            payment = {
                'payment_id': f'PAY{i+1:06d}',
                'invoice_id': invoice_id,
                'amount': payment_amount,
                'payment_date': fake.date_between(start_date='-6m', end_date='today'),
                'payment_method': np.random.choice(['bank_transfer', 'credit_card', 'check', 'wire'])
            }
            payments.append(payment)
        
        payments_df = pd.DataFrame(payments)
        
        logger.info(f"Generated sample data: {len(invoices_df)} invoices, {len(payments_df)} payments")
        return invoices_df, payments_df
        
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        # Return minimal sample data
        invoices_df = pd.DataFrame({
            'invoice_id': ['INV001', 'INV002', 'INV003'],
            'customer_id': ['CUST001', 'CUST002', 'CUST001'],
            'amount': [15000, 25000, 8000],
            'status': ['outstanding', 'paid', 'overdue']
        })
        payments_df = pd.DataFrame({
            'payment_id': ['PAY001'],
            'invoice_id': ['INV002'],
            'amount': [25000],
            'payment_date': [datetime.now() - timedelta(days=10)]
        })
        return invoices_df, payments_df

def format_currency(amount: float, currency: str = 'USD') -> str:
    """Format currency values consistently."""
    try:
        if pd.isna(amount) or amount == 0:
            return f'{currency} 0'
        
        if amount >= 1_000_000:
            return f'{currency} {amount/1_000_000:.1f}M'
        elif amount >= 1_000:
            return f'{currency} {amount/1_000:.1f}K'
        else:
            return f'{currency} {amount:.0f}'
    except:
        return f'{currency} 0'

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage values consistently."""
    try:
        if pd.isna(value):
            return '0.0%'
        return f'{value:.{decimals}f}%'
    except:
        return '0.0%'

def hash_string(text: str) -> str:
    """Generate consistent hash for a string."""
    try:
        return hashlib.md5(text.encode()).hexdigest()
    except:
        return hashlib.md5('default'.encode()).hexdigest()

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return numerator / denominator
    except:
        return default

def get_business_days_between(start_date: datetime, end_date: datetime) -> int:
    """Calculate business days between two dates."""
    try:
        return len(pd.bdate_range(start=start_date, end=end_date))
    except:
        return (end_date - start_date).days

def categorize_amount(amount: float, thresholds: Dict[str, float] = None) -> str:
    """Categorize amounts into ranges."""
    if thresholds is None:
        thresholds = {
            'Small': 10000,
            'Medium': 50000,
            'Large': 100000
        }
    
    try:
        if amount <= thresholds['Small']:
            return 'Small'
        elif amount <= thresholds['Medium']:
            return 'Medium'
        elif amount <= thresholds['Large']:
            return 'Large'
        else:
            return 'Enterprise'
    except:
        return 'Unknown'

def get_age_category(days: int) -> str:
    """Categorize age in days."""
    try:
        if days <= 0:
            return 'Current'
        elif days <= 30:
            return '1-30 days'
        elif days <= 60:
            return '31-60 days'
        elif days <= 90:
            return '61-90 days'
        else:
            return '90+ days'
    except:
        return 'Unknown'

def export_to_csv(df: pd.DataFrame, filename: str) -> bool:
    """Export DataFrame to CSV file."""
    try:
        df.to_csv(filename, index=False)
        logger.info(f"Data exported to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return False

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations."""
    import re
    try:
        # Remove invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Limit length
        sanitized = sanitized[:100]
        # Ensure it has an extension
        if '.' not in sanitized:
            sanitized += '.csv'
        return sanitized
    except:
        return 'export.csv'

