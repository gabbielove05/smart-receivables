#!/usr/bin/env python3
"""
Generate Sample Data for InvoiceFlow
Creates realistic invoice and payment CSV files for demonstration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import os

# Initialize Faker for realistic data generation
fake = Faker()
Faker.seed(42)  # For reproducible data
np.random.seed(42)
random.seed(42)

def generate_sample_invoices(num_invoices=500):
    """Generate sample invoice data with realistic patterns."""
    print(f"Generating {num_invoices} sample invoices...")
    
    invoices = []
    
    # Define customer segments with different profiles
    customer_segments = {
        'Enterprise': {'count': 50, 'avg_amount': 75000, 'payment_reliability': 0.85},
        'Mid-Market': {'count': 150, 'avg_amount': 25000, 'payment_reliability': 0.75},
        'Small Business': {'count': 250, 'avg_amount': 8000, 'payment_reliability': 0.65},
        'Startup': {'count': 100, 'avg_amount': 15000, 'payment_reliability': 0.55}
    }
    
    # Generate customers for each segment
    customers = {}
    customer_id = 1
    
    for segment, details in customer_segments.items():
        for _ in range(details['count']):
            customers[f"CUST_{customer_id:04d}"] = {
                'segment': segment,
                'company_name': fake.company(),
                'avg_amount': details['avg_amount'],
                'payment_reliability': details['payment_reliability']
            }
            customer_id += 1
    
    # Generate invoices
    for i in range(num_invoices):
        customer_id = random.choice(list(customers.keys()))
        customer = customers[customer_id]
        
        # Generate realistic amounts based on customer segment
        base_amount = customer['avg_amount']
        amount = np.random.lognormal(
            mean=np.log(base_amount),
            sigma=0.5
        )
        amount = max(1000, min(amount, base_amount * 3))  # Keep within reasonable bounds
        
        # Generate dates
        issue_date = fake.date_between(start_date='-120d', end_date='today')
        due_date = issue_date + timedelta(days=random.choice([15, 30, 45, 60]))
        
        # Determine status based on due date and customer reliability
        days_since_due = (datetime.now().date() - due_date).days
        reliability = customer['payment_reliability']
        
        # Status logic - more realistic distribution
        if days_since_due < 0:
            # Not yet due
            status = 'outstanding'
        elif days_since_due <= 30:
            # Recently due - check reliability
            if random.random() < reliability * 1.2:  # Higher chance of payment when just due
                status = 'paid'
            else:
                status = 'outstanding'
        elif days_since_due <= 60:
            # Moderately overdue
            if random.random() < reliability * 0.8:
                status = 'paid'
            elif random.random() < 0.7:
                status = 'overdue'
            else:
                status = 'outstanding'
        else:
            # Very overdue
            if random.random() < reliability * 0.4:
                status = 'paid'
            else:
                status = 'overdue'
        
        # Create invoice record
        invoice = {
            'invoice_id': f"INV_{i+1:06d}",
            'customer_id': customer_id,
            'customer_name': customer['company_name'],
            'amount': round(amount, 2),
            'issue_date': issue_date.strftime('%Y-%m-%d'),
            'due_date': due_date.strftime('%Y-%m-%d'),
            'status': status,
            'payment_terms': random.choice(['Net 15', 'Net 30', 'Net 45', 'Net 60']),
            'customer_segment': customer['segment'],
            'currency': 'USD',
            'description': fake.catch_phrase(),
            'account_manager': fake.name()
        }
        
        invoices.append(invoice)
    
    # Convert to DataFrame
    df = pd.DataFrame(invoices)
    
    # Display status distribution
    status_counts = df['status'].value_counts()
    print(f"Invoice Status Distribution:")
    for status, count in status_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {status.title()}: {count} ({percentage:.1f}%)")
    
    return df

def generate_sample_payments(invoices_df, payment_rate=0.7):
    """Generate sample payment data based on invoices."""
    print(f"Generating payments for {len(invoices_df)} invoices...")
    
    payments = []
    payment_id = 1
    
    # Generate payments for paid invoices and some outstanding ones
    for _, invoice in invoices_df.iterrows():
        # Always generate payments for paid invoices
        if invoice['status'] == 'paid':
            should_pay = True
        # Generate partial payments for some outstanding/overdue invoices
        elif invoice['status'] in ['outstanding', 'overdue']:
            should_pay = random.random() < 0.3  # 30% chance of partial payment
        else:
            should_pay = False
        
        if should_pay:
            # Determine payment amount
            if invoice['status'] == 'paid':
                # Full payment for paid invoices
                payment_amount = invoice['amount']
                payment_count = 1 if random.random() < 0.8 else random.randint(2, 3)  # Sometimes split payments
            else:
                # Partial payment for outstanding/overdue
                payment_amount = invoice['amount'] * random.uniform(0.3, 0.8)
                payment_count = 1
            
            # Generate payment(s)
            remaining_amount = payment_amount
            due_date = pd.to_datetime(invoice['due_date']).date()
            
            for payment_num in range(payment_count):
                if payment_count == 1:
                    amount = remaining_amount
                else:
                    if payment_num == payment_count - 1:
                        amount = remaining_amount
                    else:
                        amount = remaining_amount * random.uniform(0.3, 0.7)
                        remaining_amount -= amount
                
                # Payment date logic
                if invoice['status'] == 'paid':
                    # Paid invoices - payment before or slightly after due date
                    days_offset = random.randint(-10, 30)
                else:
                    # Outstanding - recent partial payment
                    days_offset = random.randint(0, 10)
                
                payment_date = due_date + timedelta(days=days_offset)
                # Ensure payment date is not in the future
                payment_date = min(payment_date, datetime.now().date())
                
                payment = {
                    'payment_id': f"PAY_{payment_id:06d}",
                    'invoice_id': invoice['invoice_id'],
                    'customer_id': invoice['customer_id'],
                    'amount': round(amount, 2),
                    'payment_date': payment_date.strftime('%Y-%m-%d'),
                    'payment_method': random.choice(['ACH', 'Wire Transfer', 'Check', 'Credit Card']),
                    'reference_number': f"REF_{payment_id:08d}",
                    'notes': random.choice(['', 'Partial payment', 'Early payment discount', 'Regular payment'])
                }
                
                payments.append(payment)
                payment_id += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(payments)
    
    if not df.empty:
        print(f"Generated {len(df)} payments")
        
        # Show payment method distribution
        method_counts = df['payment_method'].value_counts()
        print("Payment Method Distribution:")
        for method, count in method_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {method}: {count} ({percentage:.1f}%)")
    else:
        print("No payments generated")
    
    return df

def save_sample_data():
    """Generate and save sample data files."""
    print("=" * 50)
    print("InvoiceFlow Sample Data Generator")
    print("=" * 50)
    
    # Generate data
    invoices_df = generate_sample_invoices(500)
    payments_df = generate_sample_payments(invoices_df)
    
    # Save to CSV files
    invoice_file = 'sample_invoices.csv'
    payment_file = 'sample_payments.csv'
    
    invoices_df.to_csv(invoice_file, index=False)
    payments_df.to_csv(payment_file, index=False)
    
    print(f"\n✅ Sample data saved:")
    print(f"📄 Invoices: {invoice_file} ({len(invoices_df)} records)")
    print(f"💰 Payments: {payment_file} ({len(payments_df)} records)")
    
    # Summary statistics
    print(f"\n📊 Data Summary:")
    print(f"Total Invoice Amount: ${invoices_df['amount'].sum():,.2f}")
    print(f"Total Payments: ${payments_df['amount'].sum():,.2f}")
    
    outstanding_amount = invoices_df[invoices_df['status'] == 'outstanding']['amount'].sum()
    overdue_amount = invoices_df[invoices_df['status'] == 'overdue']['amount'].sum()
    paid_amount = invoices_df[invoices_df['status'] == 'paid']['amount'].sum()
    
    print(f"Outstanding Amount: ${outstanding_amount:,.2f}")
    print(f"Overdue Amount: ${overdue_amount:,.2f}")
    print(f"Paid Amount: ${paid_amount:,.2f}")
    
    print(f"\n✨ Data generation complete!")
    return invoices_df, payments_df

if __name__ == "__main__":
    # Check if data files already exist
    if os.path.exists('sample_invoices.csv') and os.path.exists('sample_payments.csv'):
        response = input("Sample data files already exist. Regenerate? (y/N): ")
        if response.lower() != 'y':
            print("Using existing sample data files.")
            exit(0)
    
    # Generate sample data
    save_sample_data()
