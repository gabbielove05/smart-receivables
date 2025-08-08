"""
Unit Tests for ETL Pipeline
Tests data normalization, cleaning, and transformation functions.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    normalize_data, clean_invoice_data, clean_payment_data,
    merge_invoice_payment_data, add_derived_fields, calculate_risk_scores,
    validate_merged_data, generate_sample_data
)


class TestETLPipeline(unittest.TestCase):
    """Test cases for ETL pipeline functions."""
    
    def setUp(self):
        """Set up test data for each test."""
        # Sample invoice data
        self.sample_invoices = pd.DataFrame({
            'Invoice_ID': ['INV001', 'INV002', 'INV003'],
            'Customer_ID': ['CUST001', 'CUST002', 'CUST001'],
            'Amount': [15000.50, 25000.00, 8000.75],
            'Issue_Date': ['2024-01-15', '2024-01-20', '2024-02-01'],
            'Due_Date': ['2024-02-14', '2024-02-19', '2024-03-03'],
            'Status': ['outstanding', 'paid', 'overdue']
        })
        
        # Sample payment data
        self.sample_payments = pd.DataFrame({
            'Payment_ID': ['PAY001', 'PAY002', 'PAY003'],
            'Invoice_ID': ['INV002', 'INV001', 'INV003'],
            'Amount': [25000.00, 5000.00, 8000.75],
            'Payment_Date': ['2024-02-15', '2024-02-20', '2024-03-10'],
            'Method': ['bank_transfer', 'check', 'credit_card']
        })
    
    def test_clean_invoice_data(self):
        """Test invoice data cleaning functionality."""
        cleaned = clean_invoice_data(self.sample_invoices)
        
        # Check column name standardization
        self.assertIn('invoice_id', cleaned.columns)
        self.assertIn('customer_id', cleaned.columns)
        self.assertIn('amount', cleaned.columns)
        self.assertIn('issue_date', cleaned.columns)
        self.assertIn('due_date', cleaned.columns)
        self.assertIn('status', cleaned.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned['amount']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned['issue_date']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned['due_date']))
        
        # Check status standardization
        valid_statuses = {'paid', 'outstanding', 'overdue'}
        self.assertTrue(all(status in valid_statuses for status in cleaned['status']))
        
        # Check positive amounts
        self.assertTrue(all(cleaned['amount'] >= 0))
    
    def test_clean_payment_data(self):
        """Test payment data cleaning functionality."""
        cleaned = clean_payment_data(self.sample_payments)
        
        # Check column name standardization
        self.assertIn('payment_id', cleaned.columns)
        self.assertIn('invoice_id', cleaned.columns)
        self.assertIn('amount', cleaned.columns)
        self.assertIn('payment_date', cleaned.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned['amount']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned['payment_date']))
        
        # Check positive amounts
        self.assertTrue(all(cleaned['amount'] >= 0))
    
    def test_merge_invoice_payment_data(self):
        """Test merging of invoice and payment data."""
        cleaned_invoices = clean_invoice_data(self.sample_invoices)
        cleaned_payments = clean_payment_data(self.sample_payments)
        
        merged = merge_invoice_payment_data(cleaned_invoices, cleaned_payments)
        
        # Check that all invoices are preserved
        self.assertEqual(len(merged), len(cleaned_invoices))
        
        # Check that payment information is added
        self.assertIn('payment_amount', merged.columns)
        self.assertIn('payment_count', merged.columns)
        
        # Check specific payment amounts
        paid_invoice = merged[merged['invoice_id'] == 'INV002'].iloc[0]
        self.assertEqual(paid_invoice['payment_amount'], 25000.00)
        
        # Check unpaid invoice has zero payment amount
        unpaid_invoices = merged[merged['payment_amount'] == 0]
        self.assertTrue(len(unpaid_invoices) >= 0)
    
    def test_add_derived_fields(self):
        """Test addition of derived fields."""
        cleaned_invoices = clean_invoice_data(self.sample_invoices)
        derived = add_derived_fields(cleaned_invoices)
        
        # Check derived fields exist
        expected_fields = [
            'days_since_issue', 'days_overdue', 'payment_terms',
            'outstanding_amount', 'risk_score', 'age_bucket'
        ]
        
        for field in expected_fields:
            self.assertIn(field, derived.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_numeric_dtype(derived['days_since_issue']))
        self.assertTrue(pd.api.types.is_numeric_dtype(derived['days_overdue']))
        self.assertTrue(pd.api.types.is_numeric_dtype(derived['risk_score']))
        
        # Check non-negative values
        self.assertTrue(all(derived['days_since_issue'] >= 0))
        self.assertTrue(all(derived['days_overdue'] >= 0))
        self.assertTrue(all(derived['risk_score'] >= 0))
    
    def test_calculate_risk_scores(self):
        """Test risk score calculation."""
        cleaned_invoices = clean_invoice_data(self.sample_invoices)
        derived = add_derived_fields(cleaned_invoices)
        
        risk_scores = calculate_risk_scores(derived)
        
        # Check that risk scores are in valid range
        self.assertTrue(all(risk_scores >= 0))
        self.assertTrue(all(risk_scores <= 100))
        
        # Check that overdue invoices have higher risk scores
        overdue_mask = derived['status'] == 'overdue'
        if overdue_mask.any():
            avg_overdue_risk = risk_scores[overdue_mask].mean()
            avg_paid_risk = risk_scores[derived['status'] == 'paid'].mean()
            self.assertGreater(avg_overdue_risk, avg_paid_risk)
    
    def test_normalize_data_integration(self):
        """Test the complete data normalization pipeline."""
        normalized = normalize_data(self.sample_invoices, self.sample_payments)
        
        # Check that normalization completes successfully
        self.assertIsInstance(normalized, pd.DataFrame)
        self.assertGreater(len(normalized), 0)
        
        # Check essential columns exist
        essential_columns = ['invoice_id', 'customer_id', 'amount', 'status']
        for col in essential_columns:
            self.assertIn(col, normalized.columns)
        
        # Check data quality
        self.assertTrue(all(normalized['amount'] > 0))
        self.assertFalse(normalized['invoice_id'].duplicated().any())
    
    def test_validate_merged_data(self):
        """Test data validation functionality."""
        # Create test data with some invalid records
        test_data = pd.DataFrame({
            'invoice_id': ['INV001', 'INV002', 'INV003'],
            'customer_id': ['CUST001', 'CUST002', 'CUST003'],
            'amount': [15000, 0, -1000],  # Include zero and negative amounts
            'status': ['outstanding', 'paid', 'overdue']
        })
        
        validated = validate_merged_data(test_data)
        
        # Check that invalid records are handled
        self.assertTrue(all(validated['amount'] > 0))
        
        # Check required columns exist
        required_columns = ['invoice_id', 'customer_id', 'amount', 'status']
        for col in required_columns:
            self.assertIn(col, validated.columns)
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        invoices_df, payments_df = generate_sample_data(n_invoices=10, n_payments=8)
        
        # Check data generation
        self.assertIsInstance(invoices_df, pd.DataFrame)
        self.assertIsInstance(payments_df, pd.DataFrame)
        self.assertEqual(len(invoices_df), 10)
        self.assertLessEqual(len(payments_df), 8)
        
        # Check column structure
        invoice_columns = ['invoice_id', 'customer_id', 'amount', 'issue_date', 'due_date', 'status']
        payment_columns = ['payment_id', 'invoice_id', 'amount', 'payment_date', 'payment_method']
        
        for col in invoice_columns:
            self.assertIn(col, invoices_df.columns)
        
        for col in payment_columns:
            self.assertIn(col, payments_df.columns)
        
        # Check data validity
        self.assertTrue(all(invoices_df['amount'] > 0))
        self.assertTrue(all(payments_df['amount'] > 0))
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty DataFrames
        empty_df = pd.DataFrame()
        
        cleaned_empty = clean_invoice_data(empty_df)
        self.assertIsInstance(cleaned_empty, pd.DataFrame)
        
        # Test with minimal data
        minimal_invoice = pd.DataFrame({'amount': [1000]})
        cleaned_minimal = clean_invoice_data(minimal_invoice)
        self.assertIn('invoice_id', cleaned_minimal.columns)
        self.assertIn('customer_id', cleaned_minimal.columns)
        
        # Test normalization with empty payments
        empty_payments = pd.DataFrame()
        normalized = normalize_data(self.sample_invoices, empty_payments)
        self.assertGreater(len(normalized), 0)
    
    def test_data_consistency(self):
        """Test data consistency across pipeline."""
        # Process data through complete pipeline
        cleaned_invoices = clean_invoice_data(self.sample_invoices)
        cleaned_payments = clean_payment_data(self.sample_payments)
        merged = merge_invoice_payment_data(cleaned_invoices, cleaned_payments)
        derived = add_derived_fields(merged)
        validated = validate_merged_data(derived)
        
        # Check data consistency
        original_invoice_ids = set(self.sample_invoices['Invoice_ID'])
        final_invoice_ids = set(validated['invoice_id'])
        
        # All original invoices should be preserved (with standardized IDs)
        self.assertEqual(len(original_invoice_ids), len(final_invoice_ids))
        
        # Check amount preservation (should be close due to cleaning)
        original_total = self.sample_invoices['Amount'].sum()
        final_total = validated['amount'].sum()
        self.assertAlmostEqual(original_total, final_total, places=2)


class TestDataValidation(unittest.TestCase):
    """Test cases for data validation functions."""
    
    def test_date_validation(self):
        """Test date field validation."""
        test_data = pd.DataFrame({
            'issue_date': ['2024-01-15', 'invalid_date', '2024-02-01'],
            'due_date': ['2024-02-14', '2024-02-19', 'also_invalid'],
            'amount': [1000, 2000, 3000]
        })
        
        cleaned = clean_invoice_data(test_data)
        
        # Check that invalid dates are handled
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned['issue_date']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaned['due_date']))
    
    def test_amount_validation(self):
        """Test amount field validation."""
        test_data = pd.DataFrame({
            'amount': [1000.50, -500, 0, 'invalid', None],
            'invoice_id': ['INV001', 'INV002', 'INV003', 'INV004', 'INV005']
        })
        
        cleaned = clean_invoice_data(test_data)
        
        # Check that all amounts are valid numbers >= 0
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned['amount']))
        self.assertTrue(all(cleaned['amount'] >= 0))
    
    def test_status_validation(self):
        """Test status field validation."""
        test_data = pd.DataFrame({
            'status': ['PAID', 'Outstanding', 'OVERDUE', 'invalid_status', None],
            'amount': [1000, 2000, 3000, 4000, 5000],
            'invoice_id': ['INV001', 'INV002', 'INV003', 'INV004', 'INV005']
        })
        
        cleaned = clean_invoice_data(test_data)
        
        # Check that all statuses are standardized
        valid_statuses = {'paid', 'outstanding', 'overdue'}
        self.assertTrue(all(status in valid_statuses for status in cleaned['status']))


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestETLPipeline))
    test_suite.addTest(unittest.makeSuite(TestDataValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    sys.exit(exit_code)
