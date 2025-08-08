#!/usr/bin/env python3
"""
Test script for email functionality
"""

import os
import sys
sys.path.append('.')

from email_utils import draft_email, send_email, test_email_connection

def test_email_functionality():
    print("🧪 Testing Email Functionality")
    print("=" * 40)
    
    # Test 1: Email connection
    print("1. Testing Gmail connection...")
    if test_email_connection():
        print("✅ Gmail connection successful")
    else:
        print("❌ Gmail connection failed")
        return False
    
    # Test 2: Draft generation
    print("\n2. Testing email draft generation...")
    try:
        draft = draft_email("Write a professional reminder about an overdue invoice")
        print("✅ Draft generated successfully")
        print(f"Preview: {draft[:100]}...")
    except Exception as e:
        print(f"❌ Draft generation failed: {e}")
        return False
    
    # Test 3: Email sending (optional)
    print("\n3. Testing email sending...")
    test_email = input("Enter a test email address (or press Enter to skip): ").strip()
    
    if test_email:
        try:
            success = send_email(
                test_email,
                "Test Email from InvoiceFlow",
                f"Hello,\n\nThis is a test email from InvoiceFlow.\n\nGenerated draft:\n{draft[:200]}...\n\nBest regards,\nInvoiceFlow Team"
            )
            if success:
                print("✅ Test email sent successfully")
            else:
                print("❌ Test email failed to send")
        except Exception as e:
            print(f"❌ Email sending error: {e}")
    else:
        print("⏭️ Skipping email send test")
    
    print("\n🎉 Email functionality test complete!")
    return True

if __name__ == "__main__":
    test_email_functionality()
