#!/usr/bin/env python3
"""
Test the fixed email functionality
"""

import streamlit as st
import sys
sys.path.append('.')

def test_email_flow():
    print("🧪 Testing Fixed Email Flow")
    print("=" * 40)
    
    # Simulate session state
    st.session_state = {}
    
    # Test 1: Email recipient setup
    print("1. Testing email recipient setup...")
    st.session_state["recipient"] = "test@example.com"
    print(f"   ✅ Email set: {st.session_state['recipient']}")
    
    # Test 2: Draft generation
    print("\n2. Testing draft generation...")
    from email_utils import draft_email
    try:
        draft = draft_email("Write a professional reminder about overdue invoice")
        print("   ✅ Draft generated successfully")
        print(f"   Preview: {draft[:100]}...")
    except Exception as e:
        print(f"   ❌ Draft generation failed: {e}")
        return False
    
    # Test 3: Email sending
    print("\n3. Testing email sending...")
    from email_utils import send_email
    try:
        success = send_email(
            st.session_state["recipient"],
            "Test Email",
            "This is a test email from InvoiceFlow"
        )
        if success:
            print("   ✅ Email sent successfully")
        else:
            print("   ❌ Email sending failed")
    except Exception as e:
        print(f"   ❌ Email sending error: {e}")
    
    print("\n🎉 Email flow test complete!")
    return True

if __name__ == "__main__":
    test_email_flow()
