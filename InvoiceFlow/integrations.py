"""
Communication and Integration Module
Handles SMTP email, SendGrid API, MS Teams webhooks, and phone call functionality.
"""

import os
import smtplib
import logging
import requests
import json
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, Any, Optional
import streamlit as st

logger = logging.getLogger(__name__)

# Email configuration from environment variables
SMTP_HOST = os.getenv('SMTP_HOST')
SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASS = os.getenv('SMTP_PASS')
FROM_EMAIL = os.getenv('FROM_EMAIL')
SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY')
MS_TEAMS_WEBHOOK = os.getenv('MS_TEAMS_WEBHOOK')

def send_email(to_email: str, subject: str, body: str, html_body: str = None) -> bool:
    """
    Send email using default email client with mailto link.
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        body: Plain text email body
        html_body: Optional HTML email body (not used with mailto)
        
    Returns:
        bool: True if mailto link generated successfully
    """
    try:
        return _create_mailto_link(to_email, subject, body)
            
    except Exception as e:
        logger.error(f"Email generation failed: {e}")
        return False

def _send_via_smtp(to_email: str, subject: str, body: str, html_body: str = None) -> bool:
    """Send email via SMTP."""
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = FROM_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Add plain text part
        msg.attach(MIMEText(body, 'plain'))
        
        # Add HTML part if provided
        if html_body:
            msg.attach(MIMEText(html_body, 'html'))
        
        # Send email
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        
        logger.info(f"SMTP email sent successfully to {to_email}")
        return True
        
    except Exception as e:
        logger.error(f"SMTP email failed: {e}")
        return False

def _send_via_sendgrid(to_email: str, subject: str, body: str, html_body: str = None) -> bool:
    """Send email via SendGrid API."""
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail, Email, To, Content
        
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        
        message = Mail(
            from_email=Email(FROM_EMAIL or "noreply@jpmorgan.com"),
            to_emails=To(to_email),
            subject=subject
        )
        
        if html_body:
            message.content = Content("text/html", html_body)
        else:
            message.content = Content("text/plain", body)
        
        response = sg.send(message)
        logger.info(f"SendGrid email sent successfully to {to_email}, status: {response.status_code}")
        return True
        
    except Exception as e:
        logger.error(f"SendGrid email failed: {e}")
        return False

def _create_mailto_link(to_email: str, subject: str, body: str) -> bool:
    """Create mailto link as fallback."""
    try:
        import urllib.parse
        
        mailto_url = f"mailto:{to_email}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
        
        st.markdown(f"""
            📧 **Email Draft Created**
            
            Click the link below to open your default email client:
            
            [📮 Send Email to {to_email}]({mailto_url})
            
            **Subject**: {subject}
            
            **Body**: {body[:100]}{'...' if len(body) > 100 else ''}
        """)
        
        logger.info(f"Mailto link created for {to_email}")
        return True
        
    except Exception as e:
        logger.error(f"Mailto link creation failed: {e}")
        return False

def call_client(phone: str, client_name: str = "Client") -> str:
    """
    Create a clickable phone call link.
    
    Args:
        phone: Phone number to call
        client_name: Name of the client for display
        
    Returns:
        HTML string with clickable phone link
    """
    try:
        # Clean phone number
        clean_phone = ''.join(filter(str.isdigit, phone))
        if len(clean_phone) >= 10:
            formatted_phone = f"+1-{clean_phone[-10:-7]}-{clean_phone[-7:-4]}-{clean_phone[-4:]}"
        else:
            formatted_phone = phone
        
        call_link = f'<a href="tel:{clean_phone}" style="color: #0066CC; text-decoration: none;">📞 Call {client_name} ({formatted_phone})</a>'
        
        logger.info(f"Phone call link created for {phone}")
        return call_link
        
    except Exception as e:
        logger.error(f"Phone link creation failed: {e}")
        return f"📞 Call {client_name}: {phone}"

def send_teams_alert(message: str, title: str = "Receivables Alert", priority: str = "normal") -> bool:
    """
    Send alert to Microsoft Teams via webhook.
    
    Args:
        message: Alert message
        title: Alert title
        priority: Alert priority (low, normal, high)
        
    Returns:
        bool: True if alert sent successfully
    """
    try:
        if not MS_TEAMS_WEBHOOK:
            logger.warning("MS Teams webhook not configured")
            return False
        
        # Color coding based on priority
        color_map = {
            "low": "00FF00",      # Green
            "normal": "FFD700",   # Gold
            "high": "FF0000"      # Red
        }
        
        color = color_map.get(priority.lower(), "FFD700")
        
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color,
            "title": title,
            "text": message,
            "sections": [
                {
                    "activityTitle": "JPMorgan Smart Receivables Navigator",
                    "activitySubtitle": f"Priority: {priority.capitalize()}",
                    "activityImage": "https://logo.clearbit.com/jpmorgan.com",
                    "facts": [
                        {
                            "name": "Timestamp",
                            "value": f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} UTC"
                        },
                        {
                            "name": "Source",
                            "value": "Smart Receivables Navigator"
                        }
                    ]
                }
            ],
            "potentialAction": [
                {
                    "@type": "OpenUri",
                    "name": "View Dashboard",
                    "targets": [
                        {
                            "os": "default",
                            "uri": "https://your-streamlit-app.com"
                        }
                    ]
                }
            ]
        }
        
        response = requests.post(
            MS_TEAMS_WEBHOOK,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"Teams alert sent successfully: {title}")
            return True
        else:
            logger.error(f"Teams alert failed with status {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Teams alert failed: {e}")
        return False

def send_collection_reminder(customer_info: Dict[str, Any]) -> Dict[str, bool]:
    """
    Send collection reminder via multiple channels.
    
    Args:
        customer_info: Dictionary with customer details
        
    Returns:
        Dictionary with success status for each channel
    """
    try:
        customer_name = customer_info.get('name', 'Customer')
        email = customer_info.get('email', '')
        phone = customer_info.get('phone', '')
        outstanding_amount = customer_info.get('outstanding_amount', 0)
        days_overdue = customer_info.get('days_overdue', 0)
        
        # Email reminder
        email_subject = f"Payment Reminder - Outstanding Balance ${outstanding_amount:,.2f}"
        email_body = f"""Dear {customer_name},
        
This is a friendly reminder that you have an outstanding balance of ${outstanding_amount:,.2f} 
that is {days_overdue} days overdue.

Please arrange for payment at your earliest convenience. If you have any questions or 
would like to discuss payment arrangements, please contact our accounts receivable team.

Thank you for your prompt attention to this matter.

Best regards,
JPMorgan Accounts Receivable Team
"""
        
        html_body = f"""
        <html>
        <body>
            <h2 style="color: #0066CC;">Payment Reminder</h2>
            <p>Dear {customer_name},</p>
            
            <p>This is a friendly reminder that you have an outstanding balance of 
            <strong>${outstanding_amount:,.2f}</strong> that is <strong>{days_overdue} days overdue</strong>.</p>
            
            <p>Please arrange for payment at your earliest convenience. If you have any questions or 
            would like to discuss payment arrangements, please contact our accounts receivable team.</p>
            
            <p>Thank you for your prompt attention to this matter.</p>
            
            <p>Best regards,<br>
            <strong>JPMorgan Accounts Receivable Team</strong></p>
        </body>
        </html>
        """
        
        # Send notifications
        results = {}
        
        # Email
        if email:
            results['email'] = send_email(email, email_subject, email_body, html_body)
        else:
            results['email'] = False
            logger.warning(f"No email address for customer {customer_name}")
        
        # Teams alert for high-value overdue accounts
        if outstanding_amount > 10000:
            teams_message = (f"High-value overdue account alert:\n"
                           f"Customer: {customer_name}\n"
                           f"Amount: ${outstanding_amount:,.2f}\n"
                           f"Days Overdue: {days_overdue}")
            results['teams'] = send_teams_alert(teams_message, "High-Value Overdue Alert", "high")
        else:
            results['teams'] = False
        
        logger.info(f"Collection reminder sent for {customer_name}: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Collection reminder failed: {e}")
        return {'email': False, 'teams': False}

def test_integrations() -> Dict[str, bool]:
    """Test all integration endpoints and return status."""
    results = {}
    
    # Test SMTP configuration
    if all([SMTP_HOST, SMTP_USER, SMTP_PASS]):
        results['smtp'] = True
        logger.info("SMTP configuration: OK")
    else:
        results['smtp'] = False
        logger.warning("SMTP configuration: Missing credentials")
    
    # Test SendGrid configuration
    if SENDGRID_API_KEY:
        results['sendgrid'] = True
        logger.info("SendGrid configuration: OK")
    else:
        results['sendgrid'] = False
        logger.warning("SendGrid configuration: Missing API key")
    
    # Test Teams webhook
    if MS_TEAMS_WEBHOOK:
        try:
            test_payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "title": "Integration Test",
                "text": "This is a test message from Smart Receivables Navigator"
            }
            response = requests.post(MS_TEAMS_WEBHOOK, json=test_payload, timeout=5)
            results['teams'] = response.status_code == 200
            logger.info(f"Teams webhook test: {'OK' if results['teams'] else 'Failed'}")
        except:
            results['teams'] = False
            logger.warning("Teams webhook test: Failed")
    else:
        results['teams'] = False
        logger.warning("Teams webhook: Not configured")
    
    return results
