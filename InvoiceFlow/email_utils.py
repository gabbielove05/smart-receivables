import os
import smtplib
from email.message import EmailMessage
import requests
import json
import streamlit as st
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get secrets safely
try:
    EMAIL = st.secrets["email"]["user"]
    APP_PWD = st.secrets["email"]["app_password"]
    OR_KEY = st.secrets["openrouter"]["key"]
except:
    # Fallback to environment variables
    EMAIL = os.getenv("EMAIL_USER", "GLoveEmailTest@gmail.com")
    APP_PWD = os.getenv("EMAIL_APP_PASSWORD", "shutzfeeqtpdbqnp")
    OR_KEY = os.getenv("OPENROUTER_API_KEY", "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")

FIXED_LINE = (
    "After reading this, copy and send this message to "
    "(insert whoever's email the customer is)"
)

def draft_email(prompt: str) -> str:
    """Use OpenRouter key to get an email draft."""
    try:
        headers = {
            "Authorization": f"Bearer {OR_KEY}",
            "Content-Type": "application/json"
        }
        body = {
            "model": "openai/gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": f"Write a professional email about: {prompt}"}
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        logger.info(f"Making OpenRouter API call for prompt: {prompt[:50]}...")
        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                          headers=headers, json=body, timeout=30)
        
        if r.status_code != 200:
            logger.error(f"OpenRouter API error {r.status_code}: {r.text}")
            return f"I apologize, but there was an error with the email service (HTTP {r.status_code}). Please try again."
        
        response = r.json()
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"].strip()
            logger.info("Email draft generated successfully")
            return content
        else:
            logger.error("Unexpected API response format")
            return "I apologize, but I couldn't generate an email draft at this time. Please try again."
            
    except requests.exceptions.Timeout:
        logger.error("OpenRouter API timeout")
        return "I apologize, but the email service is taking too long to respond. Please try again."
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenRouter API error: {e}")
        return "I apologize, but there was an error connecting to the email service. Please try again."
    except Exception as e:
        logger.error(f"Unexpected error in draft_email: {e}")
        return "I apologize, but there was an unexpected error. Please try again."

def send_email(to_addr: str, subj: str, body: str) -> bool:
    """Send via Gmail SMTP with app-password auth."""
    try:
        logger.info(f"Attempting to send email to: {to_addr}")
        
        # Validate inputs
        if not to_addr or not subj or not body:
            logger.error("Missing required email parameters")
            return False
            
        msg = EmailMessage()
        msg["From"] = EMAIL
        msg["To"] = to_addr
        msg["Subject"] = subj
        msg.set_content(body)

        logger.info("Connecting to Gmail SMTP...")
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            logger.info("Logging into Gmail...")
            smtp.login(EMAIL, APP_PWD)
            logger.info("Sending email...")
            smtp.send_message(msg)
            
        logger.info("Email sent successfully")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"Gmail authentication failed: {e}")
        st.error("❌ Email authentication failed. Please check the Gmail app password.")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error: {e}")
        st.error(f"❌ Email sending failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in send_email: {e}")
        st.error(f"❌ Unexpected error sending email: {str(e)}")
        return False

def test_email_connection() -> bool:
    """Test if email configuration is working."""
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(EMAIL, APP_PWD)
        return True
    except Exception as e:
        logger.error(f"Email connection test failed: {e}")
        return False
