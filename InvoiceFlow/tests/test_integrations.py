"""
Unit Tests for Integration Components
Tests communication integrations, webhook payloads, and external service connections.
"""

import unittest
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations import (
    send_email, call_client, send_teams_alert,
    send_collection_reminder, test_integrations,
    _send_via_smtp, _send_via_sendgrid, _create_mailto_link
)


class TestEmailIntegration(unittest.TestCase):
    """Test cases for email integration functionality."""
    
    def setUp(self):
        """Set up test data for each test."""
        self.test_email = "test@example.com"
        self.test_subject = "Test Payment Reminder"
        self.test_body = "This is a test payment reminder email."
        self.test_html_body = "<html><body><h1>Test Payment Reminder</h1></body></html>"
    
    @patch.dict(os.environ, {
        'SMTP_HOST': 'smtp.test.com',
        'SMTP_PORT': '587',
        'SMTP_USER': 'test@jpmorgan.com',
        'SMTP_PASS': 'testpass',
        'FROM_EMAIL': 'noreply@jpmorgan.com'
    })
    @patch('integrations.smtplib.SMTP')
    def test_smtp_email_success(self, mock_smtp):
        """Test successful SMTP email sending."""
        # Mock SMTP server
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Test email sending
        result = send_email(self.test_email, self.test_subject, self.test_body)
        
        # Verify SMTP was called
        mock_smtp.assert_called_once()
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.send_message.assert_called_once()
        
        # Verify success
        self.assertTrue(result)
    
    @patch.dict(os.environ, {
        'SENDGRID_API_KEY': 'test_sendgrid_key',
        'FROM_EMAIL': 'noreply@jpmorgan.com'
    })
    @patch('integrations.SendGridAPIClient')
    def test_sendgrid_email_success(self, mock_sendgrid):
        """Test successful SendGrid email sending."""
        # Mock SendGrid client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.send.return_value = mock_response
        mock_sendgrid.return_value = mock_client
        
        # Clear SMTP env vars to force SendGrid usage
        with patch.dict(os.environ, {}, clear=True):
            os.environ['SENDGRID_API_KEY'] = 'test_sendgrid_key'
            os.environ['FROM_EMAIL'] = 'noreply@jpmorgan.com'
            
            result = send_email(self.test_email, self.test_subject, self.test_body, self.test_html_body)
        
        # Verify SendGrid was called
        mock_sendgrid.assert_called_once()
        mock_client.send.assert_called_once()
        
        # Verify success
        self.assertTrue(result)
    
    @patch('streamlit.markdown')
    def test_mailto_fallback(self, mock_st_markdown):
        """Test mailto link fallback when no email service is configured."""
        # Clear all email environment variables
        with patch.dict(os.environ, {}, clear=True):
            result = send_email(self.test_email, self.test_subject, self.test_body)
        
        # Verify mailto link was created
        mock_st_markdown.assert_called_once()
        call_args = mock_st_markdown.call_args[0][0]
        self.assertIn("mailto:", call_args)
        self.assertIn(self.test_email, call_args)
        
        # Verify success (mailto creation should succeed)
        self.assertTrue(result)
    
    def test_email_validation(self):
        """Test email input validation."""
        # Test with invalid inputs
        result = send_email("", self.test_subject, self.test_body)
        self.assertIsInstance(result, bool)
        
        result = send_email(self.test_email, "", self.test_body)
        self.assertIsInstance(result, bool)
        
        result = send_email(self.test_email, self.test_subject, "")
        self.assertIsInstance(result, bool)


class TestTeamsIntegration(unittest.TestCase):
    """Test cases for Microsoft Teams webhook integration."""
    
    def setUp(self):
        """Set up test data for each test."""
        self.test_message = "High-value overdue account alert"
        self.test_title = "Receivables Alert"
        self.test_webhook_url = "https://outlook.office.com/webhook/test"
    
    @patch('integrations.requests.post')
    @patch.dict(os.environ, {'MS_TEAMS_WEBHOOK': 'https://outlook.office.com/webhook/test'})
    def test_teams_alert_success(self, mock_post):
        """Test successful Teams webhook alert."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Send Teams alert
        result = send_teams_alert(self.test_message, self.test_title, "high")
        
        # Verify webhook was called
        mock_post.assert_called_once()
        
        # Verify payload structure
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        
        self.assertEqual(payload['@type'], 'MessageCard')
        self.assertEqual(payload['title'], self.test_title)
        self.assertEqual(payload['text'], self.test_message)
        self.assertEqual(payload['themeColor'], 'FF0000')  # Red for high priority
        
        # Verify success
        self.assertTrue(result)
    
    @patch('integrations.requests.post')
    @patch.dict(os.environ, {'MS_TEAMS_WEBHOOK': 'https://outlook.office.com/webhook/test'})
    def test_teams_alert_failure(self, mock_post):
        """Test Teams webhook alert failure handling."""
        # Mock failed response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response
        
        # Send Teams alert
        result = send_teams_alert(self.test_message, self.test_title, "normal")
        
        # Verify failure
        self.assertFalse(result)
    
    def test_teams_alert_no_webhook(self):
        """Test Teams alert when webhook is not configured."""
        # Clear webhook environment variable
        with patch.dict(os.environ, {}, clear=True):
            result = send_teams_alert(self.test_message, self.test_title)
        
        # Should return False when no webhook configured
        self.assertFalse(result)
    
    def test_teams_payload_structure(self):
        """Test Teams webhook payload structure for different priorities."""
        test_cases = [
            ("low", "00FF00"),    # Green
            ("normal", "FFD700"), # Gold  
            ("high", "FF0000")    # Red
        ]
        
        with patch('integrations.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            with patch.dict(os.environ, {'MS_TEAMS_WEBHOOK': 'https://test.webhook.com'}):
                for priority, expected_color in test_cases:
                    send_teams_alert(self.test_message, self.test_title, priority)
                    
                    # Get the last call's payload
                    call_args = mock_post.call_args
                    payload = call_args[1]['json']
                    
                    # Verify color mapping
                    self.assertEqual(payload['themeColor'], expected_color)
                    
                    # Verify required fields
                    self.assertIn('@type', payload)
                    self.assertIn('@context', payload)
                    self.assertIn('title', payload)
                    self.assertIn('text', payload)
                    self.assertIn('sections', payload)
                    self.assertIn('potentialAction', payload)


class TestPhoneIntegration(unittest.TestCase):
    """Test cases for phone call integration."""
    
    def test_call_client_formatting(self):
        """Test phone number formatting and link generation."""
        test_cases = [
            ("+1-555-1234", "John Doe"),
            ("555-555-5555", "Jane Smith"),
            ("15551234567", "Bob Johnson"),
            ("(555) 123-4567", "Alice Brown")
        ]
        
        for phone, name in test_cases:
            result = call_client(phone, name)
            
            # Verify HTML link is generated
            self.assertIsInstance(result, str)
            self.assertIn('href="tel:', result)
            self.assertIn(name, result)
            
            # Verify phone number is included
            clean_phone = ''.join(filter(str.isdigit, phone))
            if len(clean_phone) >= 10:
                self.assertIn(clean_phone, result)
    
    def test_call_client_edge_cases(self):
        """Test phone call functionality with edge cases."""
        # Test with invalid phone number
        result = call_client("invalid", "Test Client")
        self.assertIsInstance(result, str)
        self.assertIn("Test Client", result)
        
        # Test with empty phone number
        result = call_client("", "Empty Phone")
        self.assertIsInstance(result, str)
        self.assertIn("Empty Phone", result)


class TestCollectionReminder(unittest.TestCase):
    """Test cases for collection reminder functionality."""
    
    def setUp(self):
        """Set up test customer data."""
        self.test_customer = {
            'name': 'Test Corporation',
            'email': 'accounts@testcorp.com',
            'phone': '+1-555-1234',
            'outstanding_amount': 25000.50,
            'days_overdue': 45
        }
    
    @patch('integrations.send_email')
    @patch('integrations.send_teams_alert')
    def test_collection_reminder_success(self, mock_teams, mock_email):
        """Test successful collection reminder sending."""
        # Mock successful email and teams alert
        mock_email.return_value = True
        mock_teams.return_value = True
        
        # Send collection reminder
        result = send_collection_reminder(self.test_customer)
        
        # Verify email was sent
        mock_email.assert_called_once()
        email_args = mock_email.call_args
        self.assertEqual(email_args[0][0], self.test_customer['email'])
        self.assertIn('Payment Reminder', email_args[0][1])
        
        # Verify Teams alert for high-value account
        mock_teams.assert_called_once()
        teams_args = mock_teams.call_args
        self.assertIn('High-value overdue account', teams_args[0][0])
        
        # Verify results
        self.assertTrue(result['email'])
        self.assertTrue(result['teams'])
    
    @patch('integrations.send_email')
    @patch('integrations.send_teams_alert')
    def test_collection_reminder_low_value(self, mock_teams, mock_email):
        """Test collection reminder for low-value account."""
        # Low-value customer
        low_value_customer = self.test_customer.copy()
        low_value_customer['outstanding_amount'] = 5000
        
        mock_email.return_value = True
        
        # Send collection reminder
        result = send_collection_reminder(low_value_customer)
        
        # Verify email was sent but no Teams alert for low value
        mock_email.assert_called_once()
        mock_teams.assert_not_called()
        
        # Verify results
        self.assertTrue(result['email'])
        self.assertFalse(result['teams'])
    
    def test_collection_reminder_missing_email(self):
        """Test collection reminder with missing email."""
        # Customer without email
        customer_no_email = self.test_customer.copy()
        del customer_no_email['email']
        
        with patch('integrations.send_email') as mock_email:
            result = send_collection_reminder(customer_no_email)
            
            # Email should not be attempted
            mock_email.assert_not_called()
            self.assertFalse(result['email'])


class TestIntegrationTesting(unittest.TestCase):
    """Test cases for integration testing functionality."""
    
    @patch.dict(os.environ, {
        'SMTP_HOST': 'smtp.test.com',
        'SMTP_USER': 'test@jpmorgan.com', 
        'SMTP_PASS': 'testpass',
        'SENDGRID_API_KEY': 'sg.test_key',
        'MS_TEAMS_WEBHOOK': 'https://test.webhook.com'
    })
    def test_integrations_all_configured(self):
        """Test integration testing when all services are configured."""
        with patch('integrations.requests.post') as mock_post:
            # Mock successful Teams webhook test
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            results = test_integrations()
            
            # Verify all integrations are detected as configured
            self.assertTrue(results['smtp'])
            self.assertTrue(results['sendgrid'])
            self.assertTrue(results['teams'])
    
    def test_integrations_none_configured(self):
        """Test integration testing when no services are configured."""
        # Clear all environment variables
        with patch.dict(os.environ, {}, clear=True):
            results = test_integrations()
            
            # Verify all integrations are detected as not configured
            self.assertFalse(results['smtp'])
            self.assertFalse(results['sendgrid'])
            self.assertFalse(results['teams'])
    
    @patch.dict(os.environ, {'MS_TEAMS_WEBHOOK': 'https://test.webhook.com'})
    def test_teams_webhook_connectivity(self):
        """Test Teams webhook connectivity testing."""
        with patch('integrations.requests.post') as mock_post:
            # Test successful connection
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            results = test_integrations()
            
            # Verify webhook was tested
            mock_post.assert_called_once()
            self.assertTrue(results['teams'])
            
            # Test failed connection
            mock_post.reset_mock()
            mock_post.side_effect = Exception("Connection failed")
            
            results = test_integrations()
            self.assertFalse(results['teams'])


class TestWebhookPayloads(unittest.TestCase):
    """Test cases for webhook payload validation."""
    
    def test_teams_webhook_payload_structure(self):
        """Test Teams webhook payload structure compliance."""
        test_message = "Test alert message"
        test_title = "Test Alert"
        
        # Mock the requests.post to capture payload
        with patch('integrations.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            with patch.dict(os.environ, {'MS_TEAMS_WEBHOOK': 'https://test.webhook.com'}):
                send_teams_alert(test_message, test_title, "high")
            
            # Extract payload
            call_args = mock_post.call_args
            payload = call_args[1]['json']
            
            # Validate required Microsoft Teams fields
            required_fields = ['@type', '@context', 'title', 'text', 'themeColor']
            for field in required_fields:
                self.assertIn(field, payload)
            
            # Validate payload structure
            self.assertEqual(payload['@type'], 'MessageCard')
            self.assertEqual(payload['title'], test_title)
            self.assertEqual(payload['text'], test_message)
            
            # Validate sections structure
            self.assertIn('sections', payload)
            self.assertIsInstance(payload['sections'], list)
            
            if payload['sections']:
                section = payload['sections'][0]
                self.assertIn('activityTitle', section)
                self.assertIn('facts', section)
                self.assertIsInstance(section['facts'], list)
            
            # Validate potential actions
            self.assertIn('potentialAction', payload)
            self.assertIsInstance(payload['potentialAction'], list)
    
    def test_teams_payload_serialization(self):
        """Test that Teams webhook payloads are properly JSON serializable."""
        test_message = "Test message with special characters: üñíçødé & <tags>"
        
        with patch('integrations.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            with patch.dict(os.environ, {'MS_TEAMS_WEBHOOK': 'https://test.webhook.com'}):
                # This should not raise a JSON serialization error
                result = send_teams_alert(test_message, "Test Title")
                
                # Verify the payload can be JSON serialized
                call_args = mock_post.call_args
                payload = call_args[1]['json']
                
                # This should not raise an exception
                json_string = json.dumps(payload)
                self.assertIsInstance(json_string, str)
                
                # Verify special characters are preserved
                parsed_payload = json.loads(json_string)
                self.assertEqual(parsed_payload['text'], test_message)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestEmailIntegration,
        TestTeamsIntegration, 
        TestPhoneIntegration,
        TestCollectionReminder,
        TestIntegrationTesting,
        TestWebhookPayloads
    ]
    
    for test_class in test_classes:
        test_suite.addTest(unittest.makeSuite(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Integration Tests Summary")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Print failed tests details
    if result.failures:
        print(f"\nFailures:")
        for test, trace in result.failures:
            print(f"  - {test}: {trace.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, trace in result.errors:
            print(f"  - {test}: {trace.split('Error:')[-1].strip()}")
    
    print(f"{'='*60}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    sys.exit(exit_code)
