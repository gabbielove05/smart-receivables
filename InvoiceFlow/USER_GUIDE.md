# 🏦 InvoiceFlow Smart Receivables Navigator - User Guide

Welcome to the InvoiceFlow Smart Receivables Navigator! This comprehensive guide will help you get the most out of your accounts receivable management platform.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Application Overview](#application-overview)
3. [Smart Actions - The Complete Guide](#smart-actions---the-complete-guide)
4. [Email System](#email-system)
5. [Dashboard Features](#dashboard-features)
6. [Data Management](#data-management)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Tips](#advanced-tips)

---

## 🚀 Quick Start

### 1. Launch the Application

The easiest way to start is using our automated launcher:

```bash
cd InvoiceFlow
./run_app.sh
```

This script will:
- ✅ Set up a virtual environment
- ✅ Install all dependencies
- ✅ Generate sample data
- ✅ Launch the application

**Alternative Manual Launch:**
```bash
cd InvoiceFlow
source venv/bin/activate
streamlit run app.py
```

### 2. First Time Setup

When you first open the application:

1. **Navigate to the "Smart Actions" tab**
2. **Click "⚙️ Email Configuration"** to set up your email
3. **Enter your email address** where you want to receive draft emails
4. **Optionally configure SMTP** for automatic sending (or use mailto links)
5. **Start exploring your data!**

---

## 🏦 Application Overview

### Main Tabs

| Tab | Purpose | Key Features |
|-----|---------|--------------|
| **📊 Executive Dashboard** | High-level overview for leadership | KPIs, trends, executive summaries |
| **🔥 Risk Analysis** | Risk assessment and heatmaps | Interactive heatmaps, risk scoring |
| **⚡ Smart Actions** | AI-powered action recommendations | Email drafts, call scripts, collections |
| **🤖 AI Assistant** | Natural language Q&A | Financial expertise, data insights |
| **📈 What-If Simulator** | Scenario planning | Impact analysis, forecasting |

### Key Metrics Tracked

- **Days Sales Outstanding (DSO)**
- **Collection Rate**
- **Aging Analysis**
- **Risk Scores**
- **Customer Segments**

---

## ⚡ Smart Actions - The Complete Guide

The Smart Actions tab is the heart of InvoiceFlow, providing AI-powered recommendations and full-screen action interfaces.

### 🔧 Email Configuration

**Before using any email features, configure your settings:**

1. **Go to Smart Actions tab**
2. **Expand "⚙️ Email Configuration"**
3. **Fill in required fields:**
   - **Your Email Address:** Where draft emails will be sent
   - **Your Name:** For email signatures
   - **SMTP Settings (Optional):** For automatic sending

**SMTP Setup (Optional):**
- **Host:** `smtp.gmail.com` (for Gmail)
- **Username:** Your email address
- **Password:** Your app password (not regular password!)

> 💡 **Pro Tip:** For Gmail, you'll need to generate an "App Password" in your Google Account settings.

### 📧 Email Actions (Full-Screen Mode)

When you click "📧 Email" on any action:

#### What Happens:
1. **Full-screen interface opens** (other actions auto-collapse)
2. **Customer information displays** with contact details and phone number
3. **AI-generated email draft** appears with professional content
4. **You can edit** the subject and body before sending

#### Email Workflow:
1. **Review customer info** (including phone number underneath)
2. **Edit the email draft** as needed
3. **Click "📧 Send Email Draft"**
4. **Check your email** for the forwarding message
5. **Copy the content** and send to your customer

#### Email Features:
- ✅ **Professional templates** based on urgency level
- ✅ **Customer-specific content** with amounts and days overdue
- ✅ **Editable subjects and bodies**
- ✅ **Email preview** before sending
- ✅ **Contact information** including phone numbers

### 📞 Call Actions (Full-Screen Mode)

When you click "📞 Call" on any action:

#### Features:
- **Customer contact information** with phone number prominently displayed
- **Call planning tools:**
  - Call objective selection
  - Preferred time slots
  - Pre-call notes area
- **AI-generated call script** with professional talking points
- **Action buttons:**
  - "📞 Make Call Now" - initiates call
  - "📅 Schedule Call" - for later follow-up
  - "📝 Log Call Notes" - save conversation details

#### Call Script Includes:
- **Professional opening** with customer name
- **Clear purpose statement** with specific amounts
- **Discussion points** checklist
- **Professional closing** with next steps

### 📋 Collections (Full-Screen Mode)

When you click "📋 Collections" on any action:

#### Escalation Configuration:
- **Escalation Type:** Standard, Legal Action, Payment Plan, Account Closure
- **Escalation Reason:** No Response, Failed Promises, Dispute, High Risk
- **Urgency Level:** Automatically set based on AI analysis

#### Risk Assessment:
- **Payment History:** Excellent to Poor rating
- **Communication Response:** Responsive to Unresponsive
- **Recovery Likelihood:** High to Very Low

#### Actions Available:
- **🚨 Escalate Now** - immediate collections escalation
- **📧 Send Final Notice** - automated final warning
- **📊 Generate Report** - comprehensive collections report

### ℹ️ Details View (Full-Screen Mode)

The details view provides comprehensive information:

#### Account Information:
- Customer ID and contact details
- Outstanding amounts and days overdue
- Priority scores and urgency levels
- Payment history and trends

#### Contact Information:
- **Email address** for the customer
- **Phone number** displayed prominently
- **Contact person** name

#### Email Draft Preview:
- **Full email content** generated by AI
- **Professional formatting** based on urgency
- **Customizable content** for your specific needs

---

## 📧 Email System

### How It Works

InvoiceFlow uses a **simplified email workflow** designed for maximum ease of use:

1. **You configure your email** in the Smart Actions tab
2. **When you send an email draft**, it goes to **YOUR email address**
3. **You forward it to the customer** after reviewing
4. **No complex API setup required!**

### Email Content Features

#### AI-Generated Content:
- **Smart subject lines** based on urgency and amount
- **Professional email bodies** with:
  - Customer-specific details
  - Outstanding amounts
  - Days overdue information
  - Appropriate urgency tone
  - Clear next steps

#### Customizable Elements:
- **Edit subject lines** before sending
- **Modify email body** content
- **Add personal touches** to messages
- **Preview before sending**

### Email Types

| Urgency Level | Email Type | Tone |
|---------------|------------|------|
| **Critical** | Final Notice | Urgent, formal |
| **High** | Urgent Reminder | Direct, professional |
| **Medium** | Payment Reminder | Friendly, professional |
| **Low** | Courtesy Notice | Gentle, relationship-focused |

---

## 📊 Dashboard Features

### Executive Dashboard

**Key Metrics Cards:**
- Total Receivables
- Days Sales Outstanding (DSO)
- Collection Rate
- Overdue Amount

**Visualizations:**
- Aging analysis charts
- Trend lines
- Customer segment breakdowns
- Monthly comparisons

### Risk Analysis Dashboard

**Enhanced Heatmap Features:**
- **Real-time risk scoring** across customer segments
- **Interactive tooltips** with specific risk scores
- **Color-coded risk levels:**
  - 🔴 Red: High Risk (70+)
  - 🟡 Yellow: Medium Risk (40-69)
  - 🟢 Green: Low Risk (<40)

**Risk Categories:**
- Days Overdue (0-30, 31-60, 61-90, 90+)
- High Amount (>$50K)
- Payment History Issues
- Communication Unresponsive
- Credit Risk Concerns

**Summary Statistics:**
- Average risk score across all areas
- Highest risk score identification
- Count of high/medium/low risk areas
- Risk trend indicators

---

## 📈 Data Management

### Sample Data

InvoiceFlow comes with realistic sample data including:

- **500 sample invoices** across different customer segments
- **Realistic payment patterns** based on customer reliability
- **Various customer segments:** Enterprise, Mid-Market, Small Business, Startup
- **Multiple payment methods:** ACH, Wire, Check, Credit Card

### Data Sources

The application can work with:
- **CSV file uploads** for invoices and payments
- **Sample data** for demonstration
- **Real data integration** (contact support for custom connectors)

### Data Quality

Built-in data validation ensures:
- ✅ Proper date formatting
- ✅ Valid amount ranges
- ✅ Customer ID consistency
- ✅ Status validation

---

## 🛠️ Troubleshooting

### Common Issues

#### Application Won't Start
```bash
# Try the automated launcher
./run_app.sh

# Or manually:
source venv/bin/activate
pip install -e .
streamlit run app.py
```

#### Email Features Not Working
1. **Check email configuration** in Smart Actions tab
2. **Verify your email address** is correctly entered
3. **For SMTP issues:** Ensure you're using an app password, not your regular password
4. **Fallback:** The system will use mailto links if SMTP fails

#### No Data Showing
1. **Regenerate sample data:**
   ```bash
   python3 generate_sample_data.py
   ```
2. **Check file permissions** on CSV files
3. **Restart the application**

#### Smart Actions Not Expanding
1. **Click the close (❌) button** on any expanded action
2. **Refresh the page** if actions seem stuck
3. **Clear browser cache** if issues persist

### Error Messages

| Error | Solution |
|-------|----------|
| "No data available" | Generate sample data or upload CSV files |
| "Email configuration missing" | Set up email in Smart Actions tab |
| "Import error" | Run `pip install -e .` to install dependencies |
| "Port already in use" | Close other Streamlit instances or use different port |

---

## 💡 Advanced Tips

### Maximizing Efficiency

1. **Set up email configuration first** - this enables all email features
2. **Use full-screen actions** - click any action button for detailed interfaces
3. **Customize email templates** - edit content before sending
4. **Monitor the heatmap** - it shows your highest-risk areas
5. **Use the AI assistant** - ask questions about your data

### Best Practices

#### Email Management:
- **Review all email drafts** before forwarding to customers
- **Personalize messages** when appropriate
- **Keep track of sent emails** for follow-up
- **Use appropriate urgency levels** based on customer relationship

#### Risk Management:
- **Check the heatmap daily** for new high-risk areas
- **Prioritize actions** based on AI recommendations
- **Focus on high-value customers** for personal outreach
- **Use collections escalation** sparingly and strategically

#### Data Analysis:
- **Monitor DSO trends** for overall performance
- **Segment analysis** helps identify patterns
- **Use what-if scenarios** for strategic planning
- **Regular risk assessment** prevents issues from escalating

### Keyboard Shortcuts

- **Tab navigation** between main sections
- **Expand/collapse** actions with Enter key
- **Scroll to top** with Home key
- **Refresh data** with F5

---

## 🆘 Support & Resources

### Getting Help

1. **Built-in AI Assistant** - Ask questions about your data
2. **User Guide** - This comprehensive documentation
3. **Error messages** - Usually include specific guidance
4. **Sample data** - Use for testing and learning

### Feature Requests

InvoiceFlow is continuously improving. Common requested features:
- Advanced email templates
- Direct integration with accounting systems
- Mobile-responsive design
- Additional risk models
- Custom reporting

---

## 🎉 Success Stories

### Typical User Journey

1. **Day 1:** Set up email configuration, explore sample data
2. **Day 2:** Start using Smart Actions for high-priority accounts
3. **Week 1:** Establish routine of checking heatmap and sending emails
4. **Month 1:** See improvement in DSO and collection rates
5. **Ongoing:** Use AI insights for strategic decisions

### Expected Benefits

- **Reduced DSO** through proactive communication
- **Improved collection rates** with AI-powered prioritization
- **Better customer relationships** through professional communication
- **Risk mitigation** through early identification
- **Time savings** through automation and templates

---

**🚀 Ready to transform your receivables management? Start with the Smart Actions tab and configure your email settings!**

---

*For additional support or questions, use the built-in AI Assistant or refer to the troubleshooting section above.*
