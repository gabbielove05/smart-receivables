# 🚀 Smart Receivables Navigator - Deployment Guide

## 📍 **How to Access Your App Locally**

### **Option 1: Direct Access**
Your app is currently running at: **http://localhost:8501**

### **Option 2: Start the App**
```bash
cd /Users/gabbielove/smart-receivables/InvoiceFlow
source venv/bin/activate
streamlit run app.py --server.port 8501
```

### **Option 3: VS Code/Cursor Debug**
1. Open VS Code/Cursor
2. Press `F5` or go to Run → Start Debugging
3. Select "Streamlit App" configuration

---

## 🌐 **Deploy for Public Use**

### **Option 1: Streamlit Cloud (Recommended)**

#### **Step 1: Prepare Your Repository**
```bash
# Create a requirements.txt file
cd /Users/gabbielove/smart-receivables/InvoiceFlow
pip freeze > requirements.txt
```

#### **Step 2: Create GitHub Repository**
1. Go to [GitHub.com](https://github.com)
2. Create a new repository named `smart-receivables`
3. Upload your files:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/smart-receivables.git
   git push -u origin main
   ```

#### **Step 3: Deploy on Streamlit Cloud**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `smart-receivables`
5. Set main file path: `InvoiceFlow/app.py`
6. Add environment variables:
   - `OPENAI_API_KEY`: `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`
7. Click "Deploy"

**Your app will be available at:** `https://YOUR_APP_NAME.streamlit.app`

---

### **Option 2: Heroku**

#### **Step 1: Create Heroku Files**
```bash
# Create Procfile
echo "web: streamlit run InvoiceFlow/app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Create runtime.txt
echo "python-3.11.0" > runtime.txt

# Update requirements.txt
cd InvoiceFlow
pip freeze > requirements.txt
```

#### **Step 2: Deploy to Heroku**
```bash
# Install Heroku CLI
brew install heroku/brew/heroku

# Login to Heroku
heroku login

# Create Heroku app
heroku create your-app-name

# Set environment variables
heroku config:set OPENAI_API_KEY="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"

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

---

### **Option 3: Railway**

#### **Step 1: Prepare for Railway**
1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository

#### **Step 2: Configure**
- **Build Command:** `pip install -r InvoiceFlow/requirements.txt`
- **Start Command:** `streamlit run InvoiceFlow/app.py --server.port=$PORT --server.address=0.0.0.0`
- **Environment Variables:**
  - `OPENAI_API_KEY`: `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`

---

### **Option 4: DigitalOcean App Platform**

#### **Step 1: Prepare App Spec**
Create `app.yaml`:
```yaml
name: smart-receivables
services:
- name: web
  source_dir: /InvoiceFlow
  github:
    repo: YOUR_USERNAME/smart-receivables
    branch: main
  run_command: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
  environment_slug: python
  instance_count: 1
  instance_size_slug: basic-xxs
  envs:
  - key: OPENAI_API_KEY
    value: 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
```

#### **Step 2: Deploy**
1. Go to [cloud.digitalocean.com](https://cloud.digitalocean.com)
2. Create App Platform
3. Connect GitHub repository
4. Deploy

---

## 🔧 **Troubleshooting**

### **Common Issues:**

#### **1. Import Errors**
```bash
# Make sure you're in the right directory
cd /Users/gabbielove/smart-receivables/InvoiceFlow
source venv/bin/activate
pip install -r requirements.txt
```

#### **2. Port Already in Use**
```bash
# Kill existing process
pkill -f streamlit
# Or use different port
streamlit run app.py --server.port 8502
```

#### **3. API Key Issues**
- Check if API key is set in environment variables
- Verify OpenRouter account has credits
- Try reducing `max_tokens` in chatbot.py

#### **4. Deployment Issues**
- Ensure all dependencies are in `requirements.txt`
- Check that main file path is correct
- Verify environment variables are set

---

## 📊 **Monitoring Your App**

### **Local Monitoring**
```bash
# Check if app is running
curl http://localhost:8501

# View logs
tail -f ~/.streamlit/logs/streamlit.log
```

### **Production Monitoring**
- **Streamlit Cloud:** Built-in analytics
- **Heroku:** `heroku logs --tail`
- **Railway:** Built-in logging
- **DigitalOcean:** App Platform monitoring

---

## 🔐 **Security Considerations**

### **For Production:**
1. **Use environment variables** for API keys
2. **Enable authentication** if needed
3. **Set up HTTPS** (automatic on most platforms)
4. **Monitor usage** and costs
5. **Backup data** regularly

### **API Key Security:**
```bash
# Never commit API keys to git
echo "*.env" >> .gitignore
echo "InvoiceFlow/.env" >> .gitignore
```

---

## 🎯 **Quick Start Commands**

### **Local Development:**
```bash
cd /Users/gabbielove/smart-receivables/InvoiceFlow
source venv/bin/activate
streamlit run app.py
```

### **Production Deploy:**
```bash
# Streamlit Cloud (easiest)
# Just push to GitHub and connect to Streamlit Cloud

# Heroku
heroku create your-app-name
git push heroku main

# Railway
# Connect GitHub repo to Railway dashboard
```

---

## 📞 **Support**

If you encounter issues:
1. Check the logs for error messages
2. Verify all dependencies are installed
3. Ensure API keys are correctly set
4. Test locally before deploying

**Your app is ready to deploy! 🚀**
