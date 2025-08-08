# 🚀 Deploy to Streamlit Cloud - Step by Step

## **Step 1: Create GitHub Repository**

1. **Go to** [GitHub.com](https://github.com)
2. **Sign in** to your account (gabbielove05)
3. **Click** the green "New" button
4. **Repository name:** `smart-receivables`
5. **Make it Public** ✅
6. **Don't check** "Add a README file"
7. **Click** "Create repository"

## **Step 2: Push Your Code**

After creating the repository, run these commands:

```bash
cd /Users/gabbielove/smart-receivables
git remote set-url origin https://github.com/gabbielove05/smart-receivables.git
git push -u origin main
```

## **Step 3: Deploy on Streamlit Cloud**

1. **Go to** [share.streamlit.io](https://share.streamlit.io)
2. **Sign in** with GitHub
3. **Click** "New app"
4. **Repository:** `gabbielove05/smart-receivables`
5. **Branch:** `main`
6. **Main file path:** `InvoiceFlow/app.py`
7. **Click** "Deploy"

## **Step 4: Configure Secrets**

After deployment, go to your app settings:

1. **Click** on your app name
2. **Go to** "Settings" → "Secrets"
3. **Add this secret:**
   ```
   OPENAI_API_KEY = "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"
   ```
4. **Save** the secrets

## **Step 5: Your App is Live!**

Your app will be available at:
`https://smart-receivables-gabbielove05.streamlit.app`

---

## **Troubleshooting**

### **If GitHub repo creation fails:**
- Make sure you're logged into the correct GitHub account
- Try a different repository name if `smart-receivables` is taken

### **If push fails:**
- Make sure the repository exists on GitHub
- Check that you have the correct permissions

### **If Streamlit deployment fails:**
- Verify the main file path is correct: `InvoiceFlow/app.py`
- Check that all dependencies are in `requirements.txt`
- Ensure secrets are properly configured

---

## **Quick Commands**

```bash
# Create and push to GitHub
git remote set-url origin https://github.com/gabbielove05/smart-receivables.git
git push -u origin main

# Then go to share.streamlit.io and deploy!
```

**Your app will be live and accessible to anyone! 🌐**
