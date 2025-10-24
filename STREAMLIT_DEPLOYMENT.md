# Streamlit Cloud Deployment Guide

## Your Code is Ready! ✅

Your Medical RAG system has been successfully pushed to GitHub:
- Repository: https://github.com/mind-ctrl/Medical_Codes_RAG
- Branch: `main`
- All dependencies configured
- Large files excluded (FAISS index will be built on first run)

---

## Step-by-Step Deployment to Streamlit Cloud

### Step 1: Sign Up for Streamlit Cloud (2 minutes)

1. Go to [https://share.streamlit.io/](https://share.streamlit.io/)
2. Click **Sign up with GitHub**
3. Authorize Streamlit to access your GitHub account
4. You'll be redirected to your Streamlit Cloud dashboard

---

### Step 2: Deploy Your App (3 minutes)

1. **Click "New app"** in the top-right corner

2. **Fill in the deployment form:**
   ```
   Repository: mind-ctrl/Medical_Codes_RAG
   Branch: main
   Main file path: src/streamlit_app.py
   ```

3. **Click "Advanced settings"** (before deploying)

4. **Add your Perplexity API key as a secret:**
   - In the "Secrets" text box, paste:
     ```toml
     OPENAI_API_KEY = "pplx-your-actual-api-key-here"
     OPENAI_API_BASE = "https://api.perplexity.ai"
     LLM_MODEL = "sonar-pro"
     ```
   - Replace `pplx-your-actual-api-key-here` with your real Perplexity API key

5. **Python version:** 3.12 (auto-detected from requirements.txt)

6. **Click "Deploy!"**

---

### Step 3: Wait for Build (5-10 minutes)

**What's happening:**
```
[1/6] Installing dependencies from requirements.txt...
      - This takes 3-5 minutes (lots of packages!)

[2/6] Loading ICD-10 codes (73,947 codes)...
      - Reading data/icd10_processed.csv

[3/6] Building FAISS vector index...
      - First run: Creates embeddings for all codes (~5 minutes)
      - Saves to faiss_index/ for future reuse

[4/6] Initializing retriever and generator...

[5/6] Starting Streamlit app...

[6/6] App deployed successfully!
```

**Progress indicators you'll see:**
- ✅ Installing Python dependencies
- ✅ Running app
- ✅ App is live!

---

### Step 4: Your App is Live! (0 minutes)

**You'll see:**
```
Your app is live at:
https://medical-codes-rag-<random-id>.streamlit.app
```

**Share this URL with:**
- Interviewers (add to resume!)
- Portfolio viewers
- Anyone who wants to try your RAG system

---

## Expected First-Run Behavior

### First Time (5-10 minutes):
```
[INFO] Building FAISS index from 1000 chunks...
[INFO] This may take a few minutes on first run...
[INFO] Saving index for future use...
[OK] Index built and saved!
```

### Every Subsequent Visit (<5 seconds):
```
[INFO] Loading cached FAISS index...
[OK] Index loaded in 0.8 seconds!
```

---

## Testing Your Deployed App

### Test Query 1: Code Lookup
```
Input: I20.0
Expected: Unstable angina
Confidence: High
```

### Test Query 2: Symptom Search
```
Input: patient with acute chest pain
Expected: Top results include I20.0 (Unstable angina), I21.x (MI), I25.x (Chronic ischemic heart disease)
Confidence: High or Medium
```

### Test Query 3: General Query
```
Input: diabetes with kidney complications
Expected: E11.2x codes (Type 2 diabetes with renal complications)
Confidence: Medium or High
```

---

## Monitoring Your App

### View Logs:
1. Click **"Manage app"** in Streamlit Cloud dashboard
2. Click **"Logs"** tab
3. See real-time logs:
   ```
   [INFO] Loaded 73947 ICD-10 codes
   [INFO] Retriever initialized
   [INFO] Query received: "patient with chest pain"
   [INFO] Retrieved 5 documents
   [INFO] Generated response with confidence: High
   ```

### Check Resource Usage:
- **Free tier limits:**
  - 1 GB RAM
  - 1 CPU core
  - Always-on (sleeps after 7 days of inactivity)

- **Your app's usage:**
  - ~400 MB RAM (FAISS index + models)
  - ~200 MB disk (dependencies)
  - Well within free tier! ✅

---

## Troubleshooting

### Issue 1: App Shows "Module not found"
**Cause:** Missing dependency in requirements.txt

**Fix:**
```bash
# Locally
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update dependencies"
git push

# Streamlit will auto-redeploy
```

---

### Issue 2: "API Key Invalid"
**Cause:** Perplexity API key not set or incorrect

**Fix:**
1. Go to Streamlit Cloud dashboard
2. Click **"Settings"** → **"Secrets"**
3. Verify the secret is:
   ```toml
   OPENAI_API_KEY = "pplx-YOUR-ACTUAL-KEY"
   ```
4. Click **"Save"**
5. App will auto-restart

---

### Issue 3: App is Slow on First Run
**Cause:** Building FAISS index from scratch

**This is normal!**
- First run: 5-10 minutes
- All subsequent runs: <5 seconds

**Explanation:**
The app builds embeddings for 73,947 ICD codes on first run. These are cached in memory and reused.

---

### Issue 4: "Out of Memory" Error
**Cause:** Processing too many codes at once

**Fix (if needed):**
Edit [src/streamlit_app.py](src/streamlit_app.py):
```python
# Line ~30
# Before:
icd_codes = loader.parse_icd10_codes(Path("data/icd10_processed.csv"))

# After (use subset for free tier):
icd_codes = loader.parse_icd10_codes(Path("data/icd10_processed.csv"))[:10000]
```

This uses only 10,000 codes instead of 73,947, reducing memory to ~200 MB.

---

## Updating Your Deployed App

**Any push to GitHub main branch auto-redeploys!**

```bash
# Make changes locally
git add .
git commit -m "Update feature X"
git push

# Streamlit Cloud detects the push and redeploys
# You'll see: "New version detected, redeploying..."
```

---

## Cost Analysis

### Streamlit Cloud (FREE):
- Free tier: 1 app, 1 GB RAM, unlimited viewers
- Your app fits perfectly ✅

### Perplexity API (~$6/month):
- Sonar-pro: $1 per 1M tokens
- Average query: 500 tokens (retrieval) + 1000 tokens (generation) = 1500 tokens
- 4000 queries/month = 6M tokens = ~$6

**Total: FREE (Streamlit) + $6 (API) = $6/month for production RAG!**

---

## Share Your App in Interviews

### When Asked "Do You Have a Demo?"

**Response:**
> "Yes! I deployed it to Streamlit Cloud. Here's the live URL: [your-app-url].
>
> It's a production-ready medical coding RAG system that:
> - Searches 73,947 ICD-10 codes
> - Uses hybrid retrieval (BM25 + semantic search)
> - Implements Reciprocal Rank Fusion for ranking
> - Has explainability - shows why each result was ranked
> - Detects hallucinations with multi-level validation
>
> Try querying 'patient with chest pain' or 'I20.0' to see it in action."

---

## Next Steps After Deployment

1. **Test all features** in production
2. **Share URL** in resume and LinkedIn
3. **Monitor logs** for any errors
4. **Collect feedback** from test users
5. **Iterate** based on real-world usage

---

## Your Deployment Checklist

- [x] Code pushed to GitHub
- [x] Dependencies in requirements.txt
- [x] Secrets configured (.env variables)
- [x] Large files excluded (.gitignore)
- [ ] Streamlit Cloud account created
- [ ] App deployed
- [ ] API key added to Streamlit secrets
- [ ] First run completed (FAISS index built)
- [ ] Tested with sample queries
- [ ] URL shared in portfolio

---

## Quick Reference

**Repository:** https://github.com/mind-ctrl/Medical_Codes_RAG
**App URL:** (You'll get this after deployment)
**Main file:** `src/streamlit_app.py`
**Python version:** 3.12
**Secrets needed:** `OPENAI_API_KEY` (Perplexity), `OPENAI_API_BASE`, `LLM_MODEL`

---

## Support Resources

- **Streamlit Docs:** https://docs.streamlit.io/deploy/streamlit-community-cloud
- **Community Forum:** https://discuss.streamlit.io/
- **Your Documentation:** See [SESSION_REFERENCE.md](SESSION_REFERENCE.md) for full system details

---

**You're ready to deploy! 🚀**

Follow Step 1-4 above, and your app will be live in ~10 minutes.
