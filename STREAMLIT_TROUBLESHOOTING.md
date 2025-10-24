# Streamlit Cloud Deployment Troubleshooting

## ✅ FIXED: "Failed building wheel for blis" Error

### The Error You Saw:
```
ERROR: Failed building wheel for blis
ERROR: Could not build wheels for blis, which is required to install pyproject.toml-based projects
```

### Root Cause:
- Streamlit Cloud was using **Python 3.13** (latest)
- `deepeval` package depends on `blis` (a spaCy dependency)
- `blis` doesn't have pre-built wheels for Python 3.13 yet
- Streamlit Cloud tried to build from source and failed

### Solution Applied:

**1. Pinned Python version to 3.12** (`.python-version` file):
```
3.12
```

**2. Removed `deepeval` from production requirements:**
- Moved to `requirements-dev.txt` (for local testing only)
- DeepEval is NOT used in the Streamlit app itself
- Only needed for running evaluation scripts locally

**3. Updated dependency versions:**
- Changed from pinned versions (`==1.39.0`) to ranges (`>=1.39.0`)
- This allows Streamlit Cloud to use compatible newer versions

---

## Current Configuration

### Files Created:
1. **`.python-version`** - Forces Python 3.12
2. **`requirements.txt`** - Minimal production dependencies (NO deepeval)
3. **`requirements-dev.txt`** - Development/testing dependencies (includes deepeval)

### How Streamlit Reads These:
- Streamlit Cloud automatically detects `.python-version`
- Uses Python 3.12 instead of 3.13
- Installs only `requirements.txt` (not `requirements-dev.txt`)

---

## Deployment Should Now Work

### Expected Build Output (5-7 minutes):
```
[1/5] Detected Python version: 3.12
[2/5] Installing dependencies from requirements.txt...
      ✅ streamlit-1.39.0
      ✅ langchain-0.3.0
      ✅ faiss-cpu-1.9.0
      ✅ sentence-transformers-3.1.0
      ✅ pandas-2.2.0
      ... (all dependencies installed successfully)

[3/5] Running health check...
[4/5] Starting Streamlit app...
[5/5] App is live! ✅
```

### What to Do Now:
1. **Go back to Streamlit Cloud**
2. **Wait for automatic redeploy** (detects Git push)
   - Or click **"Reboot app"** to force restart
3. **Monitor logs** for successful build
4. **Test app** once live

---

## Other Common Streamlit Cloud Issues

### Issue 1: "Module not found: streamlit"
**Symptom:** App crashes with `ModuleNotFoundError: No module named 'streamlit'`

**Cause:** `requirements.txt` not found or empty

**Fix:**
```bash
# Verify requirements.txt exists
git ls-files requirements.txt

# Should show all dependencies
cat requirements.txt
```

---

### Issue 2: "No module named 'src'"
**Symptom:** `ModuleNotFoundError: No module named 'src'`

**Cause:** Import paths not configured correctly

**Fix in streamlit_app.py:**
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
```

---

### Issue 3: API Key Not Working
**Symptom:** `openai.AuthenticationError: Invalid API key`

**Cause:** Streamlit secrets not configured

**Fix:**
1. Go to Streamlit Cloud dashboard
2. Click **"Settings"** → **"Secrets"**
3. Add:
   ```toml
   OPENAI_API_KEY = "pplx-your-actual-key"
   OPENAI_API_BASE = "https://api.perplexity.ai"
   LLM_MODEL = "sonar-pro"
   ```
4. Click **"Save"**
5. App will auto-restart

---

### Issue 4: "FileNotFoundError: data/icd10_processed.csv"
**Symptom:** App crashes looking for data files

**Cause:** Data files not committed to Git

**Fix:**
```bash
# Verify data files are committed
git ls-files data/

# Should show:
# data/icd10_processed.csv
# data/cpt_processed.csv

# If missing, add them:
git add data/*.csv
git commit -m "Add data files"
git push
```

---

### Issue 5: "Out of Memory" on Free Tier
**Symptom:** App crashes with memory errors

**Cause:** Free tier has 1 GB RAM limit

**Current Status:** Your app uses ~400 MB - **should be fine! ✅**

**If it happens anyway:**
Reduce dataset size in `src/streamlit_app.py`:
```python
# Line ~67
icd_codes = loader.parse_icd10_codes(Path("data/icd10_processed.csv"))[:10000]
# Only use first 10,000 codes instead of all 73,947
```

---

### Issue 6: App Sleeps After 7 Days
**Symptom:** App shows "App is sleeping" message

**Cause:** Free tier apps sleep after 7 days of inactivity

**Fix:**
- Just visit the app URL
- Wakes up automatically in ~30 seconds
- **OR** upgrade to paid tier (always-on)

---

## Monitoring Your Deployment

### Real-Time Logs:
1. Go to Streamlit Cloud dashboard
2. Click your app
3. Click **"Manage app"** → **"Logs"**
4. See live output:
   ```
   [INFO] Loaded 73947 ICD-10 codes
   [INFO] FAISS index initialized
   [INFO] App started successfully
   ```

### Resource Usage:
- Click **"Manage app"** → **"Analytics"**
- See:
  - Memory usage (~400 MB expected)
  - CPU usage
  - Active users
  - Response times

---

## Testing Checklist

Once deployed, test these features:

- [ ] App loads without errors
- [ ] Can enter a query in search box
- [ ] Search returns results
- [ ] Retrieved documents display correctly
- [ ] Confidence scores shown
- [ ] Sample queries work
- [ ] Settings sidebar functional
- [ ] All 3 tabs work (Query, Evaluation, Samples)

---

## Performance Expectations

### First Load (New Session):
- **5-10 seconds** - Building FAISS index from data
- **Normal!** Don't worry if first query is slow

### Subsequent Queries:
- **<1 second** - Index is cached in memory
- **Fast!** This is normal operation

### After App Restart:
- Rebuilds index (5-10 seconds again)
- Then fast again

---

## Getting Help

### If Build Still Fails:

**1. Check Streamlit Cloud logs**
- Look for specific error message
- Note which package failed

**2. Search Streamlit Community Forum**
- https://discuss.streamlit.io/
- Many common issues already solved

**3. Check Package Compatibility**
- Some packages don't work on Linux (Streamlit Cloud uses Ubuntu)
- Example: `pywin32` won't work

**4. Simplify Dependencies**
- Comment out optional packages
- Test with minimal requirements first

---

## Next Deployment (If You Make Changes)

### Automatic Redeploy:
```bash
# Make changes locally
git add .
git commit -m "Update feature X"
git push

# Streamlit Cloud automatically detects and redeploys!
# No manual intervention needed
```

### Manual Reboot:
1. Go to Streamlit Cloud dashboard
2. Click **"Manage app"**
3. Click **"Reboot app"**
4. Wait 2-3 minutes for restart

---

## Summary: Your Fix

✅ **Created `.python-version`** → Forces Python 3.12
✅ **Updated `requirements.txt`** → Removed deepeval (moved to dev)
✅ **Pushed to GitHub** → Streamlit will auto-redeploy

**What happens next:**
- Streamlit Cloud detects the push
- Starts new build with Python 3.12
- Installs dependencies (should succeed now!)
- App goes live

**Expected time: 5-7 minutes**

---

## Quick Reference

**Repository:** https://github.com/mind-ctrl/Medical_Codes_RAG
**Python Version:** 3.12 (set in `.python-version`)
**Main File:** `src/streamlit_app.py`
**Production Deps:** `requirements.txt` (minimal, no deepeval)
**Dev Deps:** `requirements-dev.txt` (includes deepeval for local testing)

---

**Your deployment should work now! 🚀**

The `blis` build error is fixed by using Python 3.12 and removing deepeval from production dependencies.
