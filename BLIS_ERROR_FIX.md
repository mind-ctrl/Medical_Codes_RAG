# BLIS Build Error - Complete Fix Guide

## The Error (Again):
```
ERROR: Failed building wheel for blis
ERROR: Could not build wheels for blis, which is required to install pyproject.toml-based projects
```

---

## What We've Tried (Multiple Approaches):

### ✅ Attempt 1: `.python-version` file
**Created:** `.python-version` with content `3.12`
**Result:** Streamlit Cloud may not respect this file
**Status:** Kept as backup

### ✅ Attempt 2: `runtime.txt` (RECOMMENDED)
**Created:** `runtime.txt` with content `python-3.12.7`
**Why:** Streamlit Cloud officially uses `runtime.txt` for Python version
**Status:** **This is the standard method**

### ✅ Attempt 3: Downgrade sentence-transformers
**Changed:** `sentence-transformers==3.1.0` → `sentence-transformers==2.7.0`
**Why:** Version 3.x might have transitive dependencies causing issues
**Status:** Applied

### ✅ Attempt 4: Removed DeepEval
**Moved:** `deepeval` to `requirements-dev.txt`
**Why:** DeepEval depends on spaCy which depends on blis
**Status:** Already done

---

## Current Configuration (Latest Push):

### File 1: `runtime.txt` (NEW - Most Important!)
```
python-3.12.7
```

**This is what Streamlit Cloud actually reads!**

### File 2: `requirements.txt` (Updated)
```python
# Downgraded sentence-transformers
sentence-transformers==2.7.0  # Was 3.1.0

# All other dependencies pinned to known-working versions
streamlit==1.39.0
langchain==0.3.0
faiss-cpu==1.8.0
# ... etc
```

### File 3: `.python-version` (Backup)
```
3.12
```

---

## Why This Should Work Now:

1. **`runtime.txt` is the official method** - Streamlit Cloud documentation specifically mentions this file
2. **sentence-transformers 2.7.0** - Older, stable version with better compatibility
3. **No spaCy/blis dependencies** - Completely removed from production
4. **Strict version pinning** - No dependency resolver conflicts

---

## What to Do Right Now:

### Step 1: Clear Streamlit Cache (IMPORTANT!)
Sometimes Streamlit Cloud caches the old build configuration.

**Method A: Via Dashboard**
1. Go to https://share.streamlit.io/
2. Click on your app
3. Click **"⋮" (three dots)** → **"Delete app"**
4. Create NEW app with same settings
   - Repository: `mind-ctrl/Medical_Codes_RAG`
   - Branch: `main`
   - Main file: `src/streamlit_app.py`

**Method B: Reboot** (try this first)
1. Go to app settings
2. Click **"Reboot app"**
3. Watch logs for Python version detection

---

### Step 2: Monitor Build Logs

**Look for this in logs:**
```
✅ Detected Python version: 3.12.7 (from runtime.txt)
✅ Installing streamlit-1.39.0
✅ Installing sentence-transformers-2.7.0
✅ Installing faiss-cpu-1.8.0
...
✅ Successfully installed all dependencies!
```

**Red flag (if you see this, runtime.txt isn't working):**
```
❌ Detected Python version: 3.13.x
```

---

## If It STILL Fails:

### Nuclear Option 1: Use packages.txt
Create `packages.txt` with system dependencies:
```
build-essential
python3.12-dev
```

### Nuclear Option 2: Remove sentence-transformers entirely
Replace with simpler embedding model:
```python
# In requirements.txt
# sentence-transformers==2.7.0  # Comment out
huggingface-hub==0.20.0
```

Then modify retrieval code to use basic Hugging Face without sentence-transformers wrapper.

### Nuclear Option 3: Use pre-built embeddings
Generate embeddings locally and commit them to repo:
```bash
# Locally
python -c "
from src.data_loader import MedicalDataLoader
from src.retrieval import HybridRetriever
loader = MedicalDataLoader()
retriever = HybridRetriever()
icd_codes = loader.parse_icd10_codes('data/icd10_processed.csv')
chunks = loader.create_chunks_with_metadata(icd_codes)
retriever.initialize_vectorstore(chunks)
retriever.vectorstore.save_local('embeddings_prebuilt')
"

# Commit the embeddings
git add embeddings_prebuilt/
git commit -m "Add pre-built embeddings"
git push
```

Then modify app to load pre-built instead of generating.

---

## Alternative: Use Streamlit in Containers

If Streamlit Cloud keeps failing, deploy via Docker on:

### Option A: Heroku
```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD streamlit run src/streamlit_app.py --server.port=$PORT
```

### Option B: Railway.app
- Connect GitHub repo
- Railway auto-detects Python app
- Set build command: `pip install -r requirements.txt`
- Set start command: `streamlit run src/streamlit_app.py`

### Option C: Render.com
- Similar to Railway
- Free tier available
- Better for Python apps than Streamlit Cloud sometimes

---

## Debugging Commands (If You Have SSH Access)

**Check actual Python version:**
```bash
python --version
```

**Check if blis is actually needed:**
```bash
pip install -r requirements.txt --dry-run 2>&1 | grep blis
```

**Install without blis:**
```bash
pip install -r requirements.txt --no-deps
pip install <each package individually>
```

---

## Understanding the Root Cause

### Why does `blis` need to be built?

1. **`sentence-transformers`** → depends on **`transformers`**
2. **`transformers`** → optionally depends on **`spaCy`** (for some models)
3. **`spaCy`** → depends on **`thinc`**
4. **`thinc`** → depends on **`blis`** (fast BLAS routines)

### Why does building fail?

- `blis` is a Cython extension (needs compilation)
- Python 3.13 is brand new (released Oct 2024)
- `blis` maintainers haven't released pre-built wheels for 3.13 yet
- So pip tries to build from source
- Building needs C compiler + headers
- Streamlit Cloud's build environment might not have these

### The Fix:

**Stay on Python 3.12** where all packages have pre-built wheels!

---

## Verification Checklist

After deployment, verify:

- [ ] App loads without errors
- [ ] Python version is 3.12.x (check logs)
- [ ] No "building wheel for blis" messages in logs
- [ ] All dependencies installed successfully
- [ ] FAISS index builds on first run
- [ ] Queries return results
- [ ] No import errors

---

## Current Status Summary

**Files pushed to GitHub:**
- ✅ `runtime.txt` - Python 3.12.7 (NEW - most important!)
- ✅ `requirements.txt` - Downgraded sentence-transformers to 2.7.0
- ✅ `.python-version` - 3.12 (backup)
- ✅ `requirements-dev.txt` - Dev dependencies separate

**What should happen:**
1. Streamlit Cloud reads `runtime.txt`
2. Uses Python 3.12.7
3. Installs sentence-transformers 2.7.0 (has wheels for 3.12)
4. No blis compilation needed
5. Build succeeds

**Time to redeploy:** ~5-7 minutes

**Your action:** Reboot app or delete/recreate app in Streamlit Cloud

---

## If ALL ELSE Fails: Contact Me

If none of these work:

1. **Share full build logs** - Copy entire log output
2. **Check Streamlit Community** - https://discuss.streamlit.io/
3. **Consider alternative deployment** - Railway, Render, Heroku

**Most likely issue:** Streamlit Cloud not reading `runtime.txt`
**Solution:** Delete app and create new one (fresh build)

---

## Success Indicators

**Build logs should show:**
```
Python version: 3.12.7
Installing collected packages: sentence-transformers-2.7.0
Successfully installed faiss-cpu-1.8.0
App started successfully
```

**App should:**
- Load in ~5-10 seconds
- Build FAISS index on first query
- Return results for "patient with chest pain"
- Show confidence scores
- Display retrieved documents

---

**TL;DR:**
1. Created `runtime.txt` with `python-3.12.7`
2. Downgraded `sentence-transformers` to `2.7.0`
3. Pushed to GitHub
4. **Now: Reboot or recreate app in Streamlit Cloud**

This should definitely work now! 🤞
