# Streamlit Cloud Deployment Guide

## Prerequisites

- [x] Streamlit Cloud account (you have this!)
- [x] GitHub account
- [ ] Perplexity API key
- [ ] Git installed locally

---

## Step 1: Prepare Your Repository

### 1.1 Initialize Git (if not already done)

```bash
cd "D:\Power BI Project\Medical_Code_RAG"
git init
git branch -M main
```

### 1.2 Add Essential Files

All deployment files have been created:
- ✓ `requirements.txt` - Python dependencies
- ✓ `.streamlit/config.toml` - App configuration
- ✓ `.streamlit/secrets.toml.example` - Secrets template
- ✓ `.gitignore` - Files to exclude from Git

### 1.3 Commit Your Code

```bash
# Stage all files
git add .

# Commit
git commit -m "Initial deployment for Streamlit Cloud"
```

---

## Step 2: Create GitHub Repository

### 2.1 Create Repository on GitHub

1. Go to [https://github.com/new](https://github.com/new)
2. Repository name: `Medical_Code_RAG` (or your choice)
3. Make it **Public** (required for free Streamlit Cloud)
4. Do NOT initialize with README (you have one)
5. Click "Create repository"

### 2.2 Push to GitHub

```bash
# Add remote origin (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/Medical_Code_RAG.git

# Push to GitHub
git push -u origin main
```

**Important:** If you get authentication errors, you may need to create a Personal Access Token:
- Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
- Generate new token with `repo` scope
- Use the token as your password when pushing

---

## Step 3: Deploy to Streamlit Cloud

### 3.1 Go to Streamlit Cloud

1. Visit [https://share.streamlit.io/](https://share.streamlit.io/)
2. Sign in with your Streamlit account
3. Click "New app"

### 3.2 Configure Deployment

**Repository settings:**
- Repository: `YOUR_USERNAME/Medical_Code_RAG`
- Branch: `main`
- Main file path: `src/streamlit_app.py`

**Advanced settings (click "Advanced settings"):**
- Python version: `3.12` (or 3.11)
- Click "Save"

### 3.3 Add Secrets

Before deploying, click on "Advanced settings" → "Secrets" and add:

```toml
# Paste this in the Secrets section:
OPENAI_API_KEY = "pplx-your-perplexity-api-key-here"
OPENAI_API_BASE = "https://api.perplexity.ai"
LLM_MODEL = "sonar-pro"
```

**Where to get your Perplexity API key:**
- Go to [https://www.perplexity.ai/settings/api](https://www.perplexity.ai/settings/api)
- Copy your API key
- Paste it in the secrets above

### 3.4 Deploy

1. Click "Deploy!"
2. Wait 5-10 minutes for deployment (first time takes longer)
3. Watch the logs for any errors

---

## Step 4: Verify Deployment

### 4.1 Check App Status

Once deployed, you should see:
- ✓ "Your app is live!" message
- ✓ App URL: `https://your-app-name.streamlit.app`

### 4.2 Test Your App

1. Click the app URL
2. Wait for initialization (5-10 seconds)
3. Try a test query: "patient with chest pain"
4. Verify you see:
   - Retrieved documents
   - Generated response
   - Medical codes
   - Confidence scores

---

## Troubleshooting

### Issue 1: App Won't Start

**Error:** `ModuleNotFoundError`

**Solution:** Check that all dependencies are in `requirements.txt`

---

### Issue 2: API Key Not Found

**Error:** `Error generating response: 401 Unauthorized`

**Solution:**
1. Go to app settings (⚙️ icon)
2. Click "Secrets" in left sidebar
3. Verify your secrets are properly formatted (no extra spaces)
4. Click "Save"
5. Reboot app

---

### Issue 3: Out of Memory

**Error:** `MemoryError` or app crashes during initialization

**Solutions:**
1. Reduce data size in `src/data_loader.py`:
   ```python
   # Load only first 10,000 codes for cloud
   icd_codes = icd_codes[:10000]
   ```

2. Or upgrade to a paid Streamlit plan for more resources

---

### Issue 4: Slow First Load

**Expected Behavior:** First load takes 30-60 seconds

**Why:**
- Downloading sentence-transformers model (~80MB)
- Building FAISS index from 73,947 ICD-10 codes
- Loading all dependencies

**Solution:** This is normal! Subsequent loads will be faster (5-10 seconds)

---

## File Structure for Deployment

```
Medical_Code_RAG/
├── .streamlit/
│   ├── config.toml              ✓ App configuration
│   └── secrets.toml.example     ✓ Template (don't commit secrets.toml)
├── data/
│   ├── icd10_processed.csv      ✓ Must be in repo
│   └── cpt_processed.csv        ✓ Must be in repo
├── src/
│   ├── streamlit_app.py         ✓ Main entry point
│   ├── data_loader.py           ✓ Required
│   ├── retrieval.py             ✓ Required
│   ├── generation.py            ✓ Required
│   └── eval.py                  ✓ Required
├── requirements.txt             ✓ Dependencies
├── .gitignore                   ✓ Excludes unnecessary files
└── README.md                    ✓ Documentation
```

**Note:** The `faiss_index/` folder will be rebuilt on first run in the cloud.

---

## Environment Variables

Your app uses these environment variables (loaded from Streamlit secrets):

| Variable | Purpose | Required | Default |
|----------|---------|----------|---------|
| `OPENAI_API_KEY` | Perplexity API key | ✓ Yes | None |
| `OPENAI_API_BASE` | API base URL | ✓ Yes | `https://api.perplexity.ai` |
| `LLM_MODEL` | Model to use | No | `sonar-pro` |

---

## Cost Considerations

### Streamlit Cloud (Free Tier)
- ✓ 1 GB RAM
- ✓ 1 CPU core
- ✓ Unlimited public apps
- ✓ Sufficient for this project!

### Perplexity API
- **sonar-pro:** ~$1 per 1M tokens
- **Estimated cost:** $0.01-0.05 per query
- **Recommended:** Start with $5-10 credit

**Cost-saving tips:**
1. Use caching (already implemented)
2. Limit number of retrieved documents
3. Use shorter prompts where possible

---

## Updating Your Deployment

### Option 1: Git Push (Recommended)

```bash
# Make changes locally
git add .
git commit -m "Update feature X"
git push

# Streamlit Cloud auto-deploys within 1-2 minutes!
```

### Option 2: Direct Edit on GitHub

1. Edit files directly on GitHub
2. Commit changes
3. Streamlit auto-deploys

### Option 3: Reboot from Dashboard

If you only changed secrets:
1. Go to Streamlit Cloud dashboard
2. Click on your app
3. Click "Reboot" (⚙️ → Reboot)

---

## Advanced: Custom Domain (Optional)

### Using Your Own Domain

1. Go to app settings
2. Click "Settings" → "General"
3. Under "Custom domain", enter your domain
4. Follow DNS instructions
5. Click "Save"

**Note:** Custom domains available on paid plans only

---

## Monitoring & Logs

### View Logs

1. Go to [https://share.streamlit.io/](https://share.streamlit.io/)
2. Click on your app
3. Click "Manage app"
4. View logs in real-time

### Check App Health

- **Green indicator:** App is running
- **Red indicator:** App crashed (check logs)
- **Yellow indicator:** App is rebooting

---

## Security Best Practices

### ✓ Do's
- ✓ Use Streamlit secrets for API keys
- ✓ Keep repository public (for free tier)
- ✓ Use `.gitignore` to exclude sensitive files
- ✓ Rotate API keys regularly

### ✗ Don'ts
- ✗ Never commit `.env` or `secrets.toml`
- ✗ Never hardcode API keys in code
- ✗ Never commit large models or data (> 100MB)
- ✗ Never expose admin features publicly

---

## Performance Optimization

### Already Implemented ✓
- ✓ FAISS index caching (5-sec startup)
- ✓ Session state persistence
- ✓ Lazy loading of embeddings
- ✓ Efficient chunking strategy

### Additional Optimizations (Optional)

1. **Reduce Dataset Size:**
   ```python
   # In data_loader.py
   MAX_CODES = 10000  # Instead of 73,947
   ```

2. **Use Smaller Model:**
   ```python
   # In retrieval.py
   model_name = 'all-MiniLM-L6-v2'  # Already using smallest!
   ```

3. **Cache Retrieved Results:**
   ```python
   @st.cache_data(ttl=3600)
   def cached_retrieve(query):
       return retriever.retrieve(query)
   ```

---

## Next Steps After Deployment

1. **Test thoroughly:**
   - Try 10-20 different medical queries
   - Check all features work correctly
   - Verify API costs are reasonable

2. **Share your app:**
   - Copy the app URL
   - Share with colleagues/users
   - Gather feedback

3. **Monitor usage:**
   - Check Perplexity API usage dashboard
   - Monitor Streamlit app analytics
   - Watch for errors in logs

4. **Iterate and improve:**
   - Add new features
   - Fix bugs
   - Optimize based on usage patterns

---

## Quick Deployment Checklist

- [ ] Git repository initialized
- [ ] All files committed
- [ ] GitHub repository created
- [ ] Code pushed to GitHub
- [ ] Streamlit Cloud account set up
- [ ] App created on Streamlit Cloud
- [ ] Secrets added (Perplexity API key)
- [ ] App deployed successfully
- [ ] Test query works
- [ ] Shared app URL with team

---

## Support & Resources

- **Streamlit Docs:** [https://docs.streamlit.io/](https://docs.streamlit.io/)
- **Deployment Guide:** [https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
- **Perplexity API:** [https://docs.perplexity.ai/](https://docs.perplexity.ai/)
- **Your Project README:** See `README.md` for technical details

---

## Estimated Timeline

| Step | Time |
|------|------|
| Prepare repository | 5 minutes |
| Create GitHub repo | 2 minutes |
| Push to GitHub | 2 minutes |
| Configure Streamlit Cloud | 5 minutes |
| First deployment | 10-15 minutes |
| Testing | 5-10 minutes |
| **Total** | **~30-40 minutes** |

---

## Success! 🎉

Once deployed, your Medical RAG System will be:
- ✓ Publicly accessible via unique URL
- ✓ Auto-deploying on every Git push
- ✓ Running 24/7 on Streamlit's infrastructure
- ✓ Scalable for multiple users
- ✓ Professional and shareable

**Your app URL will look like:**
`https://medical-code-rag-yourname.streamlit.app`

---

**Prepared by:** Claude Agent
**Date:** October 23, 2025
**Status:** Ready for Deployment ✓
