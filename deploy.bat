@echo off
REM Quick deployment script for Streamlit Cloud
echo.
echo ========================================
echo   Medical RAG - Deployment Helper
echo ========================================
echo.

REM Check if git is initialized
if not exist ".git" (
    echo [1/4] Initializing Git repository...
    git init
    git branch -M main
    echo [OK] Git initialized
) else (
    echo [1/4] Git already initialized [OK]
)

echo.
echo [2/4] Staging files for commit...
git add .
echo [OK] Files staged

echo.
echo [3/4] Committing changes...
git commit -m "Prepare for Streamlit Cloud deployment"
echo [OK] Changes committed

echo.
echo [4/4] Next steps:
echo.
echo 1. Create a GitHub repository:
echo    Go to: https://github.com/new
echo    Name: Medical_Code_RAG
echo    Type: Public
echo.
echo 2. Add remote and push:
echo    git remote add origin https://github.com/YOUR_USERNAME/Medical_Code_RAG.git
echo    git push -u origin main
echo.
echo 3. Deploy to Streamlit Cloud:
echo    Go to: https://share.streamlit.io/
echo    Click "New app"
echo    Select your repository: YOUR_USERNAME/Medical_Code_RAG
echo    Main file: src/streamlit_app.py
echo.
echo 4. Add secrets (in Advanced settings):
echo    OPENAI_API_KEY = "pplx-your-key-here"
echo    OPENAI_API_BASE = "https://api.perplexity.ai"
echo    LLM_MODEL = "sonar-pro"
echo.
echo For detailed instructions, see DEPLOYMENT_GUIDE.md
echo.
pause
