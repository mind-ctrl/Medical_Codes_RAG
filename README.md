# rag_medical_poc

Healthcare RAG system for automated ICD-10-CM/CPT code lookup  
Uses hybrid retrieval (UMLS, BM25, semantic) + GPT-4o-mini for code generation.

Quickstart (local demo)
-----------------------

1. Create a Python virtual environment and install dependencies (poetry or pip):

	- With Poetry (recommended):

	  ```powershell
	  poetry install
	  ```

	- With pip (if you prefer pip): create a venv and install packages from `pyproject.toml` pins or manually.

2. Create a `.env` file in the repository root with your OpenAI API key (do NOT commit this file):

	```properties
	OPENAI_API_KEY=sk-REPLACE_ME
	```

3. Run the Streamlit demo (two options below: without Docker, or with Docker):

Run without Docker (recommended for local development)
----------------------------------------------------

1. Create and activate a Python virtual environment:

	```powershell
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
	```

2. Install dependencies with pip (requirements.txt is provided):

	```powershell
	pip install -r requirements.txt
	```

	Alternatively, use Poetry if you prefer:

	```powershell
	pip install poetry
	poetry install
	```

3. Create a local `.env` with your OpenAI key (do NOT commit it):

	```powershell
	"OPENAI_API_KEY=sk-REPLACE_ME" | Out-File -Encoding utf8 .env
	```

4. Run the app:

	```powershell
	streamlit run src/streamlit_app.py
	```

Run with Docker (optional)
--------------------------

If you prefer containerized setup, Docker configuration is included.

1. Build and start the services with docker-compose:

	```powershell
	docker-compose up --build
	```

2. The Streamlit app will be available on http://localhost:8501

Notes
-----
- This repository does not include any demo dataset. When someone downloads the project they must provide the required ICD/CPT data files in `data/` (or follow instructions to obtain them). Do NOT include demo dataset files in the repo.
- Keep `.env` local and rotate any exposed keys.
- If you encounter issues installing heavy dependencies (torch, onnxruntime, scispacy), consider installing CPU-only wheels or using Poetry to manage environment isolation.

Security note (important)
-------------------------

- The repository previously contained a committed `.env` with an OpenAI API key. Rotate that key immediately if it has been pushed to any remote.
- Never commit `.env` or any secrets. This repo includes a `.gitignore` that excludes `.env` and local virtual environments.

Data and index guidance (for a portfolio)
----------------------------------------

- This repo ships with raw ICD/CPT reference files in `ICD10-CM Code Descriptions 2025/`. Those files may be large and subject to licensing. If you want a lightweight public demo, include small processed sample CSVs in `data/sample/` and optionally a tiny prebuilt index for demonstration.
- `chroma_data/` contains a persisted ChromaDB index. Do NOT commit the full `chroma_data/` produced by building the entire dataset — it's a generated artifact and can be large. This repo's `.gitignore` excludes `chroma_data/` and `*.sqlite3`.
- For reproducibility: keep code to build the index (`src/retrieval.py` and `src/data_loader.py`) and add a script (e.g., `scripts/build_index.py`) to create the index locally. If you want to share a full index, attach it as a GitHub release asset or host in external storage.

Notebook demo
-------------

- `notebooks/demo_rag_poc.ipynb` demonstrates the end-to-end pipeline. Before sharing publicly, update its data paths to point to small sample files and clear large outputs. The notebook is a good showcase for GitHub once trimmed and documented.

Contribution & Running tests
---------------------------

- Tests are in the `tests/` folder. Run with pytest:

  ```powershell
  pytest -q
  ```

Notes
-----
- Keep the `pyproject.toml` and `poetry.lock` for reproducible dependency installs.
- If you'd like, I can prepare a small sample dataset and a prebuilt sample index for the notebook/demo (kept intentionally small so the repo remains lightweight).
