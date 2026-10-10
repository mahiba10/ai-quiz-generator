# Setup, Screenshots, GitHub, Deployment and Submission Guide

## A. Run locally
1. Install Python 3.10+ from python.org (tick "Add Python to PATH" on Windows). Check: `python --version`.
2. Open a terminal in the project folder, then:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows   (macOS/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   python ml/train_model.py      # trains the model; prints real metrics
   streamlit run app.py
   ```
3. Optional tests: `pip install -r requirements-dev.txt` then `pytest`.

## B. Screenshots (capture only from your running app)
Windows: `Win + Shift + S`; macOS: `Cmd + Shift + 4`. Save into `assets/`.
1. Home page. 2. Start a Quiz page (topic cards + difficulty). 3. A quiz question in progress. 4. Results page (score + recommendations).
5. Topic Finder with a prediction shown. 6. Topic Finder → "Model evaluation" tab.
Combine into one PDF: paste the images into a Word/Google Docs file (one per page, with a caption) and export as PDF; or on Windows use "Microsoft Print to PDF" on selected images. Never use mock-up screenshots.

## C. GitHub
1. Create a free account and a new **public** repository named `ai-quiz-generator` (no README, since you already have one).
2. In the project folder:
   ```bash
   git init
   git add .
   git commit -m "Add question bank and quiz logic"
   ```
   Make several meaningful commits, e.g. "Add topic classifier and training script", "Add Streamlit interface", "Add tests", "Add documentation".
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/ai-quiz-generator.git
   git push -u origin main
   ```
3. Secret check: there are no API keys in this project. Make sure `.env` and `.streamlit/secrets.toml` are not committed (they are in `.gitignore`) and search your files for "key", "token", "password".
4. Commit `models/topic_classifier.joblib` and `models/metrics.json` after training (remove the commented lines in `.gitignore` if you ignore them).

## D. Deploy on Streamlit Community Cloud
1. Go to share.streamlit.io and sign in with GitHub.
2. Click **Create app** → pick your repo, branch `main`, **Main file path: `app.py`**.
3. Dependencies come from `requirements.txt` automatically. Click **Deploy**.
4. Paths are built from `__file__`, so they work after deployment. If you did not commit the model, open **Topic Finder** and press "Train the model now".
5. Test every page on the live URL (also on a phone). Add the URL to the README **only after it works**.

## E. Final submission checklist
- [ ] Project link (verified GitHub link and live app link)
- [ ] Concept note (PDF of `docs/concept_note.md`, with real results filled in)
- [ ] Final deliverable PPT (from `docs/presentation_outline.md`, with real screenshots)
- [ ] Final project screenshots (single PDF)
- [ ] Lean Canvas PDF (from `docs/lean_canvas.md`, in your internship template)
- [ ] Name placeholders replaced in README, concept note and slides
