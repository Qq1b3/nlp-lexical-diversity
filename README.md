# NLP IE 2025WS – Topic 5: Lexical Diversity (Milestone 1)

Preprocessing pipeline for Leipzig Corpora news datasets (English, Czech). Extracts, cleans, tokenizes, classifies pre/post‑ChatGPT, and outputs standardized CSV/JSON plus stats.

## 1) Setup

Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Linux/Mac
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Verify
```bash
python --version
python -c "import nltk, pandas; print('deps OK')"
```

## 2) Get the datasets

If the script says no datasets were found in `datasets/`, download and place them:
- Link: https://tucloud.tuwien.ac.at/index.php/s/Kn8PBB9ypSo9ZTA
- Password: NLP-Gr0up5
- Expires: 04-Nov-2026

Unzip so you have:
```
project/
  datasets/
    czech/   ces_news_YYYY_*.tar.gz
    english/ eng_news_YYYY_*.tar.gz
```
Notes: Multiple years are allowed. Articles are classified by their publication date (2022 files are split around 2022-11-01).

## 3) Run
```bash
python preprocess_leipzig.py
```
The script lists archives and asks to Process or Stop. Outputs land in `processed_data/`.

## 4) What it does
- Extract Leipzig `.tar.gz`
- Parse `sentences.txt` and `sources.txt`
- Reconstruct articles by source
- Classify pre/post‑ChatGPT using publication date
- Clean text; filter short texts (min 100 chars, 20 tokens)
- Tokenize and save outputs; compute stats

## 5) Tokenization (current)
- Uses spaCy if available, then falls back to NLTK, then regex.
- English: spaCy `en_core_web_sm` if installed; otherwise blank English tokenizer.
- Czech: blank spaCy Czech tokenizer (no model download needed).
- Emojis/symbols are filtered from the `tokens` field.
Optional (better English):
```bash
python -m spacy download en_core_web_sm
```

## 6) Outputs
- `processed_data.csv`, `processed_data.json`
- `{language}_{period}.csv` (e.g., `english_pre-chatgpt.csv`)
- `dataset_statistics.json`

Fields per entry: `text_id, language, publication_date, year, month, period, raw_text, tokens, token_count, word_count, char_count, url, dataset_source`.

## 7) Troubleshooting
- Windows activation policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
- Ensure NLTK data:
```bash
python -c "import nltk; nltk.download('punkt')"
```
