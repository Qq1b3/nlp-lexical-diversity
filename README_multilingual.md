# NLP IE 2025WS – Topic 5: Lexical Diversity (Milestone 1)

**Multilingual Preprocessing (English, Czech, German)**  
This branch extends the main pipeline by adding full preprocessing for **German news corpora** alongside **English** and **Czech**.  
It extracts, cleans, tokenizes, and classifies articles into **pre-ChatGPT** and **post-ChatGPT** periods, outputting standardized CSV/JSON datasets and combined statistics.

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

If the script says no datasets were found in `datasets/`, download and place them manually.

### Temporary Cloud Storage:
All processed datasets (English, Czech, German) are packaged together and available here:
https://drive.google.com/drive/folders/1TmIT1Urt7ahlxtDhyConFJ7gYxrobgeE?usp=sharing

Unzip so you have:
```
  datasets/
    czech/   ces_news_YYYY_*.tar.gz
    english/ eng_news_YYYY_*.tar.gz
    german/ 
```
Notes: Multiple years are allowed. Articles are classified by their publication date (2022 files are split around 2022-11-01).

## 3) Run
```bash

python preprocess_leipzig_german.py

```
The script will:
  * Detect available corpora
  * Ask whether to process or skip
  * Save all processed outputs into processed_data/

Then merge all processed CSVs:
```bash
python merge_processed_data.py
```
## 4) What it does
- Extracts .tar.gz archives from Leipzig corpora
- Parse `sentences.txt` and `sources.txt`
- Reconstruct news articles by source
- Classifies by pre/post-ChatGPT
- Clean text; filter short texts (min 100 chars, 20 tokens)
- Saves structured data and summary stats

## 5) Tokenization (current)
- Uses spaCy if available, then falls back to NLTK, then regex.
- English: spaCy `en_core_web_sm` if installed; otherwise blank English tokenizer.
- Czech: blank spaCy Czech tokenizer (no model download needed).
- German support uses the built-in spaCy German tokenizer (de_core_news_sm)
- Emojis/symbols are filtered from the `tokens` field.
Optional (better English):
```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## 6) Outputs
- `{language}_{period}.csv`  
  (e.g., `english_pre-chatgpt.csv`, `czech_post-chatgpt.csv`, `german_pre-chatgpt.csv`)
- `processed_data_all.csv` – merged multilingual dataset  
- `dataset_statistics_combined.json` – combined summary statistics

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
