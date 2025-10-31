"""
Preprocessing pipeline for Leipzig Corpora Collection news datasets.
Extracts, cleans, and formats data for Milestone 1 of NLP IE 2025WS project.

This script processes Leipzig Corpora datasets and prepares them in standard format
with pre-ChatGPT and post-ChatGPT classification.
"""

import tarfile
import csv
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import re

# For tokenization
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    print("Warning: NLTK not available. Basic tokenization will be used.")
    nltk = None

# Try to use spaCy for better tokenization (optional)
try:
    import spacy
    from spacy.lang.cs import Czech as SpacyCzech
    from spacy.lang.de import German as SpacyGerman
    
    # Try to load English model
    try:
        nlp_en = spacy.load('en_core_web_sm')
        SPACY_EN_AVAILABLE = True
    except OSError:
        # Create blank English if model not available
        nlp_en = spacy.blank('en')
        SPACY_EN_AVAILABLE = False
    
    # Create blank Czech (no trained model available, but still better tokenization)
    nlp_cs = SpacyCzech()
    SPACY_CS_AVAILABLE = True

    # Try to load German model
    try:
        nlp_de = spacy.load('de_core_news_sm')
        SPACY_DE_AVAILABLE = True
    except OSError:
        # Create blank German if model not available
        nlp_de = SpacyGerman()
        SPACY_DE_AVAILABLE = False

    SPACY_AVAILABLE = True
    
except ImportError:
    SPACY_AVAILABLE = False
    SPACY_EN_AVAILABLE = False
    SPACY_CS_AVAILABLE = False
    SPACY_DE_AVAILABLE = False
    spacy = None


# ChatGPT launch date: November 1, 2022
CHATGPT_LAUNCH_DATE = datetime(2022, 11, 1)


def extract_tar_file(tar_path: str, extract_to: str) -> Dict[str, str]:
    """
    Extract Leipzig Corpora tar.gz file and return paths to key files.
    
    Args:
        tar_path: Path to .tar.gz file
        extract_to: Directory to extract to
        
    Returns:
        Dictionary with paths to sentences.txt, sources.txt, and co_s.txt
    """
    os.makedirs(extract_to, exist_ok=True)
    
    file_paths = {}
    with tarfile.open(tar_path, 'r:gz') as tar:
        # Extract all files
        tar.extractall(extract_to)
        
        # Find key files
        for member in tar.getmembers():
            if 'sentences.txt' in member.name and not member.name.endswith('.txt.bak'):
                file_paths['sentences'] = os.path.join(extract_to, member.name)
            elif 'sources.txt' in member.name and not member.name.endswith('.txt.bak'):
                file_paths['sources'] = os.path.join(extract_to, member.name)
            elif 'co_s.txt' in member.name and not member.name.endswith('.txt.bak'):
                file_paths['co_s'] = os.path.join(extract_to, member.name)
    
    return file_paths


def parse_sentences_file(sentences_path: str) -> Dict[int, str]:
    """
    Parse sentences.txt file from Leipzig Corpora.
    Format: ID \t sentence_text
    
    Args:
        sentences_path: Path to sentences.txt
        
    Returns:
        Dictionary mapping sentence ID to sentence text
    """
    sentences = {}
    
    with open(sentences_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t', 1)
            if len(parts) == 2:
                sent_id = int(parts[0])
                sent_text = parts[1]
                sentences[sent_id] = sent_text
    
    return sentences


def parse_sources_file(sources_path: str) -> Dict[int, Dict]:
    """
    Parse sources.txt file from Leipzig Corpora.
    Format: ID \t URL \t date
    
    Args:
        sources_path: Path to sources.txt
        
    Returns:
        Dictionary mapping source ID to {'url': ..., 'date': ...}
    """
    sources = {}
    
    with open(sources_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 3:
                source_id = int(parts[0])
                url = parts[1]
                date_str = parts[2].strip()
                
                # Parse date
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                except ValueError:
                    # Try alternative date formats
                    try:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        print(f"Warning: Could not parse date '{date_str}', skipping source {source_id}")
                        continue

                # Filter out obviously invalid years to avoid corrupted stats/outputs
                if date_obj.year < 1990 or date_obj.year > 2030:
                    # Skip pathological dates like year 11 or 5020 sometimes found in scraped sources
                    continue
                
                sources[source_id] = {
                    'url': url,
                    'date': date_obj,
                    'date_str': date_str
                }
    
    return sources


def parse_co_s_file(co_s_path: str) -> Dict[int, int]:
    """
    Parse co_s.txt file that maps sentences to sources.
    Format: sentence_id \t source_id \t frequency \t score
    
    Args:
        co_s_path: Path to co_s.txt
        
    Returns:
        Dictionary mapping sentence ID to source ID
    """
    sentence_to_source = {}
    
    try:
        with open(co_s_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    sentence_id = int(parts[0])
                    source_id = int(parts[1])
                    sentence_to_source[sentence_id] = source_id
    except FileNotFoundError:
        print(f"Warning: co_s.txt not found, sentence IDs will be used as source IDs")
    
    return sentence_to_source


def classify_period(date: datetime) -> str:
    """
    Classify date as pre-ChatGPT or post-ChatGPT.
    
    Args:
        date: Publication date
        
    Returns:
        'pre-chatgpt' or 'post-chatgpt'
    """
    if date < CHATGPT_LAUNCH_DATE:
        return 'pre-chatgpt'
    else:
        return 'post-chatgpt'


def is_valid_token(token: str) -> bool:
    """
    Check if token is valid (not emoji, symbol, or special character).
    
    Args:
        token: Token to check
        
    Returns:
        True if token is valid word/alphanumeric
    """
    # Remove leading/trailing whitespace
    token = token.strip()
    
    # Must contain at least one alphanumeric character
    if not re.search(r'[a-zA-Z0-9]', token):
        return False
    
    # Filter out tokens that are mostly symbols/emojis
    # Allow alphanumeric, some punctuation like apostrophes/hyphens within words
    # But filter out pure emoji/symbol tokens
    if re.match(r'^[\W_]+$', token):  # Only symbols/emoji
        return False
    
    return True


def tokenize_text(text: str, language: str = 'english') -> List[str]:
    """
    Tokenize text using appropriate tokenizer for the language.
    Prioritizes spaCy if available, falls back to NLTK, then regex.
    Filters out emojis and symbols.
    
    Args:
        text: Text to tokenize
        language: Language code ('english' or 'czech')
        
    Returns:
        List of tokens (lowercased, emoji/symbol filtered)
    """
    # Use spaCy if available
    if SPACY_AVAILABLE and spacy:
        try:
            if language.lower() == 'english':
                # Use pre-initialized English model (either loaded or blank)
                doc = nlp_en(text)
                tokens = [token.text.lower() for token in doc 
                         if not token.is_space and not token.is_punct]
            elif language.lower() == 'czech':
                # Use pre-initialized Czech model (blank, but better than NLTK)
                doc = nlp_cs(text)
                tokens = [token.text.lower() for token in doc 
                         if not token.is_space and not token.is_punct]
            elif language.lower() == 'german':
                # Use pre-initialized German model (either loaded or blank)
                doc = nlp_de(text)
                tokens = [token.text.lower() for token in doc
                          if not token.is_space and not token.is_punct]
                    
            else:
                # For other languages, fall back to NLTK
                tokens = []
            
            # Filter out emojis and invalid tokens
            tokens = [t for t in tokens if is_valid_token(t)]
            
            if tokens:
                return tokens
        except Exception as e:
            # If spaCy fails, fall through to NLTK
            pass
    
    # Fallback to NLTK
    if nltk:
        try:
            # NLTK language mapping
            if language.lower() == 'german':
                nltk_lang = 'german'
            elif language.lower() == 'czech':
                nltk_lang = 'czech'
            else:
                nltk_lang = 'english'

            tokens = word_tokenize(text, language=nltk_lang)
            # Filter and lowercase - keep only alphanumeric tokens
            tokens = [t.lower() for t in tokens if is_valid_token(t)]
            return tokens
        except Exception:
            pass

    # Simple regex fallback
    tokens = re.findall(r'\b\w+\b', text.lower())
    tokens = [t for t in tokens if is_valid_token(t)]
    return tokens


def clean_text(text: str) -> str:
    """
    Basic text cleaning: normalize whitespace and remove artifacts.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def process_dataset(tar_path: str, language: str, output_dir: str) -> Tuple[List[Dict], int, int]:
    """
    Process a single Leipzig Corpora dataset.
    
    Args:
        tar_path: Path to .tar.gz file
        language: Language code ('english' or 'czech')
        output_dir: Directory for extracted files
        
    Returns:
        Tuple of (processed_texts, pre_count, post_count)
    """
    print(f"Processing {tar_path}...")
    
    # Extract tar file
    extract_dir = os.path.join(output_dir, f"extracted_{os.path.basename(tar_path).replace('.tar.gz', '')}")
    file_paths = extract_tar_file(tar_path, extract_dir)
    
    if 'sentences' not in file_paths or 'sources' not in file_paths:
        raise ValueError(f"Could not find required files in {tar_path}")
    
    # Parse files
    print("  Parsing sentences...")
    sentences = parse_sentences_file(file_paths['sentences'])
    print(f"  Found {len(sentences)} sentences")
    
    print("  Parsing sources...")
    sources = parse_sources_file(file_paths['sources'])
    print(f"  Found {len(sources)} sources")
    
    # Parse sentence-source mapping
    print("  Parsing sentence-source mapping...")
    if 'co_s' in file_paths:
        sentence_to_source = parse_co_s_file(file_paths['co_s'])
        print(f"  Found {len(sentence_to_source)} sentence-source mappings")
    else:
        # Fallback: use sentence ID as source ID
        sentence_to_source = {sent_id: sent_id for sent_id in sentences.keys()}
        print("  No co_s file found, using sentence IDs as source IDs")
    
    # Group sentences by source
    print("  Grouping sentences by source...")
    source_sentences = {}
    for sent_id, sent_text in sentences.items():
        # Get source ID from mapping, or use sentence ID as fallback
        source_id = sentence_to_source.get(sent_id, sent_id)
        if source_id not in source_sentences:
            source_sentences[source_id] = []
        source_sentences[source_id].append(sent_text)
    
    # Process each source (article)
    processed_texts = []
    pre_count = 0
    post_count = 0
    
    print("  Processing articles...")
    for source_id, sent_list in source_sentences.items():
        if source_id not in sources:
            continue
        
        source_info = sources[source_id]
        date = source_info['date']
        period = classify_period(date)
        
        # Combine sentences into article
        raw_text = ' '.join(sent_list)
        cleaned_text = clean_text(raw_text)
        
        # Skip very short texts
        if len(cleaned_text) < 100:  # Minimum 100 characters
            continue
        
        # Tokenize
        tokens = tokenize_text(cleaned_text, language)
        
        if len(tokens) < 20:  # Minimum 20 tokens
            continue
        
        # Create processed entry
        text_entry = {
            'text_id': f"{language}_{date.year}_{source_id}",
            'language': language,
            'publication_date': source_info['date_str'],
            'year': date.year,
            'month': date.month,
            'period': period,
            'raw_text': cleaned_text,
            'tokens': ' '.join(tokens),
            'token_count': len(tokens),
            'word_count': len(cleaned_text.split()),  # Approximate word count
            'char_count': len(cleaned_text),
            'url': source_info['url'],
            'dataset_source': os.path.basename(tar_path)
        }
        
        processed_texts.append(text_entry)
        
        if period == 'pre-chatgpt':
            pre_count += 1
        else:
            post_count += 1
        
        if len(processed_texts) % 1000 == 0:
            print(f"    Processed {len(processed_texts)} articles...")
    
    print(f"  Processed {len(processed_texts)} articles")
    print(f"    Pre-ChatGPT: {pre_count}, Post-ChatGPT: {post_count}")
    
    return processed_texts, pre_count, post_count


def save_to_csv(processed_texts: List[Dict], output_path: str):
    """Save processed texts to CSV file."""
    if not processed_texts:
        print(f"Warning: No data to save to {output_path}")
        return
    
    fieldnames = processed_texts[0].keys()
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_texts)
    
    print(f"Saved {len(processed_texts)} entries to {output_path}")


def save_to_json(processed_texts: List[Dict], output_path: str):
    """Save processed texts to JSON file."""
    if not processed_texts:
        print(f"Warning: No data to save to {output_path}")
        return
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_texts, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(processed_texts)} entries to {output_path}")


def generate_statistics(all_texts: List[Dict], output_path: str):
    """Generate dataset statistics."""
    stats = {
        'total_articles': len(all_texts),
        'by_language': {},
        'by_period': {'pre-chatgpt': 0, 'post-chatgpt': 0},
        'by_year': {},
        'date_range': {
            'earliest': None,
            'latest': None
        },
        'token_statistics': {
            'mean': 0,
            'median': 0,
            'min': float('inf'),
            'max': 0
        }
    }
    
    token_counts = []
    dates = []
    
    for text in all_texts:
        # Language
        lang = text['language']
        stats['by_language'][lang] = stats['by_language'].get(lang, 0) + 1
        
        # Period
        period = text['period']
        stats['by_period'][period] += 1
        
        # Year (guard against malformed years)
        year = text['year']
        if 1990 <= int(year) <= 2030:
            stats['by_year'][year] = stats['by_year'].get(year, 0) + 1
        
        # Date
        date_str = text['publication_date']
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            dates.append(date_obj)
        except:
            pass
        
        # Tokens
        token_count = text['token_count']
        token_counts.append(token_count)
    
    # Calculate token statistics
    if token_counts:
        import statistics
        stats['token_statistics']['mean'] = statistics.mean(token_counts)
        stats['token_statistics']['median'] = statistics.median(token_counts)
        stats['token_statistics']['min'] = min(token_counts)
        stats['token_statistics']['max'] = max(token_counts)
    
    # Date range
    if dates:
        stats['date_range']['earliest'] = min(dates).strftime('%Y-%m-%d')
        stats['date_range']['latest'] = max(dates).strftime('%Y-%m-%d')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Statistics saved to {output_path}")
    return stats


def find_dataset_files(datasets_dir: Path, language: str) -> List[Path]:
    """
    Auto-detect ALL dataset files for a language.
    Looks for files matching: {lang}_news_YYYY_*.tar.gz (multiple years allowed).

    Note: We don't pre-split by year here. Articles are classified later using
    their exact publication_date, so 2022 datasets will be split correctly
    around November 1, 2022.

    Args:
        datasets_dir: Root datasets directory
        language: Language name ('english' or 'czech')

    Returns:
        Sorted list of tar.gz paths for the language (can be multiple years)
    """
    lang_dir = datasets_dir / language
    if not lang_dir.exists():
        return []

    # Language prefix mapping
    lang_prefixes = {
        'english': 'eng',
        'czech': 'ces',
        'german' : 'deu',
    }
    prefix = lang_prefixes.get(language, language[:3])

    tar_files: List[Path] = list(lang_dir.glob(f'{prefix}_news_*.tar.gz'))
    # Sort by name (which includes year and size label like 10K/1M)
    tar_files.sort()
    return tar_files


def scan_all_datasets(datasets_dir: Path) -> Dict[str, List[Path]]:
    """
    Scan datasets directory for all languages and list archives found.

    Returns mapping of language -> list of tar.gz paths.
    """
    languages = ['english', 'czech','german']
    found: Dict[str, List[Path]] = {}
    for lang in languages:
        files = find_dataset_files(datasets_dir, lang)
        if files:
            found[lang] = files
    return found

def main():
    """Main preprocessing pipeline."""
    # Configuration
    datasets_dir = Path('datasets')
    output_dir = Path('processed_data')
    output_dir.mkdir(exist_ok=True)
    
    all_processed = []
    
    # Auto-detect and process datasets for each language (multiple years allowed)
    languages = ['english', 'czech','german']

    # Pre-flight: show found datasets and prompt user
    found = scan_all_datasets(datasets_dir)
    print("\nDataset directory:", datasets_dir.resolve())
    if found:
        print("\nDatasets detected (will classify per-article by publication date):")
        for lang, paths in found.items():
            for p in paths:
                print(f"  - {lang}: {p.name}")
        print("\nSelect an action:")
        print("  [1] Process the listed datasets")
        print("  [2] Stop")
        choice = input("Enter 1 / 2: ").strip()
        if choice != '1':
            print("\nStopping without processing.")
            return
    else:
        print("\nNo datasets were found under the datasets/ directory.")
        print("Please download the datasets folder (with 'english/' , 'czech/' , 'german/' subfolders) as described in the README, then rerun this script.")
        return

    for language in languages:
        print(f"\n{'='*60}")
        print(f"Processing {language.upper()} datasets...")
        print(f"{'='*60}")
        
        # Find dataset files automatically
        dataset_files = find_dataset_files(datasets_dir, language)

        if not dataset_files:
            print(f"Warning: No dataset files found for {language}, skipping...")
            continue
        
        # Process each found dataset (any year). Period is decided per-article
        # using publication_date in classify_period().
        for tar_path in dataset_files:
            if not tar_path.exists():
                print(f"Warning: {tar_path} not found, skipping...")
                continue

            print(f"\nFound dataset: {tar_path.name}")

            processed, pre_count, post_count = process_dataset(
                str(tar_path),
                language,
                str(output_dir),
            )
            all_processed.extend(processed)
    
    # Save combined datasets
    print("\nSaving processed data...")
    
    # Save as CSV
    csv_path = output_dir / 'processed_data.csv'
    save_to_csv(all_processed, str(csv_path))
    
    # Save as JSON
    json_path = output_dir / 'processed_data.json'
    save_to_json(all_processed, str(json_path))
    
    # Save by language and period
    for language in ['english', 'czech','german']:
        for period in ['pre-chatgpt', 'post-chatgpt']:
            filtered = [t for t in all_processed 
                       if t['language'] == language and t['period'] == period]
            
            if filtered:
                csv_path = output_dir / f'{language}_{period}.csv'
                save_to_csv(filtered, str(csv_path))
    
    # Generate statistics
    stats_path = output_dir / 'dataset_statistics.json'
    stats = generate_statistics(all_processed, str(stats_path))
    
    print(f"\n{'='*60}")
    print("Preprocessing complete!")
    print(f"{'='*60}")
    print(f"Total articles processed: {len(all_processed)}")
    print(f"Pre-ChatGPT: {stats['by_period']['pre-chatgpt']}")
    print(f"Post-ChatGPT: {stats['by_period']['post-chatgpt']}")
    print(f"Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
    print(f"\nOutput directory: {output_dir}")


if __name__ == '__main__':
    main()
