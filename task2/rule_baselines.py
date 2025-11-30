#!/usr/bin/env python3
"""
Rule-based Baseline Methods for Text Classification
Milestone 2: Pre-ChatGPT vs Post-ChatGPT Classification

This script implements rule-based baseline methods that can be compared
directly with ML methods using the same test set.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from load_processed_data import prepare_for_classification, split_train_test

# Import lexical diversity computation from ml_baselines
try:
    from task2.ml_baselines import compute_lexical_diversity_features_batch
    LEXICAL_AVAILABLE = True
except ImportError:
    LEXICAL_AVAILABLE = False
    print("[WARNING] Could not import lexical diversity functions. Lexical diversity rule will be skipped.")


def save_predictions(
    method_name: str,
    text_ids: List[str],
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    predicted_proba: np.ndarray,
    languages: List[str],
    dates: List[str],
    output_dir: Path
):
    """
    Save predictions in the same format as ML methods for comparison.
    
    Args:
        method_name: Name of the method
        text_ids: List of text IDs
        true_labels: True labels (0 or 1)
        predicted_labels: Predicted labels (0 or 1)
        predicted_proba: Prediction probabilities (N x 2 array)
        languages: List of languages
        dates: List of publication dates
        output_dir: Output directory
    """
    predictions_data = {
        'text_id': text_ids,
        'true_label': true_labels.tolist() if isinstance(true_labels, np.ndarray) else true_labels,
        'predicted_label': predicted_labels.tolist() if isinstance(predicted_labels, np.ndarray) else predicted_labels,
        'predicted_proba': predicted_proba.tolist() if isinstance(predicted_proba, np.ndarray) else predicted_proba,
        'language': languages,
        'publication_date': dates
    }
    
    method_name_clean = method_name.replace(' ', '_').replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(':', '').lower()
    predictions_file = output_dir / f"predictions_{method_name_clean}.json"
    
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Saved predictions to {predictions_file}")


def rule_based_date_threshold(
    dates: List[str],
    threshold_date: str = "2022-11-01"
) -> np.ndarray:
    """
    Simple rule: classify based on publication date.
    Before threshold = pre-ChatGPT (0), after = post-ChatGPT (1)
    """
    threshold = datetime.strptime(threshold_date, "%Y-%m-%d")
    predictions = []
    
    for date_str in dates:
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            if date < threshold:
                predictions.append(0)  # pre-ChatGPT
            else:
                predictions.append(1)  # post-ChatGPT
        except:
            # If date parsing fails, default to post-ChatGPT
            predictions.append(1)
    
    return np.array(predictions)


def rule_based_lexical_diversity(
    lexical_features: pd.DataFrame,
    threshold_ttr: float = 0.5
) -> np.ndarray:
    """
    Rule: classify based on lexical diversity (TTR threshold).
    High TTR = post-ChatGPT (1), Low TTR = pre-ChatGPT (0)
    """
    if lexical_features is None or 'ttr' not in lexical_features.columns:
        raise ValueError("Lexical features with TTR required")
    
    predictions = (lexical_features['ttr'].values > threshold_ttr).astype(int)
    return predictions


def rule_based_text_length(
    texts: List[str],
    threshold_length: int = 500
) -> np.ndarray:
    """
    Rule: classify based on text length.
    Longer texts = post-ChatGPT (1), Shorter = pre-ChatGPT (0)
    """
    predictions = []
    for text in texts:
        if len(text) > threshold_length:
            predictions.append(1)  # post-ChatGPT
        else:
            predictions.append(0)  # pre-ChatGPT
    
    return np.array(predictions)


def evaluate_rule_based_method(
    method_name: str,
    predictions: np.ndarray,
    true_labels: np.ndarray,
    text_ids: List[str],
    languages: List[str],
    dates: List[str],
    output_dir: Path
) -> Dict:
    """Evaluate a rule-based method and save predictions."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix
    )
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    
    precision_per_class = precision_score(true_labels, predictions, average=None, zero_division=0)
    recall_per_class = recall_score(true_labels, predictions, average=None, zero_division=0)
    f1_per_class = f1_score(true_labels, predictions, average=None, zero_division=0)
    
    cm = confusion_matrix(true_labels, predictions)
    
    # Create dummy probabilities (rule-based methods don't have probabilities)
    proba = np.zeros((len(predictions), 2))
    proba[np.arange(len(predictions)), predictions] = 1.0
    
    # Save predictions
    save_predictions(
        method_name,
        text_ids,
        true_labels,
        predictions,
        proba,
        languages,
        dates,
        output_dir
    )
    
    results = {
        'classifier': method_name,
        'accuracy': float(accuracy),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'class_names': ['pre-chatgpt', 'post-chatgpt']
    }
    
    # Print results
    print(f"\nEvaluating {method_name}...")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (weighted): {precision:.4f}")
    print(f"  Recall (weighted): {recall:.4f}")
    print(f"  F1 (weighted): {f1:.4f}")
    print(f"\n  Per-class metrics:")
    for i, class_name in enumerate(['pre-chatgpt', 'post-chatgpt']):
        print(f"    {class_name}:")
        print(f"      Precision: {precision_per_class[i]:.4f}")
        print(f"      Recall: {recall_per_class[i]:.4f}")
        print(f"      F1: {f1_per_class[i]:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    {cm}")
    
    return results


def main():
    """Main function to run rule-based baseline methods."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Rule-based Baseline Methods for Pre-ChatGPT vs Post-ChatGPT Classification')
    parser.add_argument('--test', action='store_true', help='Enable test mode (use subset of data)')
    parser.add_argument('--test-percent', type=float, default=5.0, help='Percentage of data to use in test mode (default: 5.0)')
    parser.add_argument('--no-test', action='store_true', help='Disable test mode (use full dataset)')
    args = parser.parse_args()
    
    # Determine TEST_MODE
    if args.no_test:
        TEST_MODE = False
    elif args.test:
        TEST_MODE = True
    else:
        # Default: use test mode
        TEST_MODE = True
    
    TEST_PERCENT = args.test_percent
    
    print("=" * 80)
    print("Rule-based Baseline Methods for Pre-ChatGPT vs Post-ChatGPT Classification")
    print("=" * 80)
    if TEST_MODE:
        print(f"[CONFIG] Test mode: ON ({TEST_PERCENT}% of data)")
    else:
        print("[CONFIG] Test mode: OFF (full dataset)")
    print("=" * 80)
    
    # Load data (use same split as ML methods)
    data_dir = Path('processed_data')
    if not (data_dir / 'processed_data.csv').exists():
        print(f"Error: Processed data not found in {data_dir}")
        return
    
    print("\n1. Loading data...")
    features_df, labels = prepare_for_classification(data_dir, min_length=20)
    
    # For testing: use subset of data if TEST_MODE is enabled
    if TEST_MODE:
        print(f"[TEST MODE] Using {TEST_PERCENT}% of data for faster testing...")
        divisor = int(100 / TEST_PERCENT)
        sample_size = max(1, len(features_df) // divisor)
        sampled_indices = features_df.sample(n=sample_size, random_state=42).index
        features_df = features_df.loc[sampled_indices].reset_index(drop=True)
        labels = labels.loc[sampled_indices].reset_index(drop=True)
    
    print(f"   Loaded {len(features_df)} articles")
    
    # Use same split as ML methods (random_state=42, test_size=0.2)
    print("\n2. Splitting data (same split as ML methods)...")
    X_train, X_test, y_train, y_test = split_train_test(
        features_df, labels, test_size=0.2, random_state=42
    )
    print(f"   Test set: {len(X_test)} articles")
    
    # Prepare test data
    X_test_text = X_test['raw_text'].tolist()
    X_test_ids = X_test['text_id'].tolist()
    X_test_languages = X_test['language'].tolist()
    X_test_dates = X_test['publication_date'].tolist()
    
    # Compute lexical features for test set (needed for lexical diversity rule)
    print("\n3. Computing lexical diversity features for test set...")
    X_test_lexical = None
    if LEXICAL_AVAILABLE:
        X_test_lexical = compute_lexical_diversity_features_batch(X_test, use_lexical_features=True)
        if X_test_lexical is not None:
            print(f"   [INFO] Computed lexical diversity features for {len(X_test_lexical)} articles")
        else:
            print("   [INFO] Lexical diversity features not available")
    else:
        print("   [INFO] Skipping lexical diversity features (not available)")
    
    output_dir = Path('task2/results')
    output_dir.mkdir(exist_ok=True)
    
    # Run rule-based methods
    print("\n4. Running rule-based methods...")
    all_results = []
    
    # Method 1: Date-based threshold
    print("\nMethod 1: Date-based threshold (2022-11-01)")
    date_predictions = rule_based_date_threshold(X_test_dates)
    results = evaluate_rule_based_method(
        "Rule-based: Date Threshold",
        date_predictions,
        y_test.values,
        X_test_ids,
        X_test_languages,
        X_test_dates,
        output_dir
    )
    all_results.append(results)
    
    # Method 2: Text length threshold
    print("\nMethod 2: Text length threshold (500 chars)")
    length_predictions = rule_based_text_length(X_test_text, threshold_length=500)
    results = evaluate_rule_based_method(
        "Rule-based: Text Length",
        length_predictions,
        y_test.values,
        X_test_ids,
        X_test_languages,
        X_test_dates,
        output_dir
    )
    all_results.append(results)
    
    # Method 3: Lexical diversity (if features available)
    if X_test_lexical is not None and 'ttr' in X_test_lexical.columns:
        print("\nMethod 3: Lexical diversity threshold (TTR > 0.5)")
        try:
            lexical_predictions = rule_based_lexical_diversity(X_test_lexical, threshold_ttr=0.5)
            results = evaluate_rule_based_method(
                "Rule-based: Lexical Diversity",
                lexical_predictions,
                y_test.values,
                X_test_ids,
                X_test_languages,
                X_test_dates,
                output_dir
            )
            all_results.append(results)
        except Exception as e:
            print(f"   [WARNING] Lexical diversity rule failed: {e}")
    else:
        print("\nMethod 3: Lexical diversity - SKIPPED (features not available)")
    
    # Save results
    print("\n5. Saving results...")
    results_file = output_dir / 'rule_baseline_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"   [INFO] Saved results to {results_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY OF ALL RULE-BASED METHODS")
    print("=" * 80)
    print(f"{'Method':<40} {'Accuracy':<12} {'F1 (weighted)':<15}")
    print("-" * 80)
    for result in all_results:
        method_name = result['classifier']
        accuracy = result['accuracy']
        f1_weighted = result['f1_weighted']
        print(f"{method_name:<40} {accuracy:<12.4f} {f1_weighted:<15.4f}")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("Rule-based Baseline Evaluation Complete!")
    print("=" * 80)
    print("\nRun compare_methods.py to compare with ML methods.")


if __name__ == "__main__":
    main()

