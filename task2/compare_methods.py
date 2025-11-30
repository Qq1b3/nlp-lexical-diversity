#!/usr/bin/env python3
"""
Compare ML and Rule-based methods side-by-side.
Loads predictions from all methods and provides detailed comparison.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)


def load_predictions(predictions_file: Path) -> Dict:
    """Load predictions from a JSON file."""
    with open(predictions_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_predictions(
    method1_name: str,
    method1_preds: List[int],
    method2_name: str,
    method2_preds: List[int],
    true_labels: List[int]
) -> Dict:
    """Compare predictions from two methods."""
    method1_preds = np.array(method1_preds)
    method2_preds = np.array(method2_preds)
    true_labels = np.array(true_labels)
    
    # Agreement
    agreement = (method1_preds == method2_preds).sum()
    agreement_pct = agreement / len(true_labels) * 100
    
    # Where they agree but are wrong
    both_wrong = ((method1_preds != true_labels) & (method2_preds != true_labels)).sum()
    both_wrong_pct = both_wrong / len(true_labels) * 100
    
    # Where they agree and are correct
    both_correct = ((method1_preds == true_labels) & (method2_preds == true_labels)).sum()
    both_correct_pct = both_correct / len(true_labels) * 100
    
    # Where they disagree
    disagreement = (method1_preds != method2_preds).sum()
    disagreement_pct = disagreement / len(true_labels) * 100
    
    # Method 1 correct, Method 2 wrong
    m1_correct_m2_wrong = ((method1_preds == true_labels) & (method2_preds != true_labels)).sum()
    
    # Method 2 correct, Method 1 wrong
    m2_correct_m1_wrong = ((method2_preds == true_labels) & (method1_preds != true_labels)).sum()
    
    return {
        'method1': method1_name,
        'method2': method2_name,
        'total_samples': len(true_labels),
        'agreement': int(agreement),
        'agreement_pct': float(agreement_pct),
        'disagreement': int(disagreement),
        'disagreement_pct': float(disagreement_pct),
        'both_correct': int(both_correct),
        'both_correct_pct': float(both_correct_pct),
        'both_wrong': int(both_wrong),
        'both_wrong_pct': float(both_wrong_pct),
        'method1_correct_method2_wrong': int(m1_correct_m2_wrong),
        'method2_correct_method1_wrong': int(m2_correct_m1_wrong)
    }


def find_disagreement_examples(
    method1_preds: List[int],
    method2_preds: List[int],
    true_labels: List[int],
    text_ids: List[str],
    texts: List[str],
    languages: List[str],
    dates: List[str],
    n_examples: int = 10
) -> List[Dict]:
    """Find examples where methods disagree."""
    method1_preds = np.array(method1_preds)
    method2_preds = np.array(method2_preds)
    true_labels = np.array(true_labels)
    
    disagreements = []
    for idx in range(len(method1_preds)):
        if method1_preds[idx] != method2_preds[idx]:
            disagreements.append({
                'text_id': text_ids[idx],
                'true_label': 'pre-chatgpt' if true_labels[idx] == 0 else 'post-chatgpt',
                'method1_pred': 'pre-chatgpt' if method1_preds[idx] == 0 else 'post-chatgpt',
                'method2_pred': 'pre-chatgpt' if method2_preds[idx] == 0 else 'post-chatgpt',
                'text_preview': texts[idx][:200] + '...' if len(texts[idx]) > 200 else texts[idx],
                'language': languages[idx],
                'publication_date': dates[idx],
                'method1_correct': method1_preds[idx] == true_labels[idx],
                'method2_correct': method2_preds[idx] == true_labels[idx]
            })
    
    return disagreements[:n_examples]


def main():
    """Main comparison function."""
    print("=" * 80)
    print("Comparing ML and Rule-based Methods")
    print("=" * 80)
    
    results_dir = Path('task2/results')
    
    # Find all prediction files (ML methods save as predictions_*.json)
    # Rule-based methods should also save in the same format
    prediction_files = list(results_dir.glob('predictions_*.json'))
    
    if not prediction_files:
        print("No prediction files found.")
        print("For ML methods: Run ml_baselines.py first.")
        print("For rule-based methods: Ensure they save predictions in the same format.")
        return
    
    print(f"\nFound {len(prediction_files)} prediction files:")
    for pf in prediction_files:
        print(f"  - {pf.name}")
    
    # Load all predictions
    all_predictions = {}
    for pf in prediction_files:
        method_name = pf.stem.replace('predictions_', '')
        all_predictions[method_name] = load_predictions(pf)
        print(f"\nLoaded {method_name}: {len(all_predictions[method_name]['text_id'])} predictions")
    
    # Get common test set (should be the same for all methods)
    if len(all_predictions) > 0:
        first_method = list(all_predictions.keys())[0]
        common_text_ids = set(all_predictions[first_method]['text_id'])
        
        # Verify all methods have the same test set
        for method_name, preds in all_predictions.items():
            method_text_ids = set(preds['text_id'])
            if method_text_ids != common_text_ids:
                print(f"Warning: {method_name} has different test set!")
                common_text_ids = common_text_ids.intersection(method_text_ids)
        
        print(f"\nCommon test set size: {len(common_text_ids)} samples")
    
    # Pairwise comparisons
    print("\n" + "=" * 80)
    print("PAIRWISE COMPARISONS")
    print("=" * 80)
    
    method_names = list(all_predictions.keys())
    comparison_results = []
    
    for i, method1 in enumerate(method_names):
        for method2 in method_names[i+1:]:
            preds1 = all_predictions[method1]
            preds2 = all_predictions[method2]
            
            # Align by text_id
            text_id_to_idx1 = {tid: idx for idx, tid in enumerate(preds1['text_id'])}
            text_id_to_idx2 = {tid: idx for idx, tid in enumerate(preds2['text_id'])}
            
            common_ids = [tid for tid in common_text_ids if tid in text_id_to_idx1 and tid in text_id_to_idx2]
            
            if not common_ids:
                continue
            
            aligned_preds1 = [preds1['predicted_label'][text_id_to_idx1[tid]] for tid in common_ids]
            aligned_preds2 = [preds2['predicted_label'][text_id_to_idx2[tid]] for tid in common_ids]
            aligned_true = [preds1['true_label'][text_id_to_idx1[tid]] for tid in common_ids]
            
            comparison = compare_predictions(
                method1, aligned_preds1,
                method2, aligned_preds2,
                aligned_true
            )
            comparison_results.append(comparison)
            
            print(f"\n{method1} vs {method2}:")
            print(f"  Agreement: {comparison['agreement_pct']:.2f}% ({comparison['agreement']}/{comparison['total_samples']})")
            print(f"  Disagreement: {comparison['disagreement_pct']:.2f}% ({comparison['disagreement']}/{comparison['total_samples']})")
            print(f"  Both correct: {comparison['both_correct_pct']:.2f}%")
            print(f"  Both wrong: {comparison['both_wrong_pct']:.2f}%")
            print(f"  {method1} correct, {method2} wrong: {comparison['method1_correct_method2_wrong']}")
            print(f"  {method2} correct, {method1} wrong: {comparison['method2_correct_method1_wrong']}")
    
    # Save comparison results
    comparison_file = results_dir / 'method_comparisons.json'
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Saved comparison results to {comparison_file}")
    
    # Create summary table with all methods
    print("\n" + "=" * 80)
    print("SUMMARY TABLE - ALL METHODS")
    print("=" * 80)
    
    def normalize_method_name(name: str) -> str:
        """Normalize method name for comparison (remove special chars, lowercase)."""
        return name.lower().replace(' ', '_').replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(':', '').replace('+', '_').replace('-', '_').replace('device', '').replace('__', '_').strip('_')
    
    # Load quantitative results if available
    results_file = results_dir / 'ml_baseline_results.json'
    all_method_metrics = {}
    
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            ml_results = json.load(f)
            for result in ml_results:
                # Use display name as key, normalized for comparison
                normalized_key = normalize_method_name(result['classifier'])
                all_method_metrics[normalized_key] = result
    
    # Load rule-based results if available
    rule_results_file = results_dir / 'rule_baseline_results.json'
    if rule_results_file.exists():
        with open(rule_results_file, 'r', encoding='utf-8') as f:
            rule_results = json.load(f)
            for result in rule_results:
                normalized_key = normalize_method_name(result['classifier'])
                all_method_metrics[normalized_key] = result
    
    # Calculate metrics for methods that have predictions but no results file
    for method_name, preds in all_predictions.items():
        normalized_key = normalize_method_name(method_name)
        if normalized_key not in all_method_metrics:
            # Calculate metrics for this method
            true_labels = np.array(preds['true_label'])
            pred_labels = np.array(preds['predicted_label'])
            
            accuracy = accuracy_score(true_labels, pred_labels)
            precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
            recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
            f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
            
            all_method_metrics[normalized_key] = {
                'classifier': method_name,  # Keep original method name from predictions
                'accuracy': float(accuracy),
                'precision_weighted': float(precision),
                'recall_weighted': float(recall),
                'f1_weighted': float(f1)
            }
    
    # Print summary sorted by accuracy
    print(f"\n{'Method':<50} {'Accuracy':<12} {'F1 (weighted)':<15}")
    print("-" * 80)
    for result in sorted(all_method_metrics.values(), key=lambda x: x['accuracy'], reverse=True):
        method_name = result['classifier']
        print(f"{method_name:<50} {result['accuracy']:<12.4f} {result['f1_weighted']:<15.4f}")
    
    # Save combined results
    combined_results_file = results_dir / 'all_methods_comparison.json'
    with open(combined_results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': list(all_method_metrics.values()),
            'pairwise_comparisons': comparison_results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Saved combined comparison to {combined_results_file}")
    
    print("\n" + "=" * 80)
    print("Comparison Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

