#!/usr/bin/env python3
"""
Machine Learning Baseline Methods for Text Classification
Milestone 2: Pre-ChatGPT vs Post-ChatGPT Classification

This script implements multiple ML baseline methods:
- Logistic Regression with TF-IDF
- Naive Bayes with TF-IDF
- SVM with TF-IDF
- Models with lexical diversity features
"""

import sys
import os
import argparse
# Suppress cuML verbose output BEFORE any imports
os.environ['CUML_LOG_LEVEL'] = 'ERROR'

from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
import warnings
import io
import time
import gc
from contextlib import redirect_stdout, redirect_stderr
from threading import Thread
from multiprocessing import Pool, cpu_count
warnings.filterwarnings('ignore')
# Suppress joblib resource tracker warnings (harmless but annoying)
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')

# Add parent directory to path to import load_processed_data
sys.path.insert(0, str(Path(__file__).parent.parent))
from load_processed_data import load_combined_data, prepare_for_classification, split_train_test

# ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# PyTorch for CUDA support
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
    
    # CUDA detection (will be set in main)
    def get_device():
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"[INFO] CUDA available: Using GPU ({torch.cuda.get_device_name(0)})")
            return device
        else:
            device = torch.device('cpu')
            print("[INFO] CUDA not available: Using CPU")
            return device
    
    DEVICE = None  # Will be set in main()
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None

# cuML (RAPIDS) for GPU-accelerated scikit-learn models
try:
    import logging
    # Suppress cuML logging (environment variable already set at top)
    logging.getLogger('cuml').setLevel(logging.ERROR)
    logging.getLogger('cupy').setLevel(logging.ERROR)
    logging.getLogger('raft').setLevel(logging.ERROR)
    
    import cuml
    from cuml.linear_model import LogisticRegression as cuLogisticRegression
    from cuml.naive_bayes import MultinomialNB as cuMultinomialNB
    from cuml.svm import SVC as cuSVC
    from cuml.feature_extraction.text import TfidfVectorizer as cuTfidfVectorizer
    CUML_AVAILABLE = True
    print("[INFO] cuML (RAPIDS) available: GPU-accelerated scikit-learn models enabled")
except ImportError:
    CUML_AVAILABLE = False
    print("[INFO] cuML not available: Using CPU-based scikit-learn (install with: conda install -c rapidsai -c conda-forge -c nvidia cuml)")

# Experiment tracking with Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING_AVAILABLE = True
except (ImportError, Exception):
    GPU_MONITORING_AVAILABLE = False
    pynvml = None

# CPU and RAM monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


def get_gpu_stats(device_id: int = 0) -> Dict[str, float]:
    """
    Get GPU statistics including utilization, temperature, and memory.
    
    Args:
        device_id: GPU device ID
        
    Returns:
        Dictionary with GPU stats
    """
    stats = {
        'gpu_utilization_percent': 0.0,
        'gpu_temperature_celsius': 0.0,
        'gpu_memory_used_mb': 0.0,
        'gpu_memory_total_mb': 0.0,
        'gpu_memory_allocated_mb': 0.0,
        'gpu_memory_reserved_mb': 0.0
    }
    
    if not GPU_MONITORING_AVAILABLE:
        return stats
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        
        # GPU utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        stats['gpu_utilization_percent'] = float(util.gpu)
        
        # GPU temperature
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        stats['gpu_temperature_celsius'] = float(temp)
        
        # GPU memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        stats['gpu_memory_used_mb'] = float(mem_info.used / 1024**2)
        stats['gpu_memory_total_mb'] = float(mem_info.total / 1024**2)
        
        # PyTorch memory stats if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            stats['gpu_memory_allocated_mb'] = float(torch.cuda.memory_allocated(device_id) / 1024**2)
            stats['gpu_memory_reserved_mb'] = float(torch.cuda.memory_reserved(device_id) / 1024**2)
    except Exception as e:
        # If monitoring fails, return zeros
        pass
    
    return stats


def get_cpu_ram_stats() -> Dict[str, float]:
    """
    Get CPU and RAM statistics.
    
    Returns:
        Dictionary with CPU and RAM stats
    """
    stats = {
        'cpu_utilization_percent': 0.0,
        'cpu_count': 0.0,
        'ram_used_mb': 0.0,
        'ram_total_mb': 0.0,
        'ram_percent': 0.0
    }
    
    if not PSUTIL_AVAILABLE:
        return stats
    
    try:
        # CPU stats
        stats['cpu_utilization_percent'] = float(psutil.cpu_percent(interval=0.1))
        stats['cpu_count'] = float(psutil.cpu_count())
        
        # RAM stats
        ram = psutil.virtual_memory()
        stats['ram_used_mb'] = float(ram.used / 1024**2)
        stats['ram_total_mb'] = float(ram.total / 1024**2)
        stats['ram_percent'] = float(ram.percent)
    except Exception as e:
        # If monitoring fails, return zeros
        pass
    
    return stats

# Lexical diversity
try:
    from lexicalrichness import LexicalRichness
    LEXICAL_RICHNESS_AVAILABLE = True
except ImportError:
    print("Warning: lexicalrichness not available. Install with: pip install lexicalrichness")
    LEXICAL_RICHNESS_AVAILABLE = False


def compute_lexical_diversity_features(text: str, tokens: List[str]) -> Dict[str, float]:
    """
    Compute lexical diversity features for a text.
    
    Args:
        text: Raw text string
        tokens: List of tokens
        
    Returns:
        Dictionary with lexical diversity metrics
    """
    features = {
        'ttr': 0.0,  # Type-Token Ratio
        'hdd': 0.0,  # Hypergeometric Distribution Diversity
        'mtld': 0.0,  # Measure of Textual Lexical Diversity
        'vocd': 0.0,  # Vocabulary Diversity
        'token_count': len(tokens),
        'unique_tokens': len(set(tokens)),
        'avg_token_length': 0.0
    }
    
    if len(tokens) == 0:
        return features
    
    # Basic TTR
    features['ttr'] = features['unique_tokens'] / features['token_count'] if features['token_count'] > 0 else 0.0
    
    # Average token length
    if tokens:
        features['avg_token_length'] = np.mean([len(token) for token in tokens])
    
    # Use lexicalrichness library if available
    if LEXICAL_RICHNESS_AVAILABLE and len(tokens) > 0:
        try:
            # Join tokens with space for lexicalrichness
            text_for_lex = ' '.join(tokens)
            lex = LexicalRichness(text_for_lex)
            
            # TTR (already computed, but use library version)
            features['ttr'] = lex.ttr
            
            # HD-D (Hypergeometric Distribution Diversity)
            try:
                features['hdd'] = lex.hdd(draws=42)  # Standard number of draws
            except:
                pass
            
            # MTLD (Measure of Textual Lexical Diversity)
            try:
                features['mtld'] = lex.mtld(threshold=0.72)
            except:
                pass
            
            # VocD (Vocabulary Diversity)
            try:
                features['vocd'] = lex.vocd()
            except:
                pass
        except Exception as e:
            # If lexicalrichness fails, keep basic features
            pass
    
    return features


def _process_single_article(args: Tuple) -> Dict[str, float]:
    """
    Worker function for parallel processing of lexical diversity features.
    
    Args:
        args: Tuple of (raw_text, tokens_str)
        
    Returns:
        Dictionary with lexical diversity features
    """
    raw_text, tokens_str = args
    tokens = tokens_str.split() if isinstance(tokens_str, str) else tokens_str
    return compute_lexical_diversity_features(raw_text, tokens)


def compute_lexical_diversity_features_batch(
    df: pd.DataFrame,
    use_lexical_features: bool = True,
    n_jobs: Optional[int] = None
) -> Optional[pd.DataFrame]:
    """
    Compute lexical diversity features for a batch of texts using parallel processing.
    
    Args:
        df: DataFrame with text data (must have 'raw_text' and 'tokens' columns)
        use_lexical_features: Whether to compute lexical diversity features
        n_jobs: Number of parallel workers (None = use all available CPUs, -1 = use all but 1)
        
    Returns:
        DataFrame with lexical diversity features, or None if not computed
    """
    lexical_features_df = None
    if use_lexical_features and LEXICAL_RICHNESS_AVAILABLE:
        print("Computing lexical diversity features...")
        
        # Determine number of workers
        if n_jobs is None:
            # Use all but 1 CPU to leave one for system/GPU processes
            n_jobs = max(1, cpu_count() - 1)
        elif n_jobs == -1:
            n_jobs = cpu_count()
        else:
            n_jobs = max(1, min(n_jobs, cpu_count()))
        
        print(f"  Using {n_jobs} parallel workers (out of {cpu_count()} available CPUs)...")
        
        # Prepare data for parallel processing
        # Convert DataFrame rows to list of tuples (raw_text, tokens)
        data_tuples = [
            (row['raw_text'], row['tokens'])
            for _, row in df.iterrows()
        ]
        
        # Process in parallel with progress tracking
        start_time = time.time()
        lexical_features = []
        total = len(data_tuples)
        pool = None
        
        try:
            pool = Pool(processes=n_jobs)
            # Process in chunks to show progress and allow interruption
            chunk_size = max(100, total // 20)  # Show ~20 progress updates
            for i in range(0, total, chunk_size):
                chunk = data_tuples[i:i + chunk_size]
                try:
                    chunk_results = pool.map(_process_single_article, chunk, chunksize=1)
                    lexical_features.extend(chunk_results)
                except KeyboardInterrupt:
                    print("\n[WARNING] Interrupted by user. Terminating workers...")
                    pool.terminate()
                    pool.join()
                    raise
                
                processed = min(i + chunk_size, total)
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = (total - processed) / rate if rate > 0 else 0
                print(f"  Processed {processed}/{total} articles... "
                      f"({processed/total*100:.1f}%, "
                      f"ETA: {remaining/60:.1f} min)")
            
            lexical_features_df = pd.DataFrame(lexical_features)
            total_time = time.time() - start_time
            print(f"[INFO] Computed lexical diversity features for {len(lexical_features)} articles "
                  f"in {total_time/60:.1f} minutes ({len(lexical_features)/total_time:.1f} articles/sec)")
        except KeyboardInterrupt:
            print("\n[INFO] Process interrupted. Partial results discarded.")
            return None
        finally:
            # Explicitly close and join the pool to prevent resource leaks
            if pool is not None:
                pool.close()
                pool.join()
                # Give a moment for resources to be released
                gc.collect()
    
    return lexical_features_df


class PyTorchMLP(nn.Module):
    """PyTorch MLP classifier with CUDA support."""
    
    def __init__(self, input_size: int, hidden_sizes: Tuple[int, ...] = (100, 50), num_classes: int = 2):
        super(PyTorchMLP, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class PyTorchMLPClassifier:
    """Wrapper for PyTorch MLP that works like scikit-learn."""
    
    def __init__(self, hidden_sizes: Tuple[int, ...] = (100, 50), max_epochs: int = 10, 
                 batch_size: int = 256, learning_rate: float = 0.001, device=None):
        self.hidden_sizes = hidden_sizes
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device is not None else (DEVICE if TORCH_AVAILABLE else None)
        self.model = None
        self.input_size = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the PyTorch MLP."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for PyTorchMLPClassifier")
        
        self.input_size = X.shape[1]
        self.model = PyTorchMLP(self.input_size, self.hidden_sizes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(self.max_epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"    Epoch {epoch + 1}/{self.max_epochs}, Loss: {avg_loss:.4f}")
                
                # Log GPU stats during training if wandb is available
                # (wandb_run will be passed via a class attribute if needed)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()


def _get_n_samples(X):
    """Get number of samples from array or sparse matrix."""
    if hasattr(X, 'shape'):
        return X.shape[0]
    return len(X)


class MLBaselineClassifier:
    """Wrapper class for ML baseline classifiers."""
    
    def __init__(self, name: str, model, vectorizer: Optional[TfidfVectorizer] = None):
        self.name = name
        self.model = model
        self.vectorizer = vectorizer
        self.is_fitted = False
        self.is_pytorch = isinstance(model, PyTorchMLPClassifier)
        self._needs_batched_training = False
        self._X_tfidf_sparse = None
        self._X_lexical = None
    
    def fit(self, X_text: List[str], y: np.ndarray, X_lexical: Optional[pd.DataFrame] = None, wandb_run=None):
        """Train the classifier."""
        print(f"\nTraining {self.name}...")
        
        # Vectorize text
        if self.vectorizer is None:
            # For large datasets (>100K), use CPU-based TfidfVectorizer to avoid GPU memory issues
            # For smaller datasets, use GPU-accelerated TF-IDF if cuML is available
            LARGE_DATASET_THRESHOLD = 100000
            use_gpu_tfidf = CUML_AVAILABLE and torch.cuda.is_available() and len(X_text) <= LARGE_DATASET_THRESHOLD
            
            if use_gpu_tfidf:
                self.vectorizer = cuTfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95
                )
                # cuML expects pandas Series, not list
                import pandas as pd
                X_text_series = pd.Series(X_text)
                X_tfidf = self.vectorizer.fit_transform(X_text_series)
            else:
                # Use CPU-based TfidfVectorizer for large datasets or when cuML not available
                if len(X_text) > LARGE_DATASET_THRESHOLD:
                    print(f"  [INFO] Using CPU-based TF-IDF for large dataset ({len(X_text)} samples)")
                self.vectorizer = TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    stop_words=None  # We handle multiple languages
                )
                X_tfidf = self.vectorizer.fit_transform(X_text)
        else:
            # cuML expects pandas Series, not list
            if CUML_AVAILABLE and torch.cuda.is_available() and isinstance(self.vectorizer, cuTfidfVectorizer):
                import pandas as pd
                X_text_series = pd.Series(X_text)
                X_tfidf = self.vectorizer.transform(X_text_series)
            else:
                X_tfidf = self.vectorizer.transform(X_text)
        
        # Convert CuPy arrays/sparse matrices to NumPy if needed (for scikit-learn compatibility)
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cp_sparse
            # Check if it's a CuPy array or CuPy sparse matrix
            if isinstance(X_tfidf, cp.ndarray):
                X_tfidf = cp.asnumpy(X_tfidf)
            elif isinstance(X_tfidf, cp_sparse.spmatrix):
                X_tfidf = X_tfidf.get()  # Convert CuPy sparse to scipy sparse
        except (ImportError, AttributeError):
            pass
        
        # Combine with lexical features if available
        # For large datasets, we need to process in batches to avoid memory issues
        if X_lexical is not None:
            # Check if we need batching (large dataset with lexical features)
            needs_batching = len(X_text) > 100000 and hasattr(X_tfidf, 'toarray')
            
            if needs_batching:
                # Process in batches to avoid memory issues
                # Store as a list of batches, will be combined during training
                X_combined = None  # Will be set during training in batches
                self._needs_batched_training = True
                self._X_tfidf_sparse = X_tfidf
                self._X_lexical = X_lexical
            else:
                # Small dataset: convert to dense arrays
                if hasattr(X_tfidf, 'toarray'):
                    X_tfidf_array = X_tfidf.toarray()
                else:
                    X_tfidf_array = np.asarray(X_tfidf)
                X_lexical_array = np.asarray(X_lexical.values)
                X_combined = np.hstack([X_tfidf_array, X_lexical_array])
                self._needs_batched_training = False
        else:
            if hasattr(X_tfidf, 'toarray'):
                # For large datasets without lexical features, keep sparse if possible
                if len(X_text) > 100000:
                    X_combined = X_tfidf  # Keep sparse
                else:
                    X_combined = np.asarray(X_tfidf.toarray())
            else:
                X_combined = np.asarray(X_tfidf)
            self._needs_batched_training = False
        
        # Train model (PyTorch models need dense numpy arrays)
        if self.is_pytorch:
            # Handle batched feature combination for PyTorch
            if self._needs_batched_training:
                print(f"  [PyTorch] Converting batched features to dense for {len(X_text)} samples...")
                # Process in batches and concatenate
                batch_size = 50000
                total_samples = len(X_text)
                dense_chunks = []
                
                for i in range(0, total_samples, batch_size):
                    batch_end = min(i + batch_size, total_samples)
                    batch_tfidf = self._X_tfidf_sparse[i:batch_end]
                    batch_lexical = self._X_lexical.iloc[i:batch_end]
                    
                    if hasattr(batch_tfidf, 'toarray'):
                        batch_tfidf_array = batch_tfidf.toarray()
                    else:
                        batch_tfidf_array = np.asarray(batch_tfidf)
                    batch_lexical_array = np.asarray(batch_lexical.values)
                    batch_combined = np.hstack([batch_tfidf_array, batch_lexical_array])
                    dense_chunks.append(batch_combined)
                    print(f"  [PyTorch] Converted {batch_end}/{total_samples} samples...", end='\r')
                
                X_combined = np.vstack(dense_chunks)
                print(f"  [PyTorch] Converted {total_samples} samples to dense array" + " " * 20)
            elif X_combined is not None and hasattr(X_combined, 'toarray'):
                # Convert sparse to dense for PyTorch
                X_combined = X_combined.toarray()
            
            self.model.fit(X_combined, y)
        else:
            # For SVM, show progress with elapsed time updates
            if isinstance(self.model, SVC):
                print(f"  [SVM Training] This may take a while... (training on {_get_n_samples(X_combined)} samples)")
                start_time = time.time()
                training_done = False
                
                def show_progress():
                    """Background thread to show elapsed time."""
                    while not training_done:
                        elapsed = time.time() - start_time
                        print(f"  [SVM Training] Still training... ({elapsed:.0f}s elapsed)", end='\r')
                        time.sleep(5)  # Update every 5 seconds
                
                # Start progress thread
                progress_thread = Thread(target=show_progress, daemon=True)
                progress_thread.start()
                
                # Train the model
                self.model.fit(X_combined, y)
                
                # Stop progress updates
                training_done = True
                elapsed = time.time() - start_time
                print(f"  [SVM Training] Completed in {elapsed:.1f} seconds" + " " * 30)  # Clear the line
            else:
                # For Logistic Regression, show training info
                if isinstance(self.model, (LogisticRegression, cuLogisticRegression, SGDClassifier)):
                    # For SGDClassifier on large datasets, ALWAYS use incremental learning
                    if isinstance(self.model, SGDClassifier):
                        # Get unique classes (needed for partial_fit)
                        classes = np.unique(y)
                        
                        # Handle batched feature combination if needed
                        if self._needs_batched_training:
                            # Need multiple epochs for SGDClassifier to learn properly
                            n_epochs = 100
                            total_samples = len(X_text)
                            print(f"  [Training] Using incremental learning on {total_samples} samples with batched features, {n_epochs} epochs...")
                            start_time = time.time()
                            
                            # Process in batches over multiple epochs
                            batch_size = 50000
                            
                            for epoch in range(n_epochs):
                                # Shuffle indices for each epoch
                                indices = np.random.permutation(total_samples)
                                
                                for i in range(0, total_samples, batch_size):
                                    batch_indices = indices[i:min(i + batch_size, total_samples)]
                                    batch_indices_sorted = np.sort(batch_indices)  # Sort for efficient sparse slicing
                                    
                                    batch_tfidf = self._X_tfidf_sparse[batch_indices_sorted]
                                    batch_lexical = self._X_lexical.iloc[batch_indices_sorted]
                                    
                                    if hasattr(batch_tfidf, 'toarray'):
                                        batch_tfidf_array = batch_tfidf.toarray()
                                    else:
                                        batch_tfidf_array = np.asarray(batch_tfidf)
                                    batch_lexical_array = np.asarray(batch_lexical.values)
                                    batch_X = np.hstack([batch_tfidf_array, batch_lexical_array])
                                    batch_y = y[batch_indices_sorted]
                                    
                                    self.model.partial_fit(batch_X, batch_y, classes=classes)
                                
                                elapsed = time.time() - start_time
                                print(f"  [Training] Epoch {epoch+1}/{n_epochs} completed ({elapsed:.0f}s elapsed)")
                            
                            elapsed = time.time() - start_time
                            print(f"  [Training] Completed in {elapsed:.2f}s")
                        elif _get_n_samples(X_combined) > 50000:
                            # Need multiple epochs for SGDClassifier to learn properly
                            n_epochs = 100
                            n_samples = _get_n_samples(X_combined)
                            print(f"  [Training] Using incremental learning (partial_fit) on {n_samples} samples, {n_epochs} epochs...")
                            start_time = time.time()
                            
                            # Convert sparse to dense if needed for batching
                            if hasattr(X_combined, 'toarray'):
                                X_combined = X_combined.toarray()
                            
                            # Process in batches over multiple epochs
                            batch_size = 100000
                            for epoch in range(n_epochs):
                                # Shuffle data for each epoch
                                indices = np.random.permutation(n_samples)
                                for i in range(0, n_samples, batch_size):
                                    batch_indices = indices[i:min(i + batch_size, n_samples)]
                                    batch_X = X_combined[batch_indices]
                                    batch_y = y[batch_indices]
                                    self.model.partial_fit(batch_X, batch_y, classes=classes)
                                
                                elapsed = time.time() - start_time
                                print(f"  [Training] Epoch {epoch+1}/{n_epochs} completed ({elapsed:.0f}s elapsed)")
                            
                            elapsed = time.time() - start_time
                            print(f"  [Training] Completed in {elapsed:.2f}s")
                    else:
                        print(f"  [Training] Fitting on {_get_n_samples(X_combined)} samples, {X_combined.shape[1]} features...")
                        start_time = time.time()
                        self.model.fit(X_combined, y)
                        elapsed = time.time() - start_time
                        # Try to get number of iterations if available
                        try:
                            n_iter = getattr(self.model, 'n_iter_', None)
                            if n_iter is not None:
                                print(f"  [Training] Completed in {elapsed:.2f}s ({n_iter} iterations)")
                            else:
                                print(f"  [Training] Completed in {elapsed:.2f}s")
                        except:
                            print(f"  [Training] Completed in {elapsed:.2f}s")
                # For Naive Bayes, it's fast (direct calculation)
                elif isinstance(self.model, (MultinomialNB, cuMultinomialNB)):
                    start_time = time.time()
                    classes = np.unique(y)
                    
                    # Handle batched feature combination if needed
                    if self._needs_batched_training:
                        total_samples = len(X_text)
                        print(f"  [Training] Computing probabilities on {total_samples} samples with batched features...")
                        
                        batch_size = 50000
                        for i in range(0, total_samples, batch_size):
                            batch_end = min(i + batch_size, total_samples)
                            batch_tfidf = self._X_tfidf_sparse[i:batch_end]
                            batch_lexical = self._X_lexical.iloc[i:batch_end]
                            
                            if hasattr(batch_tfidf, 'toarray'):
                                batch_tfidf_array = batch_tfidf.toarray()
                            else:
                                batch_tfidf_array = np.asarray(batch_tfidf)
                            batch_lexical_array = np.asarray(batch_lexical.values)
                            batch_X = np.hstack([batch_tfidf_array, batch_lexical_array])
                            batch_y = y[i:batch_end]
                            
                            self.model.partial_fit(batch_X, batch_y, classes=classes)
                            
                            elapsed = time.time() - start_time
                            print(f"  [Training] Processed {batch_end}/{total_samples} samples... ({elapsed:.0f}s elapsed)", end='\r')
                        
                        elapsed = time.time() - start_time
                        print(f"  [Training] Completed in {elapsed:.2f}s" + " " * 30)
                    # For CPU-based MultinomialNB on large datasets, use partial_fit
                    elif isinstance(self.model, MultinomialNB) and X_combined is not None and _get_n_samples(X_combined) > 50000:
                        print(f"  [Training] Computing probabilities on {_get_n_samples(X_combined)} samples...")
                        print(f"  [Training] Using incremental learning (partial_fit)...")
                        
                        # Convert sparse to dense if needed
                        if hasattr(X_combined, 'toarray'):
                            X_combined = X_combined.toarray()
                        
                        batch_size = 100000
                        for i in range(0, _get_n_samples(X_combined), batch_size):
                            batch_end = min(i + batch_size, _get_n_samples(X_combined))
                            batch_X = X_combined[i:batch_end]
                            batch_y = y[i:batch_end]
                            self.model.partial_fit(batch_X, batch_y, classes=classes)
                            
                            elapsed = time.time() - start_time
                            print(f"  [Training] Processed {batch_end}/{_get_n_samples(X_combined)} samples... ({elapsed:.0f}s elapsed)", end='\r')
                        
                        elapsed = time.time() - start_time
                        print(f"  [Training] Completed in {elapsed:.2f}s" + " " * 30)
                    elif X_combined is not None:
                        print(f"  [Training] Computing probabilities on {_get_n_samples(X_combined)} samples...")
                        if hasattr(X_combined, 'toarray'):
                            X_combined = X_combined.toarray()
                        self.model.fit(X_combined, y)
                        elapsed = time.time() - start_time
                        print(f"  [Training] Completed in {elapsed:.2f}s")
                # For SGDClassifier (SVM approximation) on large datasets, ALWAYS use partial_fit
                elif isinstance(self.model, SGDClassifier):
                    # Get unique classes (needed for partial_fit)
                    classes = np.unique(y)
                    n_epochs = 100  # Multiple epochs needed for learning
                    
                    # Handle batched feature combination if needed
                    if self._needs_batched_training:
                        total_samples = len(X_text)
                        print(f"  [Training] Using incremental learning on {total_samples} samples with batched features, {n_epochs} epochs...")
                        start_time = time.time()
                        
                        batch_size = 50000
                        
                        for epoch in range(n_epochs):
                            indices = np.random.permutation(total_samples)
                            
                            for i in range(0, total_samples, batch_size):
                                batch_indices = indices[i:min(i + batch_size, total_samples)]
                                batch_indices_sorted = np.sort(batch_indices)
                                
                                batch_tfidf = self._X_tfidf_sparse[batch_indices_sorted]
                                batch_lexical = self._X_lexical.iloc[batch_indices_sorted]
                                
                                if hasattr(batch_tfidf, 'toarray'):
                                    batch_tfidf_array = batch_tfidf.toarray()
                                else:
                                    batch_tfidf_array = np.asarray(batch_tfidf)
                                batch_lexical_array = np.asarray(batch_lexical.values)
                                batch_X = np.hstack([batch_tfidf_array, batch_lexical_array])
                                batch_y = y[batch_indices_sorted]
                                
                                self.model.partial_fit(batch_X, batch_y, classes=classes)
                            
                            elapsed = time.time() - start_time
                            print(f"  [Training] Epoch {epoch+1}/{n_epochs} completed ({elapsed:.0f}s elapsed)")
                        
                        elapsed = time.time() - start_time
                        print(f"  [Training] Completed in {elapsed:.2f}s")
                    elif _get_n_samples(X_combined) > 50000:
                        n_samples = _get_n_samples(X_combined)
                        print(f"  [Training] Using incremental learning on {n_samples} samples, {n_epochs} epochs...")
                        start_time = time.time()
                        
                        # Convert sparse to dense if needed
                        if hasattr(X_combined, 'toarray'):
                            X_combined = X_combined.toarray()
                        
                        batch_size = 100000
                        for epoch in range(n_epochs):
                            indices = np.random.permutation(n_samples)
                            for i in range(0, n_samples, batch_size):
                                batch_indices = indices[i:min(i + batch_size, n_samples)]
                                batch_X = X_combined[batch_indices]
                                batch_y = y[batch_indices]
                                self.model.partial_fit(batch_X, batch_y, classes=classes)
                            
                            elapsed = time.time() - start_time
                            print(f"  [Training] Epoch {epoch+1}/{n_epochs} completed ({elapsed:.0f}s elapsed)")
                        
                        elapsed = time.time() - start_time
                        print(f"  [Training] Completed in {elapsed:.2f}s")
                else:
                    # Other models
                    print(f"  [Training] Fitting on {_get_n_samples(X_combined)} samples...")
                    start_time = time.time()
                    self.model.fit(X_combined, y)
                    elapsed = time.time() - start_time
                    print(f"  [Training] Completed in {elapsed:.2f}s")
        
        self.is_fitted = True
        print(f"[INFO] {self.name} trained")
    
    def predict(self, X_text: List[str], X_lexical: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # cuML expects pandas Series, not list
        if CUML_AVAILABLE and torch.cuda.is_available() and isinstance(self.vectorizer, cuTfidfVectorizer):
            import pandas as pd
            X_text_series = pd.Series(X_text)
            X_tfidf = self.vectorizer.transform(X_text_series)
        else:
            X_tfidf = self.vectorizer.transform(X_text)
        
        # Convert CuPy arrays/sparse matrices to NumPy if needed
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cp_sparse
            if isinstance(X_tfidf, cp.ndarray):
                X_tfidf = cp.asnumpy(X_tfidf)
            elif isinstance(X_tfidf, cp_sparse.spmatrix):
                X_tfidf = X_tfidf.get()
        except (ImportError, AttributeError):
            pass
        
        # For cuML models with large datasets, ALWAYS process in batches to avoid GPU memory issues
        # Convert to dense arrays in batches instead of all at once
        if CUML_AVAILABLE and isinstance(self.model, cuLogisticRegression) and len(X_text) > 10000:
            batch_size = 25000  # Smaller batches to avoid GPU memory issues
            predictions = []
            
            for i in range(0, len(X_text), batch_size):
                batch_text = X_text[i:i + batch_size]
                batch_lexical = X_lexical.iloc[i:i + batch_size] if X_lexical is not None else None
                
                # Transform batch
                if CUML_AVAILABLE and torch.cuda.is_available() and isinstance(self.vectorizer, cuTfidfVectorizer):
                    import pandas as pd
                    batch_text_series = pd.Series(batch_text)
                    batch_tfidf = self.vectorizer.transform(batch_text_series)
                else:
                    batch_tfidf = self.vectorizer.transform(batch_text)
                
                # Convert CuPy arrays/sparse matrices to NumPy if needed
                try:
                    import cupy as cp
                    import cupyx.scipy.sparse as cp_sparse
                    if isinstance(batch_tfidf, cp.ndarray):
                        batch_tfidf_array = cp.asnumpy(batch_tfidf)
                    elif isinstance(batch_tfidf, cp_sparse.spmatrix):
                        # Convert CuPy sparse to scipy sparse first, then to dense NumPy
                        batch_tfidf_scipy = batch_tfidf.get()
                        batch_tfidf_array = batch_tfidf_scipy.toarray()
                    else:
                        # Regular scipy sparse or numpy
                        if hasattr(batch_tfidf, 'toarray'):
                            batch_tfidf_array = batch_tfidf.toarray()
                        else:
                            batch_tfidf_array = np.asarray(batch_tfidf)
                except (ImportError, AttributeError):
                    # Fallback if cupy not available
                    if hasattr(batch_tfidf, 'toarray'):
                        batch_tfidf_array = batch_tfidf.toarray()
                    else:
                        batch_tfidf_array = np.asarray(batch_tfidf)
                
                if batch_lexical is not None:
                    X_batch = np.hstack([batch_tfidf_array, np.asarray(batch_lexical.values)])
                else:
                    X_batch = batch_tfidf_array
                
                # Predict on batch
                batch_preds = self.model.predict(X_batch)
                predictions.extend(batch_preds)
            
            return np.array(predictions)
        
        # For smaller datasets or non-cuML models, process normally
        # But still use batching for large datasets with lexical features to avoid memory issues
        if X_lexical is not None:
            # For large datasets, process in batches
            if len(X_text) > 100000 and hasattr(X_tfidf, 'toarray'):
                batch_size = 50000
                predictions = []
                for i in range(0, len(X_text), batch_size):
                    batch_end = min(i + batch_size, len(X_text))
                    batch_tfidf = X_tfidf[i:batch_end]
                    batch_lexical = X_lexical.iloc[i:batch_end]
                    
                    if hasattr(batch_tfidf, 'toarray'):
                        batch_tfidf_array = batch_tfidf.toarray()
                    else:
                        batch_tfidf_array = np.asarray(batch_tfidf)
                    batch_lexical_array = np.asarray(batch_lexical.values)
                    batch_X = np.hstack([batch_tfidf_array, batch_lexical_array])
                    
                    batch_predictions = self.model.predict(batch_X)
                    predictions.extend(batch_predictions)
                
                return np.array(predictions)
            else:
                # Small dataset: convert to dense arrays
                if hasattr(X_tfidf, 'toarray'):
                    X_tfidf_array = np.asarray(X_tfidf.toarray())
                else:
                    X_tfidf_array = np.asarray(X_tfidf)
                X_combined = np.hstack([X_tfidf_array, np.asarray(X_lexical.values)])
        else:
            if hasattr(X_tfidf, 'toarray'):
                X_combined = np.asarray(X_tfidf.toarray())
            else:
                X_combined = np.asarray(X_tfidf)
        
        return self.model.predict(X_combined)
    
    def predict_proba(self, X_text: List[str], X_lexical: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # cuML expects pandas Series, not list
        if CUML_AVAILABLE and torch.cuda.is_available() and isinstance(self.vectorizer, cuTfidfVectorizer):
            import pandas as pd
            X_text_series = pd.Series(X_text)
            X_tfidf = self.vectorizer.transform(X_text_series)
        else:
            X_tfidf = self.vectorizer.transform(X_text)
        
        # Convert CuPy arrays/sparse matrices to NumPy if needed
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cp_sparse
            if isinstance(X_tfidf, cp.ndarray):
                X_tfidf = cp.asnumpy(X_tfidf)
            elif isinstance(X_tfidf, cp_sparse.spmatrix):
                X_tfidf = X_tfidf.get()
        except (ImportError, AttributeError):
            pass
        
        # For cuML models with large datasets, ALWAYS process in batches to avoid GPU memory issues
        if CUML_AVAILABLE and isinstance(self.model, cuLogisticRegression) and len(X_text) > 10000:
            batch_size = 25000  # Smaller batches to avoid GPU memory issues
            probas = []
            
            for i in range(0, len(X_text), batch_size):
                batch_text = X_text[i:i + batch_size]
                batch_lexical = X_lexical.iloc[i:i + batch_size] if X_lexical is not None else None
                
                # Transform batch
                if CUML_AVAILABLE and torch.cuda.is_available() and isinstance(self.vectorizer, cuTfidfVectorizer):
                    import pandas as pd
                    batch_text_series = pd.Series(batch_text)
                    batch_tfidf = self.vectorizer.transform(batch_text_series)
                else:
                    batch_tfidf = self.vectorizer.transform(batch_text)
                
                # Convert CuPy arrays/sparse matrices to NumPy if needed
                try:
                    import cupy as cp
                    import cupyx.scipy.sparse as cp_sparse
                    if isinstance(batch_tfidf, cp.ndarray):
                        batch_tfidf_array = cp.asnumpy(batch_tfidf)
                    elif isinstance(batch_tfidf, cp_sparse.spmatrix):
                        # Convert CuPy sparse to scipy sparse first, then to dense NumPy
                        batch_tfidf_scipy = batch_tfidf.get()
                        batch_tfidf_array = batch_tfidf_scipy.toarray()
                    else:
                        # Regular scipy sparse or numpy
                        if hasattr(batch_tfidf, 'toarray'):
                            batch_tfidf_array = batch_tfidf.toarray()
                        else:
                            batch_tfidf_array = np.asarray(batch_tfidf)
                except (ImportError, AttributeError):
                    # Fallback if cupy not available
                    if hasattr(batch_tfidf, 'toarray'):
                        batch_tfidf_array = batch_tfidf.toarray()
                    else:
                        batch_tfidf_array = np.asarray(batch_tfidf)
                
                if batch_lexical is not None:
                    X_batch = np.hstack([batch_tfidf_array, np.asarray(batch_lexical.values)])
                else:
                    X_batch = batch_tfidf_array
                
                # Predict on batch
                batch_proba = self.model.predict_proba(X_batch)
                probas.append(batch_proba)
            
            return np.vstack(probas)
        
        # For smaller datasets or non-cuML models, process normally
        if X_lexical is not None:
            if hasattr(X_tfidf, 'toarray'):
                X_tfidf_array = np.asarray(X_tfidf.toarray())
            else:
                X_tfidf_array = np.asarray(X_tfidf)
            X_combined = np.hstack([X_tfidf_array, np.asarray(X_lexical.values)])
        else:
            if hasattr(X_tfidf, 'toarray'):
                X_combined = np.asarray(X_tfidf.toarray())
            else:
                X_combined = np.asarray(X_tfidf)
        
        if self.is_pytorch:
            return self.model.predict_proba(X_combined)
        elif hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_combined)
        elif isinstance(self.model, (LinearSVC, SGDClassifier)):
            # Check if it's SGDClassifier with log loss (Logistic Regression approximation)
            if isinstance(self.model, SGDClassifier) and self.model.loss == 'log_loss':
                # SGDClassifier with log loss has predict_proba
                return self.model.predict_proba(X_combined)
            # Otherwise, use decision_function like LinearSVC
            # LinearSVC and SGDClassifier with hinge loss don't have predict_proba
            # Use decision_function and convert to probabilities
            if hasattr(self.model, 'predict_proba'):
                # SGDClassifier with some losses has predict_proba
                return self.model.predict_proba(X_combined)
            else:
                # Use decision_function and convert to probabilities using sigmoid
                decision = self.model.decision_function(X_combined)
                # Convert decision function to probabilities using sigmoid
                # decision > 0 means class 1, decision < 0 means class 0
                # Use sigmoid to convert to [0, 1] range
                try:
                    from scipy.special import expit
                    prob_class1 = expit(decision)  # Probability of class 1
                except ImportError:
                    # Fallback: manual sigmoid implementation
                    prob_class1 = 1.0 / (1.0 + np.exp(-decision))
                prob_class0 = 1 - prob_class1  # Probability of class 0
                proba = np.column_stack([prob_class0, prob_class1])
                return proba
        else:
            # For models without predict_proba, return dummy probabilities
            predictions = self.model.predict(X_combined)
            proba = np.zeros((len(predictions), 2))
            proba[np.arange(len(predictions)), predictions] = 1.0
            return proba


def evaluate_classifier(
    classifier: MLBaselineClassifier,
    X_test_text: List[str],
    y_test: np.ndarray,
    X_test_lexical: Optional[pd.DataFrame] = None,
    class_names: List[str] = ['pre-chatgpt', 'post-chatgpt'],
    log_to_wandb: bool = False,
    wandb_run = None
) -> Dict:
    """Evaluate a classifier and return metrics."""
    print(f"\nEvaluating {classifier.name}...")
    
    # Predictions
    y_pred = classifier.predict(X_test_text, X_test_lexical)
    y_proba = classifier.predict_proba(X_test_text, X_test_lexical)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'classifier': classifier.name,
        'accuracy': float(accuracy),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    # Print results
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision (weighted): {precision:.4f}")
    print(f"  Recall (weighted): {recall:.4f}")
    print(f"  F1 (weighted): {f1:.4f}")
    print(f"\n  Per-class metrics:")
    for i, class_name in enumerate(class_names):
        print(f"    {class_name}:")
        print(f"      Precision: {precision_per_class[i]:.4f}")
        print(f"      Recall: {recall_per_class[i]:.4f}")
        print(f"      F1: {f1_per_class[i]:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    {cm}")
    
    # Log to wandb if available
    if log_to_wandb and WANDB_AVAILABLE and wandb_run is not None:
        # Create a clean classifier name for wandb
        classifier_name_clean = classifier.name.replace(' ', '_').replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(':', '').lower()
        
        # Log metrics
        wandb_metrics = {
            f'{classifier_name_clean}/accuracy': accuracy,
            f'{classifier_name_clean}/precision_weighted': precision,
            f'{classifier_name_clean}/recall_weighted': recall,
            f'{classifier_name_clean}/f1_weighted': f1,
        }
        
        # Per-class metrics
        for i, class_name in enumerate(class_names):
            class_name_clean = class_name.replace('-', '_')
            wandb_metrics[f'{classifier_name_clean}/precision_{class_name_clean}'] = precision_per_class[i]
            wandb_metrics[f'{classifier_name_clean}/recall_{class_name_clean}'] = recall_per_class[i]
            wandb_metrics[f'{classifier_name_clean}/f1_{class_name_clean}'] = f1_per_class[i]
        
        # Log GPU stats if CUDA is available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_stats = get_gpu_stats(device_id=0)
            for key, value in gpu_stats.items():
                wandb_metrics[f'{classifier_name_clean}/gpu_{key}'] = value
        
        # Log CPU and RAM stats
        cpu_ram_stats = get_cpu_ram_stats()
        for key, value in cpu_ram_stats.items():
            wandb_metrics[f'{classifier_name_clean}/{key}'] = value
        
        wandb_run.log(wandb_metrics)
        
        # Log confusion matrix as image
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix - {classifier.name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            wandb_run.log({f'{classifier_name_clean}/confusion_matrix': wandb.Image(plt)})
            plt.close()
        except ImportError:
            pass  # matplotlib/seaborn not available
    
    return results, y_pred, y_proba


def analyze_misclassifications(
    df_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classifier_name: str,
    n_examples: int = 10
) -> List[Dict]:
    """Analyze misclassified examples for qualitative evaluation."""
    print(f"\nAnalyzing misclassifications for {classifier_name}...")
    
    misclassified = []
    for idx in range(len(y_test)):
        if y_test.iloc[idx] != y_pred[idx]:
            true_label = 'pre-chatgpt' if y_test.iloc[idx] == 0 else 'post-chatgpt'
            pred_label = 'pre-chatgpt' if y_pred[idx] == 0 else 'post-chatgpt'
            confidence = max(y_proba[idx])
            
            misclassified.append({
                'text_id': df_test.iloc[idx]['text_id'],
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': float(confidence),
                'text_preview': df_test.iloc[idx]['raw_text'][:200] + '...' if len(df_test.iloc[idx]['raw_text']) > 200 else df_test.iloc[idx]['raw_text'],
                'language': df_test.iloc[idx]['language'],
                'publication_date': df_test.iloc[idx]['publication_date']
            })
    
    # Sort by confidence (lowest first - most uncertain)
    misclassified.sort(key=lambda x: x['confidence'])
    
    print(f"  Found {len(misclassified)} misclassified examples")
    print(f"  Showing {min(n_examples, len(misclassified))} examples:")
    
    for i, example in enumerate(misclassified[:n_examples], 1):
        print(f"\n  Example {i}:")
        print(f"    Text ID: {example['text_id']}")
        print(f"    True: {example['true_label']}, Predicted: {example['predicted_label']}")
        print(f"    Confidence: {example['confidence']:.4f}")
        print(f"    Language: {example['language']}, Date: {example['publication_date']}")
        print(f"    Preview: {example['text_preview']}")
    
    return misclassified


class TeeOutput:
    """Capture stdout/stderr and write to both console and file."""
    def __init__(self, file_path: Path, original_stdout, original_stderr):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.original_stdout = original_stdout
        self.original_stderr = original_stderr
    
    def write(self, text):
        self.original_stdout.write(text)
        self.file.write(text)
        self.file.flush()
    
    def flush(self):
        self.original_stdout.flush()
        self.file.flush()
    
    def isatty(self):
        """Check if the original stream is a TTY."""
        return self.original_stdout.isatty()
    
    def close(self):
        self.file.close()


def main():
    """Main function to run ML baseline experiments."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='ML Baseline Methods for Pre-ChatGPT vs Post-ChatGPT Classification')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for PyTorch MLP models (default: 100)')
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
    
    EPOCHS = args.epochs
    TEST_PERCENT = args.test_percent
    # Setup terminal output capture
    output_dir = Path('task2/results')
    output_dir.mkdir(exist_ok=True)
    terminal_output_file = output_dir / 'terminal_output.txt'
    
    # Create TeeOutput to capture all prints
    tee = TeeOutput(terminal_output_file, sys.stdout, sys.stderr)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    
    try:
        print("=" * 80)
        print("ML Baseline Methods for Pre-ChatGPT vs Post-ChatGPT Classification")
        print("=" * 80)
        if TEST_MODE:
            print(f"[CONFIG] Test mode: ON ({TEST_PERCENT}% of data)")
        else:
            print("[CONFIG] Test mode: OFF (full dataset)")
        print(f"[CONFIG] PyTorch MLP epochs: {EPOCHS}")
        print("=" * 80)
        
        # Initialize wandb if available
        wandb_run = None
        if WANDB_AVAILABLE:
            wandb.init(
                project="nlp-lexical-diversity-milestone2",
                name="ml-baselines",
                config={
                    "task": "pre-chatgpt vs post-chatgpt classification",
                    "test_size": 0.2,
                    "random_state": 42,
                    "tfidf_max_features": 5000,
                    "tfidf_ngram_range": "(1, 2)",
                    "use_lexical_features": True,
                    "cuda_available": TORCH_AVAILABLE and torch.cuda.is_available() if TORCH_AVAILABLE else False,
                    "test_mode": TEST_MODE,
                    "test_percent": TEST_PERCENT if TEST_MODE else 100.0,
                    "epochs": EPOCHS,
                }
            )
            wandb_run = wandb.run
            print("[INFO] wandb initialized")
        else:
            print("[INFO] wandb not available. Install with: pip install wandb (optional)")
        
        # Initialize CUDA device if PyTorch is available
        global DEVICE
        if TORCH_AVAILABLE and DEVICE is None:
            DEVICE = get_device()
            # Log device info to wandb
            if wandb_run:
                wandb_run.config.update({
                    "device": str(DEVICE),
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
                })
        elif not TORCH_AVAILABLE:
            print("[INFO] PyTorch not available. Install with: pip install torch")
        
        # Load data
        data_dir = Path('processed_data')
        if not (data_dir / 'processed_data.csv').exists():
            print(f"Error: Processed data not found in {data_dir}")
            print("Please run preprocess_leipzig.py first to generate processed_data.csv")
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
        print(f"   Label distribution:")
        print(f"     Pre-ChatGPT (0): {(labels == 0).sum()}")
        print(f"     Post-ChatGPT (1): {(labels == 1).sum()}")
        
        # Split data
        print("\n2. Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = split_train_test(
            features_df, labels, test_size=0.2, random_state=42
        )
        print(f"   Train set: {len(X_train)} articles")
        print(f"   Test set: {len(X_test)} articles")
        
        # Prepare features
        print("\n3. Preparing features...")
        X_train_text = X_train['raw_text'].tolist()
        X_test_text = X_test['raw_text'].tolist()
        
        # Compute lexical diversity features
        print("\n4. Computing lexical diversity features...")
        X_train_lexical = compute_lexical_diversity_features_batch(X_train, use_lexical_features=True)
        X_test_lexical = compute_lexical_diversity_features_batch(X_test, use_lexical_features=True)
        
        # IMPORTANT: Normalize lexical features to match TF-IDF scale (0-1)
        # Without this, features like token_count (50-500) overwhelm TF-IDF values (0-1)
        # Use MinMaxScaler (not StandardScaler) because Naive Bayes requires non-negative values
        if X_train_lexical is not None and X_test_lexical is not None:
            print("   [INFO] Normalizing lexical features with MinMaxScaler (0-1 range)...")
            lexical_scaler = MinMaxScaler()
            X_train_lexical_scaled = pd.DataFrame(
                lexical_scaler.fit_transform(X_train_lexical),
                columns=X_train_lexical.columns,
                index=X_train_lexical.index
            )
            X_test_lexical_scaled = pd.DataFrame(
                lexical_scaler.transform(X_test_lexical),
                columns=X_test_lexical.columns,
                index=X_test_lexical.index
            )
            X_train_lexical = X_train_lexical_scaled
            X_test_lexical = X_test_lexical_scaled
        
        # Initialize classifiers
        print("\n5. Initializing ML baseline classifiers...")
        classifiers = []
        
        # Use GPU-accelerated models if cuML is available, otherwise use CPU-based scikit-learn
        # Note: For large datasets, ALWAYS use batching/incremental learning to avoid memory issues
        LARGE_DATASET_THRESHOLD = 50000  # Use batching for datasets > 50K samples
        
        if CUML_AVAILABLE and torch.cuda.is_available():
            print("   [INFO] Using GPU-accelerated models (cuML)")
            # For large datasets, ALWAYS use SGDClassifier with log loss (supports incremental learning)
            # This approximates Logistic Regression but can train in batches
            if len(X_train) > LARGE_DATASET_THRESHOLD:
                print("   [INFO] Using SGDClassifier (log loss) for large datasets (supports incremental learning)")
                classifiers.append(
                    MLBaselineClassifier(
                        "Logistic Regression (TF-IDF) [CPU - SGD]",
                        SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, learning_rate='optimal')
                    )
                )
            else:
                classifiers.append(
                    MLBaselineClassifier(
                        "Logistic Regression (TF-IDF) [GPU]",
                        cuLogisticRegression(max_iter=1000)
                    )
                )
            # Use scikit-learn Naive Bayes for large datasets (cuML has CUDA memory issues)
            if len(X_train) > LARGE_DATASET_THRESHOLD:
                print("   [INFO] Using CPU-based Naive Bayes (cuML has memory issues with large datasets)")
                classifiers.append(
                    MLBaselineClassifier(
                        "Naive Bayes (TF-IDF) [CPU]",
                        MultinomialNB()
                    )
                )
            else:
                classifiers.append(
                    MLBaselineClassifier(
                        "Naive Bayes (TF-IDF) [GPU]",
                        cuMultinomialNB()
                    )
                )
        else:
            print("   [INFO] Using CPU-based models (scikit-learn)")
            # For large datasets, ALWAYS use SGDClassifier with batching
            if len(X_train) > LARGE_DATASET_THRESHOLD:
                classifiers.append(
                    MLBaselineClassifier(
                        "Logistic Regression (TF-IDF) [CPU - SGD]",
                        SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, learning_rate='optimal')
                    )
                )
            else:
                classifiers.append(
                    MLBaselineClassifier(
                        "Logistic Regression (TF-IDF)",
                        LogisticRegression(max_iter=1000, random_state=42, verbose=1)
                    )
                )
            classifiers.append(
                MLBaselineClassifier(
                    "Naive Bayes (TF-IDF)",
                    MultinomialNB()
                )
            )
        
        # Skip SVM in TEST_MODE as it's very slow
        # For large datasets, ALWAYS use SGDClassifier with hinge loss (supports incremental learning)
        # For smaller datasets, use LinearSVC (faster)
        if not TEST_MODE:
            if len(X_train) > LARGE_DATASET_THRESHOLD:
                # Use SGDClassifier with hinge loss for large datasets (supports partial_fit)
                classifiers.append(
                    MLBaselineClassifier(
                        "SVM (TF-IDF) [CPU - SGD]",
                        SGDClassifier(loss='hinge', random_state=42, max_iter=1000, learning_rate='optimal')
                    )
                )
            else:
                # Use LinearSVC for smaller datasets (faster)
                classifiers.append(
                    MLBaselineClassifier(
                        "SVM (TF-IDF) [CPU]",
                        LinearSVC(random_state=42, max_iter=1000, dual=False)
                    )
                )
        
        # Add PyTorch MLP with CUDA support if available
        if TORCH_AVAILABLE:
            classifiers.append(
                MLBaselineClassifier(
                    f"PyTorch MLP (TF-IDF) [Device: {DEVICE}]",
                    PyTorchMLPClassifier(hidden_sizes=(100, 50), max_epochs=EPOCHS, device=DEVICE)
                )
            )
        else:
            # Fallback to sklearn MLP if PyTorch not available
            classifiers.append(
                MLBaselineClassifier(
                    "MLP Neural Network (TF-IDF) [CPU]",
                    MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
                )
            )
        
        # Add classifiers with lexical features if available
        if X_train_lexical is not None:
            if CUML_AVAILABLE and torch.cuda.is_available():
                # For large datasets, ALWAYS use SGDClassifier with log loss (supports incremental learning)
                if len(X_train) > LARGE_DATASET_THRESHOLD:
                    classifiers.append(
                        MLBaselineClassifier(
                            "Logistic Regression (TF-IDF + Lexical) [CPU - SGD]",
                            SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, learning_rate='optimal')
                        )
                    )
                else:
                    classifiers.append(
                        MLBaselineClassifier(
                            "Logistic Regression (TF-IDF + Lexical) [GPU]",
                            cuLogisticRegression(max_iter=1000)
                        )
                    )
                # Use scikit-learn Naive Bayes for large datasets (cuML has CUDA memory issues)
                if len(X_train) > LARGE_DATASET_THRESHOLD:
                    classifiers.append(
                        MLBaselineClassifier(
                            "Naive Bayes (TF-IDF + Lexical) [CPU]",
                            MultinomialNB()
                        )
                    )
                else:
                    classifiers.append(
                        MLBaselineClassifier(
                            "Naive Bayes (TF-IDF + Lexical) [GPU]",
                            cuMultinomialNB()
                        )
                    )
            else:
                # For large datasets, ALWAYS use SGDClassifier with batching
                if len(X_train) > LARGE_DATASET_THRESHOLD:
                    classifiers.append(
                        MLBaselineClassifier(
                            "Logistic Regression (TF-IDF + Lexical) [CPU - SGD]",
                            SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, learning_rate='optimal')
                        )
                    )
                else:
                    classifiers.append(
                        MLBaselineClassifier(
                            "Logistic Regression (TF-IDF + Lexical)",
                            LogisticRegression(max_iter=1000, random_state=42, verbose=1)
                        )
                    )
                classifiers.append(
                    MLBaselineClassifier(
                        "Naive Bayes (TF-IDF + Lexical)",
                        MultinomialNB()
                    )
                )
            
            # PyTorch MLP + Lexical DISABLED - requires too much RAM for dense conversion on large datasets
            # if TORCH_AVAILABLE:
            #     classifiers.append(
            #         MLBaselineClassifier(
            #             f"PyTorch MLP (TF-IDF + Lexical) [Device: {DEVICE}]",
            #             PyTorchMLPClassifier(hidden_sizes=(100, 50), max_epochs=EPOCHS, device=DEVICE)
            #         )
            #     )
        
        # Train and evaluate each classifier
        print("\n6. Training and evaluating classifiers...")
        all_results = []
        all_misclassifications = {}
        
        for classifier in classifiers:
            # Determine if this classifier uses lexical features
            use_lexical = "Lexical" in classifier.name and X_train_lexical is not None
            
            # Train
            classifier.fit(
                X_train_text,
                y_train.values,
                X_train_lexical if use_lexical else None
            )
            
            # Evaluate
            results, y_pred, y_proba = evaluate_classifier(
                classifier,
                X_test_text,
                y_test.values,
                X_test_lexical if use_lexical else None,
                log_to_wandb=WANDB_AVAILABLE,
                wandb_run=wandb_run
            )
            all_results.append(results)
            
            # Save predictions for comparison
            predictions_data = {
                'text_id': X_test['text_id'].values.tolist(),
                'true_label': y_test.values.tolist(),
                'predicted_label': y_pred.tolist(),
                'predicted_proba': y_proba.tolist(),
                'language': X_test['language'].values.tolist(),
                'publication_date': X_test['publication_date'].values.tolist()
            }
            predictions_file = output_dir / f"predictions_{classifier.name.replace(' ', '_').replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(':', '').lower()}.json"
            with open(predictions_file, 'w', encoding='utf-8') as f:
                json.dump(predictions_data, f, indent=2, ensure_ascii=False)
            print(f"   [INFO] Saved predictions for {classifier.name} to {predictions_file}")
            
            # Analyze misclassifications
            misclass = analyze_misclassifications(
                X_test,
                y_test,
                y_pred,
                y_proba,
                classifier.name,
                n_examples=5
            )
            all_misclassifications[classifier.name] = misclass
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY OF ALL CLASSIFIERS")
        print("=" * 80)
        print(f"\n{'Classifier':<40} {'Accuracy':<12} {'F1 (weighted)':<15}")
        print("-" * 80)
        for result in all_results:
            print(f"{result['classifier']:<40} {result['accuracy']:<12.4f} {result['f1_weighted']:<15.4f}")
        
        # Save results
        print("\n7. Saving results...")
        
        # Save quantitative results
        results_file = output_dir / 'ml_baseline_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"   [INFO] Saved quantitative results to {results_file}")
        
        # Save misclassifications
        misclass_file = output_dir / 'ml_baseline_misclassifications.json'
        with open(misclass_file, 'w', encoding='utf-8') as f:
            json.dump(all_misclassifications, f, indent=2, ensure_ascii=False)
        print(f"   [INFO] Saved misclassification analysis to {misclass_file}")
        
        # Log terminal output to wandb
        if wandb_run:
            # Flush and close tee, restore stdout/stderr
            tee.flush()
            tee.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
            if terminal_output_file.exists():
                with open(terminal_output_file, 'r', encoding='utf-8') as f:
                    terminal_output = f.read()
                    # Log as HTML in wandb
                    wandb_run.log({"terminal_output": wandb.Html(f"<pre style='white-space: pre-wrap; font-family: monospace;'>{terminal_output}</pre>")})
                    # Also save as artifact for download
                    artifact = wandb.Artifact("terminal_output", type="log")
                    artifact.add_file(str(terminal_output_file))
                    wandb_run.log_artifact(artifact)
            
            wandb.finish()
            print("\n[INFO] wandb run completed")
        else:
            # Restore stdout/stderr if wandb not available
            tee.flush()
            tee.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        
        print("\n" + "=" * 80)
        print("ML Baseline Evaluation Complete!")
        print("=" * 80)
    except Exception as e:
        # Make sure to restore stdout/stderr even on error
        tee.flush()
        tee.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        raise


if __name__ == "__main__":
    main()

