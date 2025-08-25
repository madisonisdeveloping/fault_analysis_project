"""
Common utilities for Power System Fault Analysis
Shared functions and error handling across modules.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from functools import wraps
from contextlib import contextmanager

import config

class PowerSystemError(Exception):
    """Base exception for power system analysis errors."""
    pass

class DataValidationError(PowerSystemError):
    """Raised when data validation fails."""
    pass

class ModelError(PowerSystemError):
    """Raised when model operations fail."""
    pass

def validate_file_exists(filepath):
    """
    Validate that a file exists and is readable.
    
    Args:
        filepath: Path to file (str or Path object)
        
    Returns:
        Path: Validated Path object
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {filepath}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {filepath}")
    return path

def ensure_output_dir():
    """Ensure output directory exists."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_csv_safely(filepath, required_columns=None):
    """
    Load CSV file with validation.
    
    Args:
        filepath: Path to CSV file
        required_columns: List of columns that must be present
        
    Returns:
        pd.DataFrame: Loaded and validated dataframe
        
    Raises:
        DataValidationError: If validation fails
    """
    try:
        path = validate_file_exists(filepath)
        df = pd.read_csv(path)
        
        if df.empty:
            raise DataValidationError(f"CSV file is empty: {filepath}")
            
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise DataValidationError(
                    f"Missing required columns in {filepath}: {missing_cols}"
                )
        
        return df
        
    except pd.errors.EmptyDataError:
        raise DataValidationError(f"CSV file is empty or corrupted: {filepath}")
    except pd.errors.ParserError as e:
        raise DataValidationError(f"Failed to parse CSV {filepath}: {e}")

def load_numpy_array(filepath, allow_pickle=False):
    """
    Load NumPy array with error handling.
    
    Args:
        filepath: Path to .npy file
        allow_pickle: Whether to allow pickle loading
        
    Returns:
        np.ndarray: Loaded array
        
    Raises:
        ModelError: If loading fails
    """
    try:
        path = validate_file_exists(filepath)
        return np.load(path, allow_pickle=allow_pickle)
    except Exception as e:
        raise ModelError(f"Failed to load numpy array from {filepath}: {e}")

def save_numpy_array(array, filepath):
    """
    Save NumPy array with directory creation.
    
    Args:
        array: Array to save
        filepath: Output path
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"{func.__name__} completed in {duration:.2f} seconds")
        return result
    return wrapper

@contextmanager
def error_handler(operation_name):
    """Context manager for consistent error handling."""
    try:
        print(f"Starting {operation_name}...")
        yield
        print(f"{operation_name} completed successfully")
    except PowerSystemError as e:
        print(f"Error in {operation_name}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error in {operation_name}: {e}")
        raise PowerSystemError(f"{operation_name} failed: {e}")

def format_percentage(value, total):
    """Format a value as percentage of total."""
    if total == 0:
        return "0.0%"
    return f"{(value / total) * 100:.1f}%"

def format_large_number(number):
    """Format large numbers with thousand separators."""
    return f"{number:,}"

def get_memory_usage():
    """Get current memory usage if psutil is available."""
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return f"{memory_mb:.1f} MB"
    except ImportError:
        return "N/A"

class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total, description="Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, increment=1):
        """Update progress counter."""
        self.current = min(self.current + increment, self.total)
        self._print_progress()
    
    def _print_progress(self):
        """Print current progress."""
        if self.total == 0:
            return
            
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        if self.current == self.total:
            print(f"{self.description}: 100.0% ({self.current}/{self.total}) - "
                  f"Completed in {elapsed:.1f}s")
        elif self.current % max(1, self.total // 20) == 0:  # Update every 5%
            eta = (elapsed / self.current) * (self.total - self.current) if self.current > 0 else 0
            print(f"{self.description}: {percentage:.1f}% ({self.current}/{self.total}) - "
                  f"ETA: {eta:.1f}s")

def validate_model_consistency(*arrays):
    """
    Validate that multiple arrays have consistent first dimensions.
    
    Args:
        *arrays: Variable number of arrays to check
        
    Raises:
        DataValidationError: If dimensions don't match
    """
    if len(arrays) < 2:
        return
        
    first_length = len(arrays[0])
    for i, array in enumerate(arrays[1:], 1):
        if len(array) != first_length:
            raise DataValidationError(
                f"Array dimension mismatch: array 0 has {first_length} items, "
                f"array {i} has {len(array)} items"
            )

def filter_power_lines(df, line_prefix=None):
    """
    Filter dataframe to include only power line data.
    
    Args:
        df: DataFrame with 'Name' column
        line_prefix: Prefix for power lines (default from config)
        
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    if line_prefix is None:
        line_prefix = config.LINE_PREFIX
        
    if 'Name' not in df.columns:
        raise DataValidationError("DataFrame must have 'Name' column for filtering")
    
    line_mask = df['Name'].str.startswith(line_prefix, na=False)
    return df[line_mask].copy()

def get_critical_line_names(line_numbers=None):
    """
    Get full line names for critical line numbers.
    
    Args:
        line_numbers: List of line numbers (default from config)
        
    Returns:
        list: List of full line names
    """
    if line_numbers is None:
        line_numbers = config.AnalysisConfig.GROUND_TRUTH_CRITICAL
        
    try:
        # Try to load actual line names if available
        with open(config.FilePaths.LINE_NAMES, 'r') as f:
            all_lines = [line.strip() for line in f.readlines()]
        
        critical_lines = []
        for line in all_lines:
            parts = line.split('_')
            if len(parts) >= 2 and any(num in parts[1:] for num in line_numbers):
                critical_lines.append(line)
        
        return critical_lines if critical_lines else [f'LI_{num}' for num in line_numbers]
        
    except FileNotFoundError:
        # Fall back to simple naming convention
        return [f'LI_{num}' for num in line_numbers]