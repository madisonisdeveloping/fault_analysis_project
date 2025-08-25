"""
Data Processing Pipeline for Power System Fault Analysis
Handles CSV data transformation and labeling for supervised learning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import config

class DataProcessor:
    """Handles power flow data processing and feature engineering."""
    
    def __init__(self, threshold=None):
        self.threshold = threshold or config.SupervisedConfig.FAULT_THRESHOLD
        
    def compute_power_deltas(self, df):
        """Calculate power flow changes across timestamps."""
        if 'Name' not in df.columns:
            raise ValueError("Input dataframe must contain 'Name' column")
            
        # Separate metadata from time series data
        name_column = df[['Name']].copy()
        time_series = df.drop(columns=['Name']).copy()
        
        # Ensure numeric data types
        for col in time_series.columns:
            time_series[col] = pd.to_numeric(time_series[col], errors='coerce')
        
        # Fill any NaN values with zero
        time_series = time_series.fillna(0.0)
        
        # Calculate differences between consecutive timestamps
        delta_data = time_series.diff(axis=1).fillna(0)
        delta_data.columns = [f'delta_{col}' for col in time_series.columns]
        
        # Combine with original names
        result = pd.concat([name_column, delta_data], axis=1)
        return result.copy()
    
    def create_labeled_dataset(self, delta_df):
        """Transform delta data into labeled training format."""
        # Get delta columns only
        delta_columns = [col for col in delta_df.columns if col.startswith('delta_')]
        
        # Melt dataframe for row-per-timestamp format
        melted = pd.melt(
            delta_df, 
            id_vars=['Name'], 
            value_vars=delta_columns,
            var_name='timestamp', 
            value_name='delta_power'
        )
        
        # Create binary fault labels based on threshold
        melted['label'] = (melted['delta_power'].abs() > self.threshold).astype(int)
        
        # Clean up column names
        result = melted[['Name', 'delta_power', 'label']].copy()
        result.rename(columns={'Name': 'line_name'}, inplace=True)
        
        return result

class DatasetBuilder:
    """Builds complete datasets from multiple CSV files."""
    
    def __init__(self, data_folder=None, threshold=None):
        self.data_folder = Path(data_folder or config.DATA_DIR)
        self.processor = DataProcessor(threshold)
        
    def get_valid_csv_files(self):
        """Get list of valid CSV files in data folder."""
        if not self.data_folder.exists():
            raise FileNotFoundError(f"Data folder not found: {self.data_folder}")
            
        csv_files = list(self.data_folder.glob(config.CSV_PATTERN))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_folder}")
            
        return sorted(csv_files)
    
    def process_single_file(self, filepath):
        """Process a single CSV file into labeled format."""
        try:
            df = pd.read_csv(filepath)
            
            if 'Name' not in df.columns:
                print(f"Skipping {filepath.name}: missing 'Name' column")
                return None
                
            delta_df = self.processor.compute_power_deltas(df)
            labeled_df = self.processor.create_labeled_dataset(delta_df)
            
            return labeled_df
            
        except Exception as e:
            print(f"Error processing {filepath.name}: {e}")
            return None
    
    def build_combined_dataset(self, output_path=None):
        """Combine all CSV files into single labeled dataset."""
        if output_path is None:
            output_path = config.FilePaths.LABELED_DATASET
        else:
            output_path = Path(output_path)
            
        print("Building labeled dataset from CSV files...")
        
        csv_files = self.get_valid_csv_files()
        print(f"Found {len(csv_files)} CSV files to process")
        
        processed_datasets = []
        
        for filepath in csv_files:
            print(f"Processing {filepath.name}...")
            labeled_data = self.process_single_file(filepath)
            
            if labeled_data is not None:
                processed_datasets.append(labeled_data)
                fault_count = labeled_data['label'].sum()
                total_count = len(labeled_data)
                fault_pct = (fault_count / total_count) * 100
                print(f"  → {total_count:,} records, {fault_count:,} faults ({fault_pct:.1f}%)")
        
        if not processed_datasets:
            raise ValueError("No valid datasets were processed")
        
        # Combine all datasets
        combined_dataset = pd.concat(processed_datasets, ignore_index=True)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save combined dataset
        combined_dataset.to_csv(output_path, index=False)
        
        # Report statistics
        total_records = len(combined_dataset)
        total_faults = combined_dataset['label'].sum()
        fault_percentage = (total_faults / total_records) * 100
        unique_lines = combined_dataset['line_name'].nunique()
        
        print(f"\nDataset creation completed:")
        print(f"  • Total records: {total_records:,}")
        print(f"  • Fault instances: {total_faults:,} ({fault_percentage:.1f}%)")
        print(f"  • Unique power lines: {unique_lines}")
        print(f"  • Saved to: {output_path}")
        
        return combined_dataset

def merge_labeled_from_folder(data_folder=None, output_path=None, threshold=None):
    """
    Main function to create labeled dataset from folder of CSV files.
    
    Args:
        data_folder: Path to folder containing CSV files
        output_path: Output path for combined dataset
        threshold: Power delta threshold for fault detection
    """
    builder = DatasetBuilder(data_folder, threshold)
    return builder.build_combined_dataset(output_path)

if __name__ == "__main__":
    # Example usage
    merge_labeled_from_folder()