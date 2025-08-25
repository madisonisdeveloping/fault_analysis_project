import numpy as np
import pandas as pd
import os
from pathlib import Path

# Configuration constants
OUTPUT_DIR = Path('output')
REQUIRED_FILES = {
    'supervised': {
        'dataset': 'labeled_dataset_full.csv',
        'true_labels': 'supervised_true.npy',
        'pred_labels': 'supervised_pred.npy'
    },
    'unsupervised': {
        'line_names': 'unsupervised_line_names.csv',
        'cluster_labels': 'cluster_labels.npy'
    }
}

class AnalysisError(Exception):
    """Custom exception for analysis-related errors."""
    pass

def validate_file_exists(filepath):
    """Check if required file exists and return Path object."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {filepath}")
    return path

def load_supervised_data():
    """Load and validate supervised model results."""
    try:
        # use the test results file that contains aligned data
        test_results_path = OUTPUT_DIR / 'supervised_test_results.csv'
        
        if not test_results_path.exists():
            # fall back to original method if test results don't exist
            files = REQUIRED_FILES['supervised']
            data_path = OUTPUT_DIR / files['dataset']
            true_path = OUTPUT_DIR / files['true_labels']
            pred_path = OUTPUT_DIR / files['pred_labels']
            
            # Validate all files exist
            for path in [data_path, true_path, pred_path]:
                validate_file_exists(path)
            
            data_df = pd.read_csv(data_path)
            true_labels = np.load(true_path)
            pred_labels = np.load(pred_path)
            
            return data_df, true_labels, pred_labels
        
        # Load from test results file (already aligned)
        test_results = pd.read_csv(test_results_path)
        
        # Extract the aligned data
        data_df = test_results[['line_name']].copy()
        true_labels = test_results['true_label'].values
        pred_labels = test_results['pred_label'].values
        
        return data_df, true_labels, pred_labels
        
    except FileNotFoundError as e:
        raise AnalysisError(f"Supervised pipeline results missing: {e}")

def load_unsupervised_data():
    """Load and validate unsupervised model results."""
    try:
        # my original code expected line names from a CSV and cluster labels from numpy
        # but the actual implementation saves line names differently
        
        # try the original expected files first
        names_path = OUTPUT_DIR / 'unsupervised_line_names.csv'
        labels_path = OUTPUT_DIR / 'cluster_labels.npy'
        
        if names_path.exists() and labels_path.exists():
            line_names = pd.read_csv(names_path)['line_name'].tolist()
            cluster_labels = np.load(labels_path)
            return line_names, cluster_labels
        
        # fall back to reconstructing from unlabeled dataset
        unlabeled_path = OUTPUT_DIR / 'unlabeled_dataset_full.csv'
        labels_path = OUTPUT_DIR / 'cluster_labels.npy'
        
        if not unlabeled_path.exists() or not labels_path.exists():
            raise FileNotFoundError("Required unsupervised results files not found")
        
        # load the unlabeled dataset to get line names
        unlabeled_df = pd.read_csv(unlabeled_path)
        line_names = unlabeled_df.columns.tolist()
        
        # load cluster labels
        cluster_labels = np.load(labels_path)
        
        return line_names, cluster_labels
        
    except FileNotFoundError as e:
        raise AnalysisError(f"Unsupervised pipeline results missing: {e}")

def validate_data_consistency(data_df, true_labels, pred_labels):
    """Ensure all data arrays have consistent dimensions."""
    lengths = [len(data_df), len(true_labels), len(pred_labels)]
    if not all(length == lengths[0] for length in lengths):
        raise AnalysisError(
            f"Data dimension mismatch: dataset={lengths[0]}, "
            f"true_labels={lengths[1]}, pred_labels={lengths[2]}"
        )

def validate_unsupervised_consistency(line_names, cluster_labels):
    """Ensure line names and cluster labels are properly aligned."""
    if len(line_names) != len(cluster_labels):
        raise AnalysisError(
            f"Unsupervised data mismatch: {len(line_names)} line names "
            f"but {len(cluster_labels)} cluster labels"
        )

def analyze_supervised_criticality():
    """Identify critical lines from supervised model predictions."""
    print("\n" + "="*60)
    print("  Supervised Model: Critical Line Analysis")
    print("="*60)

    try:
        data_df, true_labels, pred_labels = load_supervised_data()
        validate_data_consistency(data_df, true_labels, pred_labels)
        
        # add the labels to dataframe for analysis
        analysis_df = data_df.copy()
        analysis_df['true_label'] = true_labels
        analysis_df['pred_label'] = pred_labels
        
        # find correctly identified faults (true positives)
        fault_predictions = analysis_df[
            (analysis_df['true_label'] == 1) & 
            (analysis_df['pred_label'] == 1)
        ]

        if fault_predictions.empty:
            print("No faults were correctly identified by the supervised model.")
            return

        # analyze which lines are most frequently identified as critical
        critical_line_frequency = fault_predictions['line_name'].value_counts()
        
        print("Lines most frequently identified in fault conditions:")
        print("-" * 50)
        for line_name, count in critical_line_frequency.head(10).items():
            percentage = (count / len(fault_predictions)) * 100
            print(f"{line_name:20} | {count:4d} occurrences ({percentage:5.1f}%)")
        
        print(f"\nTotal fault instances correctly identified: {len(fault_predictions)}")
        
    except AnalysisError as e:
        print(f"Analysis failed: {e}")
        print("Please run the Supervised Pipeline (Option 1) first.")

def analyze_unsupervised_criticality():
    """Identify critical lines from unsupervised clustering results."""
    print("\n" + "="*60)
    print("  Unsupervised Model: Critical Line Analysis")
    print("="*60)

    try:
        line_names, cluster_labels = load_unsupervised_data()
        
        # the cluster labels correspond to timestamps/rows, not individual lines
        # need to analyze which lines show the most variation in anomalous timestamps
        
        # load the unlabeled dataset to get the actual data
        unlabeled_path = OUTPUT_DIR / 'unlabeled_dataset_full.csv'
        if not unlabeled_path.exists():
            raise AnalysisError("Unlabeled dataset not found")
            
        unlabeled_df = pd.read_csv(unlabeled_path)
        
        # valdiate consistency
        if len(cluster_labels) != len(unlabeled_df):
            raise AnalysisError(
                f"Cluster labels ({len(cluster_labels)}) don't match "
                f"dataset rows ({len(unlabeled_df)})"
            )
        
        # identify anomalous cluster (smallest group)
        cluster_counts = np.bincount(cluster_labels)
        anomalous_cluster_id = np.argmin(cluster_counts)
        
        print(f"Anomalous cluster identified: Cluster {anomalous_cluster_id}")
        print(f"Size: {cluster_counts[anomalous_cluster_id]} instances "
              f"({cluster_counts[anomalous_cluster_id]/len(cluster_labels)*100:.1f}%)")
        
        # get data points in anomalous cluster
        anomalous_mask = cluster_labels == anomalous_cluster_id
        anomalous_data = unlabeled_df[anomalous_mask]
        
        if anomalous_data.empty:
            print("No data points were assigned to the anomalous cluster.")
            return

        # analyze which lines show highest variation in anomalous states
        # Calculate mean absolute values for each line in anomalous states
        line_importance = anomalous_data.abs().mean().sort_values(ascending=False)
        
        # take top 15% of lines as critical
        num_critical = max(1, int(len(line_importance) * 0.15))
        critical_lines = line_importance.head(num_critical).index.tolist()

        print(f"\nLines identified as most critical ({len(critical_lines)} total):")
        print("-" * 50)
        for i, line in enumerate(critical_lines):
            importance = line_importance[line]
            print(f"{i+1:2d}. {line:15} | Importance: {importance:.2f}")
            
    except AnalysisError as e:
        print(f"Analysis failed: {e}")
        print("Please run the Unsupervised Pipeline (Option 2) first.")

def main():
    """Run both analysis methods."""
    analyze_supervised_criticality()
    analyze_unsupervised_criticality()

if __name__ == '__main__':
    main()