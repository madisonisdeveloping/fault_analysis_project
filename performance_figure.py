import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from pathlib import Path

def calculate_metrics(predicted_lines, true_lines):
    """Calculates precision, recall, and f1-score for critical line identification."""
    predicted_set = set(predicted_lines)
    true_set = set(true_lines)

    if not predicted_set and not true_set:
        return {"precision": 1.0, "recall": 1.0, "f1-score": 1.0}
    if not predicted_set or not true_set:
        return {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}

    true_positives = len(predicted_set.intersection(true_set))
    
    # Precision = TP / (TP + FP) = TP / len(predicted_set)
    precision = true_positives / len(predicted_set) if len(predicted_set) > 0 else 0
    
    # Recall = TP / (TP + FN) = TP / len(true_set)
    recall = true_positives / len(true_set) if len(true_set) > 0 else 0
    
    # F1-Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1-score": f1}

def get_ground_truth_lines():
    """Get ground truth critical lines from actual data."""
    # first let's try to get actual line names from the labeled dataset
    labeled_path = Path('output/labeled_dataset_full.csv')
    
    if labeled_path.exists():
        try:
            df = pd.read_csv(labeled_path)
            unique_lines = df['line_name'].unique()
            
            # look for lines that contain our critical numbers
            critical_numbers = ['3', '17', '28']
            ground_truth = []
            
            for line in unique_lines:
                # check if line name contains any of our critical numbers
                line_parts = str(line).split('_')
                for part in line_parts:
                    if part in critical_numbers:
                        ground_truth.append(line)
                        break
            
            if ground_truth:
                print(f"Found ground truth lines in data: {ground_truth}")
                return ground_truth
                
        except Exception as e:
            print(f"Could not extract ground truth from labeled data: {e}")
    
    # fallback to simple naming
    ground_truth = ['LI_3', 'LI_17', 'LI_28']
    print(f"Using fallback ground truth: {ground_truth}")
    return ground_truth

def generate_performance_comparison_figure(save_path='output/critical_line_performance_chart.png'):
    """
    Evaluates Supervised, Unsupervised, and Reinforcement Learning models on their
    ability to identify a ground-truth set of critical power lines.
    """
    print("Starting critical line identification performance comparison...")

    # Get ground truth from actual data
    GROUND_TRUTH_CRITICAL_LINES = get_ground_truth_lines()
    
    print(f"Evaluating against ground truth of {len(GROUND_TRUTH_CRITICAL_LINES)} critical lines: {GROUND_TRUTH_CRITICAL_LINES}")

    all_results = []
    
    # 1. Evaluate Supervised Model ---
    print("\nEvaluating Supervised Model...")
    try:
        # try the test results file first (more reliable - implemented this too late)
        test_results_path = 'output/supervised_test_results.csv'
        if os.path.exists(test_results_path):
            test_results_df = pd.read_csv(test_results_path)
            true_positives_df = test_results_df[
                (test_results_df['true_label'] == 1) & (test_results_df['pred_label'] == 1)
            ]
        else:
            # fallback to original method
            data_df = pd.read_csv('output/labeled_dataset_full.csv')
            true_labels = np.load('output/supervised_true.npy')
            pred_labels = np.load('output/supervised_pred.npy')
            
            # create test results from available data
            test_results_df = pd.DataFrame({
                'line_name': data_df['line_name'][-len(true_labels):],  # assume test set is at the end
                'true_label': true_labels,
                'pred_label': pred_labels
            })
            true_positives_df = test_results_df[
                (test_results_df['true_label'] == 1) & (test_results_df['pred_label'] == 1)
            ]
        
        supervised_critical_lines = true_positives_df['line_name'].unique().tolist()
        
        metrics = calculate_metrics(supervised_critical_lines, GROUND_TRUTH_CRITICAL_LINES)
        metrics['model'] = 'Supervised'
        all_results.append(metrics)
        print(f"  - Identified {len(supervised_critical_lines)} lines. F1-Score: {metrics['f1-score']:.2f}")
        print(f"  - Lines: {supervised_critical_lines[:5]}{'...' if len(supervised_critical_lines) > 5 else ''}")

    except FileNotFoundError:
        print("  - WARNING: Supervised results not found. Skipping.")

    # 2. Evaluate Unsupervised Model ---
    print("\nEvaluating Unsupervised Model...")
    try:
        unlabeled_df = pd.read_csv('output/unlabeled_dataset_full.csv')
        cluster_labels = np.load('output/cluster_labels.npy')

        # Validate data consistency
        if len(cluster_labels) != len(unlabeled_df):
            print(f"  - Data mismatch: {len(cluster_labels)} labels vs {len(unlabeled_df)} rows")
            raise IndexError("Cluster labels don't match dataset size")

        unique, counts = np.unique(cluster_labels, return_counts=True)
        anomalous_cluster_label = unique[np.argmin(counts)]
        anomalous_indices = np.where(cluster_labels == anomalous_cluster_label)[0]
        
        if len(anomalous_indices) > 0:
            anomalous_data = unlabeled_df.iloc[anomalous_indices]
            line_importance = anomalous_data.abs().mean().sort_values(ascending=False)
            
            # Identify top 15% of lines by importance as critical
            num_to_select = max(1, int(len(line_importance) * 0.15))
            unsupervised_critical_lines = line_importance.head(num_to_select).index.tolist()
        else:
            unsupervised_critical_lines = []

        metrics = calculate_metrics(unsupervised_critical_lines, GROUND_TRUTH_CRITICAL_LINES)
        metrics['model'] = 'Unsupervised'
        all_results.append(metrics)
        print(f"  - Identified {len(unsupervised_critical_lines)} lines. F1-Score: {metrics['f1-score']:.2f}")
        print(f"  - Lines: {unsupervised_critical_lines}")

    except (FileNotFoundError, KeyError, IndexError) as e:
        print(f"  - WARNING: Unsupervised results error: {e}. Skipping.")

    # 3. eval reinforcement Learning Model ---
    print("\nEvaluating Reinforcement Learning Model...")
    try:
        q_table = np.load('output/q_table.npy', allow_pickle=True).item()
        
        # try to get line names from multiple sources
        line_names_file = 'output/line_names.txt'
        if os.path.exists(line_names_file):
            with open(line_names_file, 'r') as f:
                rl_line_names = [line.strip() for line in f.readlines()]
        else:
            # fallback: get line names from unlabeled dataset
            unlabeled_df = pd.read_csv('output/unlabeled_dataset_full.csv')
            rl_line_names = unlabeled_df.columns.tolist()
            
        overload_counts = defaultdict(int)
        for state in q_table.keys():
            if sum(state) > 0:  # Only states with overloads
                for line_index, status in enumerate(state):
                    if status == 1 and line_index < len(rl_line_names):
                        overload_counts[rl_line_names[line_index]] += 1
        
        # rank lines by overload count and take top 15%
        if overload_counts:
            sorted_lines = sorted(overload_counts.items(), key=lambda item: item[1], reverse=True)
            num_to_select = max(1, int(len(sorted_lines) * 0.15))
            rl_critical_lines = [line[0] for line in sorted_lines[:num_to_select]]
        else:
            rl_critical_lines = []
        
        metrics = calculate_metrics(rl_critical_lines, GROUND_TRUTH_CRITICAL_LINES)
        metrics['model'] = 'Reinforcement Learning'
        all_results.append(metrics)
        print(f"  - Identified {len(rl_critical_lines)} lines. F1-Score: {metrics['f1-score']:.2f}")
        print(f"  - Lines: {rl_critical_lines}")

    except FileNotFoundError as e:
        print(f"  - WARNING: Reinforcement Learning results not found: {e}. Skipping.")

    # 4. generate the Bar Chart ---
    if not all_results:
        print("\nNo model results found. Cannot generate chart. Please run pipelines 1, 2, and 3 first.")
        return
        
    results_df = pd.DataFrame(all_results)
    print("\nGenerating comparison plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    n_models = len(results_df)
    index = np.arange(n_models)
    bar_width = 0.25

    bars1 = ax.bar(index - bar_width, results_df['precision'], bar_width, label='Precision', color='#4A86E8')
    bars2 = ax.bar(index, results_df['recall'], bar_width, label='Recall', color='#FF9900')
    bars3 = ax.bar(index + bar_width, results_df['f1-score'], bar_width, label='F1-Score', color='#999999')

    for bar_group in [bars1, bars2, bars3]:
        ax.bar_label(bar_group, padding=3, fmt='%.2f', fontsize=10)

    ax.set_title('Model Comparison for Critical Line Identification', fontsize=18, pad=20)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticks(index)
    ax.set_xticklabels(results_df['model'], fontsize=12)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    
    # add ground truth info as text
    ground_truth_text = f"Ground Truth: {', '.join(GROUND_TRUTH_CRITICAL_LINES)}"
    ax.text(0.02, 0.98, ground_truth_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\nPerformance comparison chart saved to: {save_path}")
    plt.close(fig)

if __name__ == '__main__':
    generate_performance_comparison_figure()