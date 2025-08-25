"""
Configuration settings for Power System Fault Analysis
"""
from pathlib import Path

# Directory structure
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'

# File patterns and extensions
CSV_PATTERN = '*.csv'
LINE_PREFIX = 'LI_'
TRANSFORMER_PREFIX = 'TRF_'

# Model parameters
class SupervisedConfig:
    EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    TRAIN_SPLIT = 0.8
    FAULT_THRESHOLD = 50.0  # Power delta threshold for fault detection
    
class UnsupervisedConfig:
    MAX_CLUSTERS = 6
    RANDOM_STATE = 42
    N_INIT = 10
    PCA_COMPONENTS = 2
    
class ReinforcementConfig:
    EPISODES = 200
    OVERLOAD_THRESHOLD = 1000.0
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.95
    EPSILON_DECAY = 0.995
    MIN_EPSILON = 0.01
    
# Visualization settings
class PlotConfig:
    DPI = 300
    FIGURE_SIZE = (10, 8)
    CONFUSION_MATRIX_SIZE = (8, 6)
    COLORS = {
        'normal': '#1f77b4',
        'anomaly': '#d62728',
        'primary': '#4A86E8',
        'secondary': '#FF9900',
        'tertiary': '#999999'
    }
    
# Analysis settings
class AnalysisConfig:
    TOP_LINES_DISPLAY = 10
    CRITICAL_LINE_THRESHOLD = 0.15  # Top 15% of lines
    GROUND_TRUTH_CRITICAL = ['3', '17', '28']  # Line numbers
    
# File paths
class FilePaths:
    # Input files
    LABELED_DATASET = OUTPUT_DIR / 'labeled_dataset_full.csv'
    UNLABELED_DATASET = OUTPUT_DIR / 'unlabeled_dataset_full.csv'
    COMBINED_DATASET = OUTPUT_DIR / 'combined_dataset.csv'
    
    # Model outputs
    SUPERVISED_MODEL = OUTPUT_DIR / 'fault_classifier.pth'
    SUPERVISED_TRUE = OUTPUT_DIR / 'supervised_true.npy'
    SUPERVISED_PRED = OUTPUT_DIR / 'supervised_pred.npy'
    SUPERVISED_RESULTS = OUTPUT_DIR / 'supervised_test_results.csv'
    
    CLUSTER_LABELS = OUTPUT_DIR / 'cluster_labels.npy'
    ANOMALOUS_CLUSTER = OUTPUT_DIR / 'anomalous_cluster_label.npy'
    UNSUPERVISED_LINES = OUTPUT_DIR / 'unsupervised_line_names.csv'
    
    Q_TABLE = OUTPUT_DIR / 'q_table.npy'
    REWARD_HISTORY = OUTPUT_DIR / 'reward_history.npy'
    LINE_NAMES = OUTPUT_DIR / 'line_names.txt'
    
    # Metrics
    SUPERVISED_METRICS = OUTPUT_DIR / 'supervised_metrics.npy'
    UNSUPERVISED_METRICS = OUTPUT_DIR / 'unsupervised_metrics.npy'
    RL_METRICS = OUTPUT_DIR / 'rl_metrics.npy'
    
    # Visualizations
    CONFUSION_MATRIX = OUTPUT_DIR / 'figure_1_confusion_matrix.png'
    CLUSTER_PLOT = OUTPUT_DIR / 'figure_2_unsupervised_clusters.png'
    RL_POLICY_PLOT = OUTPUT_DIR / 'figure_3_reinforcement_policy.png'
    PERFORMANCE_CHART = OUTPUT_DIR / 'critical_line_performance_chart.png'
    SILHOUETTE_PLOT = OUTPUT_DIR / 'silhouette_scores.png'

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)