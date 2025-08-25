# AI-Powered Power System Fault Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning framework for power system fault detection and critical infrastructure identification using three complementary AI approaches: Supervised Learning, Unsupervised Clustering, and Reinforcement Learning.

## 🎯 Project Overview

This project implements a multi-modal AI system for analyzing power grid faults and identifying critical transmission lines. By combining different machine learning paradigms, the system provides robust fault detection capabilities and strategic insights for power system operators.

### Key Features

- **🧠 Supervised Learning**: Neural network-based binary classification for fault detection
- **🔍 Unsupervised Clustering**: K-means clustering to discover anomalous system states
- **🎮 Reinforcement Learning**: Q-learning agent for optimal fault mitigation strategies
- **📊 Comprehensive Analysis**: Cross-model performance comparison and critical line identification
- **📈 Rich Visualizations**: Confusion matrices, cluster plots, and performance charts

## 🏗️ Architecture

The system processes power flow data through three parallel pipelines:

```
CSV Data Files → Data Processing → Three AI Pipelines → Analysis & Visualization
                      ↓
    ┌─────────────────┼─────────────────┐
    ↓                 ↓                 ↓
Supervised        Unsupervised    Reinforcement
(Fault Class.)    (Anomaly Det.)  (Strategy Learn.)
    ↓                 ↓                 ↓
Neural Network    K-Means Clustering  Q-Learning Agent
    ↓                 ↓                 ↓
    └─────────────────┼─────────────────┘
                      ↓
            Critical Line Analysis
                      ↓
              Performance Metrics
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn
```

### Installation

```bash
git clone https://github.com/yourusername/fault-analysis-project.git
cd fault-analysis-project
pip install -r requirements.txt
```

### Usage

1. **Place your CSV data files** in the `data/` directory
   - Files should contain power flow data with 'Name' column for line identification
   - Time series data should be in subsequent columns

2. **Run the interactive menu**:
   ```bash
   python main.py
   ```

3. **Follow the workflow**:
   - Run pipelines 1-3 to train models
   - Use options 4-8 for analysis and visualization

### Sample Workflow

```python
# Run all three ML pipelines
python -c "
from main import *
run_supervised_workflow()    # Train neural network
run_unsupervised_workflow()  # Discover clusters  
run_reinforcement_workflow() # Learn Q-table
"

# Generate comprehensive analysis
python -c "
from main import *
generate_all_figures()           # Create visualizations
analyze_critical_infrastructure() # Identify critical lines
generate_comparison_report()     # Cross-model comparison
"
```

## 🔬 Machine Learning Approaches

### 1. Supervised Learning Pipeline
- **Model**: Multi-layer neural network with BatchNorm and Dropout
- **Input**: Power flow delta changes between timestamps
- **Output**: Binary classification (Normal/Fault)
- **Features**: Handles variable input dimensions, robust to missing data

### 2. Unsupervised Learning Pipeline  
- **Model**: K-Means clustering with PCA visualization
- **Input**: Raw power flow states across all timestamps
- **Output**: System state clusters and anomaly identification
- **Analysis**: Silhouette scoring for optimal cluster selection

### 3. Reinforcement Learning Pipeline
- **Model**: Q-Learning agent with state-action value function
- **Environment**: Simulated power grid with overload states
- **Actions**: Line tripping decisions for fault mitigation
- **Reward**: System stability improvement and overload reduction

## 📊 Output Analysis

The system generates comprehensive analysis across multiple dimensions:

### Performance Metrics
- **Supervised**: F1-Score, Precision, Recall, Confusion Matrix
- **Unsupervised**: Silhouette Score, Cluster Quality Metrics  
- **Reinforcement**: Cumulative Reward, Policy Convergence
- **Cross-Model**: Critical Line Identification Accuracy

### Visualizations
- `figure_1_confusion_matrix.png` - Supervised model performance
- `figure_2_unsupervised_clusters.png` - Discovered system states
- `figure_3_reinforcement_policy.png` - Learned fault mitigation strategies
- `critical_line_performance_chart.png` - Cross-model comparison

### Critical Infrastructure Analysis
The system identifies the most vulnerable power lines through:
- Frequency analysis of fault occurrences (Supervised)
- Anomalous state participation (Unsupervised)  
- Overload pattern recognition (Reinforcement Learning)

## 📁 Project Structure

```
fault_analysis_project/
├── main.py                     # Interactive menu and workflow orchestration
├── config.py                   # Configuration settings and file paths
├── data_loader.py              # CSV data loading utilities
├── labeled_data.py             # Supervised learning data preparation
├── labeled_data_unsupervised.py # Unsupervised learning data preparation
├── supervised.py               # Neural network training pipeline
├── unsupervised.py            # K-means clustering pipeline  
├── reinforcement.py           # Q-learning training pipeline
├── critical_line_analysis.py  # Cross-model critical line identification
├── performance_figure.py      # Performance comparison visualization
├── poster_graphics.py         # Figure generation for reporting
├── poster_metrics.py          # Metrics display and comparison
├── silhouette_analysis.py     # Cluster quality analysis
├── utils.py                   # Common utilities and error handling
├── data/                      # Input CSV files directory
└── output/                    # Generated results and visualizations
```

## ⚙️ Configuration

Key parameters can be adjusted in `config.py`:

```python
# Model Training
SupervisedConfig.EPOCHS = 20
SupervisedConfig.BATCH_SIZE = 64
SupervisedConfig.FAULT_THRESHOLD = 50.0

# Clustering
UnsupervisedConfig.MAX_CLUSTERS = 6
UnsupervisedConfig.PCA_COMPONENTS = 2

# Reinforcement Learning  
ReinforcementConfig.EPISODES = 200
ReinforcementConfig.OVERLOAD_THRESHOLD = 1000.0
```

## 📈 Results and Performance

The system has been tested on power grid simulation data and demonstrates:

- **High Accuracy**: >90% fault classification accuracy
- **Robust Anomaly Detection**: Effective identification of rare system states
- **Strategic Learning**: Convergent Q-learning policies for fault mitigation
- **Critical Line Identification**: Cross-validated identification of vulnerable infrastructure

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
git clone https://github.com/yourusername/fault-analysis-project.git
cd fault-analysis-project
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🙏 Acknowledgments

- Power system simulation data provider, Nof Yasir of NDSU
- Dr. Di Wu of NDSU for mentoring me
- PyTorch and Scikit-learn communities