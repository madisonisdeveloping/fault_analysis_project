"""
Main Entry Point for Power System Fault Analysis
Provides interactive menu for running ML pipelines and generating reports.
"""

from supervised import run_supervised_pipeline
from labeled_data import merge_labeled_from_folder
from unsupervised import auto_cluster, load_unlabeled_dataset
from labeled_data_unsupervised import merge_unlabeled_from_folder
from reinforcement import run_q_learning_pipeline
from poster_graphics import (plot_confusion_matrix, plot_unsupervised_clusters, 
                           plot_reinforcement_learning_results)
from poster_metrics import display_metrics
from critical_line_analysis import (analyze_supervised_criticality, 
                                  analyze_unsupervised_criticality)
from performance_figure import generate_performance_comparison_figure
import config

def print_header():
    """Display program header."""
    print("\n" + "="*60)
    print("         AI Sustein Fault Analysis Project")
    print("="*60)

def print_menu():
    """Display main menu options."""
    print("\nWorkflow: Run pipelines (1-3) before analysis (4-8)\n")
    
    menu_items = [
        ("1", "Run Supervised Learning Pipeline", "Train neural network for fault classification"),
        ("2", "Run Unsupervised Clustering Pipeline", "Discover anomalous system states"),
        ("3", "Run Reinforcement Learning Pipeline", "Learn optimal fault mitigation strategies"),
        ("-", "-" * 50, ""),
        ("4", "Generate All Poster Figures", "Create visualization suite for reporting"),
        ("5", "Display Performance Metrics", "Compare model accuracy and efficiency"),
        ("6", "Analyze Critical Lines", "Identify most vulnerable power lines"),
        ("7", "Generate Performance Comparison", "Cross-model critical line identification"),
        ("8", "Exit", ""),
    ]
    
    for option, title, description in menu_items:
        if option == "-":
            print(title)
        else:
            print(f"{option}. {title}")
            if description:
                print(f"   {description}")

def get_user_choice():
    """Get and validate user input."""
    while True:
        choice = input("\nEnter your choice (1-8): ").strip()
        if choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
            return choice
        print("Invalid choice. Please enter a number between 1 and 8.")

def run_supervised_workflow():
    """Execute supervised learning workflow."""
    print("\n" + "="*50)
    print("  SUPERVISED LEARNING WORKFLOW")
    print("="*50)
    print("Creating labeled dataset from power flow data...")
    merge_labeled_from_folder()
    print("\nTraining neural network classifier...")
    run_supervised_pipeline()
    print("Supervised learning workflow completed.")

def run_unsupervised_workflow():
    """Execute unsupervised clustering workflow."""
    print("\n" + "="*50)
    print("  UNSUPERVISED CLUSTERING WORKFLOW")
    print("="*50)
    print("Preparing unlabeled dataset for clustering...")
    merge_unlabeled_from_folder()
    print("\nApplying clustering algorithms...")
    data = load_unlabeled_dataset(str(config.FilePaths.UNLABELED_DATASET))
    auto_cluster(data)
    print("Unsupervised clustering workflow completed.")

def run_reinforcement_workflow():
    """Execute reinforcement learning workflow."""
    print("\n" + "="*50)
    print("  REINFORCEMENT LEARNING WORKFLOW")
    print("="*50)
    print("Training Q-learning agent for fault mitigation...")
    run_q_learning_pipeline()
    print("Reinforcement learning workflow completed.")

def generate_all_figures():
    """Create all visualization figures."""
    print("\n" + "="*50)
    print("  GENERATING VISUALIZATION SUITE")
    print("="*50)
    
    figures = [
        ("Confusion Matrix", plot_confusion_matrix),
        ("Cluster Visualization", plot_unsupervised_clusters),
        ("RL Policy Analysis", plot_reinforcement_learning_results),
    ]
    
    for name, func in figures:
        print(f"Creating {name}...")
        try:
            func()
        except Exception as e:
            print(f"Warning: Failed to generate {name}: {e}")
    
    print("Visualization suite generation completed.")

def show_performance_metrics():
    """Display comprehensive performance metrics."""
    print("\n" + "="*50)
    print("  PERFORMANCE METRICS ANALYSIS")
    print("="*50)
    display_metrics()

def analyze_critical_infrastructure():
    """Perform critical line analysis."""
    print("\n" + "="*50)
    print("  CRITICAL INFRASTRUCTURE ANALYSIS")
    print("="*50)
    analyze_supervised_criticality()
    analyze_unsupervised_criticality()

def generate_comparison_report():
    """Generate cross-model performance comparison."""
    print("\n" + "="*50)
    print("  CROSS-MODEL PERFORMANCE COMPARISON")
    print("="*50)
    print("Ensuring labeled dataset is available...")
    merge_labeled_from_folder()
    print("Generating performance comparison chart...")
    generate_performance_comparison_figure()
    print("Performance comparison completed.")

def handle_menu_choice(choice):
    """Execute selected menu option."""
    actions = {
        '1': run_supervised_workflow,
        '2': run_unsupervised_workflow,
        '3': run_reinforcement_workflow,
        '4': generate_all_figures,
        '5': show_performance_metrics,
        '6': analyze_critical_infrastructure,
        '7': generate_comparison_report,
        '8': lambda: None  # Exit handled in main loop
    }
    
    if choice in actions and choice != '8':
        try:
            actions[choice]()
        except Exception as e:
            print(f"\nError executing option {choice}: {e}")
            print("Please check that all required data files are present.")

def main():
    """Main program loop."""
    print_header()
    
    while True:
        print_menu()
        choice = get_user_choice()
        
        if choice == '8':
            break
            
        handle_menu_choice(choice)

if __name__ == "__main__":
    main()