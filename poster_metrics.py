import numpy as np
import os

def display_metrics():
    """
    Loads and displays a comparative table of performance metrics
    for all three AI models, with robust error checking.
    """
    print("\n" + "="*75)
    print("          AI Model Performance & Efficiency Metrics")
    print("="*75)

    metric_paths = {
        'Supervised': 'output/supervised_metrics.npy',
        'Unsupervised': 'output/unsupervised_metrics.npy',
        'Reinforcement': 'output/rl_metrics.npy'
    }

    metrics_data = {}
    all_files_found = True

    for model_name, path in metric_paths.items():
        if os.path.exists(path):
            metrics_data[model_name] = np.load(path, allow_pickle=True).item()
        else:
            print(f"--> WARNING: Metrics file for '{model_name}' not found at '{path}'.")
            print(f"--> Please run the '{model_name}' pipeline from the main menu first.")
            metrics_data[model_name] = {}
            all_files_found = False

    if not all_files_found:
        print("\n" + "="*75)
        print("Cannot display full table as some metric files are missing.")
        print("="*75)
        return

    def format_val(value, format_str):
        if isinstance(value, (int, float)):
            return f"{value:{format_str}}"
        return "N/A"

    sup = metrics_data.get('Supervised', {})
    uns = metrics_data.get('Unsupervised', {})
    rl = metrics_data.get('Reinforcement', {})

    print(f"{'Metric':<25} | {'Supervised':<15} | {'Unsupervised':<15} | {'Reinforcement':<15}")
    print("-"*75)
    
    time_sup = format_val(sup.get('training_time_sec'), '.2f')
    time_uns = format_val(uns.get('training_time_sec'), '.2f')
    time_rl = format_val(rl.get('training_time_sec'), '.2f')
    print(f"{'Training Time (sec)':<25} | {time_sup:<15} | {time_uns:<15} | {time_rl:<15}")
    
    params_sup = format_val(sup.get('model_parameters'), ',d')
    states_rl = format_val(rl.get('q_table_size_states'), ',d')
    print(f"{'Model Complexity':<25} | {params_sup:<15} | {'N/A':<15} | {states_rl:<15}")
    print(f"{'(Trainable Params/States)':<25} | {'(params)':>15} | {'':<15} | {'(states)':>15}")

    f1_sup = format_val(sup.get('f1_score'), '.4f')
    sil_uns = format_val(uns.get('silhouette_score'), '.4f')
    rew_rl = format_val(rl.get('final_avg_reward'), '.2f')
    print(f"{'Performance Metric':<25} | {f1_sup} (F1) | {sil_uns} (Sil) | {rew_rl} (Rew)")
    
    print("="*75)

if __name__ == "__main__":
    display_metrics()
