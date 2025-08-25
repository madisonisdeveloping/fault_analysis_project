import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from collections import defaultdict
import math
import time

def generate_combined_dataset(data_folder='data/', output_path='output/combined_dataset.csv'):
    os.makedirs('output', exist_ok=True)
    transposed_dfs, all_line_names = [], []
    csv_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])
    if not csv_files: raise ValueError("No CSV files found.")
    line_pattern = r'^LI_\d+.*'
    for i, filename in enumerate(csv_files):
        df = pd.read_csv(os.path.join(data_folder, filename))
        df_filtered = df[df['Name'].str.match(line_pattern, na=False)].copy()
        if i == 0: all_line_names = df_filtered['Name'].tolist()
        df_transposed = df_filtered.drop(columns=['Name']).transpose()
        df_transposed.columns = all_line_names
        transposed_dfs.append(df_transposed)
    full_df = pd.concat(transposed_dfs, ignore_index=True)
    full_df.to_csv(output_path, index=False)
    return full_df

class PowerGridEnv:
    def __init__(self, data_df, overload_threshold=1000.0):
        self.all_data = data_df
        self.num_timesteps, self.num_lines = self.all_data.shape
        self.line_names = self.all_data.columns.tolist()
        self.overload_threshold = overload_threshold

    def get_state(self, t): return tuple((self.all_data.iloc[t] > self.overload_threshold).astype(int))

    def get_simulated_reward_and_next_state(self, t, action):
        if t + 1 >= self.num_timesteps: return self.get_state(t), 0
        next_flows = self.all_data.iloc[t + 1].copy()
        if 0 < action <= self.num_lines: next_flows.iloc[action - 1] = 0
        next_state = tuple((next_flows > self.overload_threshold).astype(int))
        reward = 10 if sum(next_state) == 0 and sum(self.get_state(t)) > 0 else 1 if sum(next_state) < sum(self.get_state(t)) else -1
        return next_state, reward

def analyze_and_display_policy(q_table, line_names):
    """
    Analyzes the trained Q-table to extract and display the learned policy,
    including critical lines and best actions for overload states.
    """
    print("\n" + "="*50)
    print("      Reinforcement Learning Policy Analysis")
    print("="*50)
    
    # Find all states where an overload occurred
    overload_states = [s for s in q_table.keys() if sum(s) > 0]
    
    # Sort states by the number of overloads, descending
    sorted_states = sorted(overload_states, key=sum, reverse=True)

    print("\nFinal Output 1: Best actions for top overload states")
    if not sorted_states:
        print("No overload states were encountered during training.")
    else:
        # Display the top 20 most severe overload states and the learned action
        print(f"{'State':<50} | {'Best Action'}")
        print("-" * 75)
        for i, state in enumerate(sorted_states[:20]):
            best_action_index = np.argmax(q_table[state])
            # Action 0 is "Do Nothing", otherwise it corresponds to the line name
            action_desc = "Do Nothing" if best_action_index == 0 else f"Trip {line_names[best_action_index - 1]}"
            # Truncate long state tuples for cleaner printing
            state_str = str(state)
            if len(state_str) > 48: state_str = state_str[:45] + "..."
            print(f"{state_str:<50} | {action_desc}")
        if len(sorted_states) > 20:
            print("... (and more states not shown)")
            
    # Count how often each line was part of an overload state
    overload_counts = defaultdict(int)
    for state in overload_states:
        for line_index, status in enumerate(state):
            if status == 1: # If the line is overloaded in this state
                overload_counts[line_index] += 1

    print("\nFinal Output 2: Most frequently overloaded (critical) lines")
    if not overload_counts:
        print("No lines were found to be overloaded.")
    else:
        # Sort lines by how often they were overloaded
        sorted_lines = sorted(overload_counts.items(), key=lambda item: item[1], reverse=True)
        for line_index, count in sorted_lines:
            print(f"  - {line_names[line_index]}: seen in {count} unique overload states")
    print("="*50)


def run_q_learning_pipeline(episodes=200):
    start_time = time.time()
    combined_df = generate_combined_dataset()
    env = PowerGridEnv(data_df=combined_df)
    q_table = defaultdict(lambda: np.zeros(env.num_lines + 1))
    reward_history, epsilon = [], 1.0
    
    for ep in range(episodes):
        total_reward = 0
        for t in range(env.num_timesteps - 1):
            state = env.get_state(t)
            action = np.random.randint(env.num_lines + 1) if np.random.random() < epsilon else np.argmax(q_table[state])
            next_state, reward = env.get_simulated_reward_and_next_state(t, action)
            total_reward += reward
            td_error = reward + 0.95 * np.max(q_table[next_state]) - q_table[state][action]
            q_table[state][action] += 0.1 * td_error
        reward_history.append(total_reward)
        epsilon = max(0.01, epsilon * 0.995)
        if (ep + 1) % 20 == 0: print(f"Ep {ep+1}/{episodes} | Reward: {total_reward}")

    end_time = time.time()
    total_time = end_time - start_time
    
    # Call the analysis function after training is complete
    analyze_and_display_policy(q_table, env.line_names)

    # --- Save final results and metrics ---
    print("\nSaving reinforcement learning results for analysis...")
    np.save('output/q_table.npy', dict(q_table))
    np.save('output/reward_history.npy', np.array(reward_history))
    with open('output/line_names.txt', 'w') as f:
        f.write('\n'.join(env.line_names))
    
    metrics = {
        "training_time_sec": total_time,
        "final_avg_reward": np.mean(reward_history[-10:]),
        "q_table_size_states": len(q_table)
    }
    np.save('output/rl_metrics.npy', metrics)
    print("Reinforcement learning results and metrics saved.")
