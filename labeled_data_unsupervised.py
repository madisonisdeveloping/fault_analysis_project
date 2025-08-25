import pandas as pd
import os

def merge_unlabeled_from_folder(data_folder='data/', output_path='output/unlabeled_dataset_full.csv'):
    """
    Merges all line data into a single, continuous time-series dataset of raw
    power values for unsupervised learning. This represents the system's state.
    """
    print("Processing files for unsupervised learning (State-Based Method)...")
    file_list = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])
    if not file_list:
        raise ValueError("No CSV files found in the data folder.")

    all_scenario_dfs = []

    for filename in file_list:
        filepath = os.path.join(data_folder, filename)
        try:
            df = pd.read_csv(filepath)
            if 'Name' in df.columns:
                # Filter for lines only
                df_lines_only = df[df['Name'].str.startswith('LI_', na=False)].copy()

                # Set line names as index, then transpose so rows are timestamps
                df_transposed = df_lines_only.set_index('Name').transpose()
                
                # Ensure data is numeric
                df_transposed = df_transposed.apply(pd.to_numeric, errors='coerce')
                all_scenario_dfs.append(df_transposed)
            else:
                print(f"--> Skipping {filename} (missing 'Name' column)")
        except Exception as e:
            print(f"--> Error processing {filename}: {e}")

    if not all_scenario_dfs:
        raise ValueError("No valid data files with line data were found.")

    # Concatenate all dataframes vertically to create one long time series of raw power values.
    # We are no longer using .diff(). We are clustering the system's state at each timestamp
    combined_df = pd.concat(all_scenario_dfs, ignore_index=True)
    
    # fills any potential NaN values
    combined_df = combined_df.fillna(0.0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    print(f"Combined unlabeled dataset saved to {output_path} ({combined_df.shape[0]} timestamps × {combined_df.shape[1]} lines)")

if __name__ == '__main__':
    merge_unlabeled_from_folder()
