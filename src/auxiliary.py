import pandas as pd
import numpy as np
import os
from src.pref_models import fit_bradley_terry

def analyze_bt_convergence(exp_name):
    scores_path = f"experiments/{exp_name}/scores.csv"
    if not os.path.exists(scores_path):
        print(f"File not found: {scores_path}")
        return

    df_scores = pd.read_csv(scores_path)

    # In case the file is empty or malformed
    if df_scores.empty:
        print("Scores file is empty.")
        return

    templates = df_scores['template'].unique()
    methods = [
        ('single', 'score_single_a', 'score_single_b'),
        ('group', 'score_group_a', 'score_group_b'),
        ('ppl', 'score_ppl_a', 'score_ppl_b')
    ]

    # Get all unique items from the dataset
    items = np.unique(np.concatenate([df_scores['item_a'].unique(), df_scores['item_b'].unique()]))

    all_results = []

    for method_name, col_a, col_b in methods:
        print(f"Processing method: {method_name}")
        for i in range(1, len(templates) + 1):
            # Select templates 0 to i-1
            current_templates = templates[:i]
            
            # Filter data
            df_subset = df_scores[df_scores['template'].isin(current_templates)].copy()
            
            # Determine winner (Higher score wins)
            # Handle potential tie or same scores? Usually > is fine, or >=. Previous code used >=.
            df_subset['winner'] = np.where(df_subset[col_a] >= df_subset[col_b], df_subset['item_a'], df_subset['item_b'])
            
            # Fit BT
            ranking, _ = fit_bradley_terry(df_subset, items)
            
            # Collect results
            for _, row in ranking.iterrows():
                all_results.append({
                    'method': method_name,
                    'iteration': i,
                    'num_templates': i,
                    'item': row['Color'],
                    'score': row['BT_Score']
                })

    df_results = pd.DataFrame(all_results)
    output_path = f"experiments/{exp_name}/bt_convergence.csv"
    df_results.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    
    return df_results

def load_experiment_rankings(exp_name):
    """
    Loads all method rankings from an experiment folder into a single DataFrame.
    """
    methods = ['single', 'group', 'ppl']
    dfs = []
    
    for method in methods:
        path = f"experiments/{exp_name}/rankings_{method}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['method'] = method
            dfs.append(df)
        else:
            print(f"Warning: {path} not found.")
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

