import os
import pandas as pd
import numpy as np
from src.pref_models import fit_bradley_terry

def calculate_bt_convergence(experiment_name, base_path='experiments'):
    """
    Calculates the convergence of Bradley-Terry weights for a given experiment.
    
    Args:
        experiment_name (str): Name of the experiment folder.
        base_path (str): Path to the experiments directory.
        
    Returns:
        pd.DataFrame: DataFrame containing weights for each color at each iteration.
    """
    file_path = os.path.join(base_path, experiment_name, 'scores.csv')
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    
    # Determine winner based on lower perplexity score
    # We assume 'score_ppl_a' and 'score_ppl_b' exist.
    # Lower PPL is better.
    df['winner'] = np.where(df['score_ppl_a'] > df['score_ppl_b'], df['item_a'], df['item_b'])
    
    # Get all unique items (colors)
    colors = sorted(list(set(df['item_a'].unique()) | set(df['item_b'].unique())))
    
    # Group by template
    # We want to iterate through templates in the order they appear or some fixed order.
    # Assuming the order in CSV or just list of unique templates.
    templates = df['template'].unique()
    
    weights_history = []
    
    # Accumulate data
    for i, _ in enumerate(templates, 1):
        current_templates = templates[:i]
        subset_df = df[df['template'].isin(current_templates)]
        
        # fit_bradley_terry returns (ranking_df, wins_matrix)
        # ranking_df has columns ['Color', 'BT_Score']
        try:
            ranking, _ = fit_bradley_terry(subset_df, colors)
            
            # Create a dictionary of weights for this iteration
            weights = {'iteration': i, 'num_templates': i}
            for _, row in ranking.iterrows():
                # Convert log-weights (beta) to "probabilities" (e^beta)
                weights[row['Color']] = np.exp(row['BT_Score'])
                
            weights_history.append(weights)
        except Exception as e:
            print(f"Error fitting BT at iteration {i} for {experiment_name}: {e}")
            
    return pd.DataFrame(weights_history)

color_map = {
    'red': '#d62728',      # Set 1 red
    'blue': '#1f77b4',     # Set 1 blue
    'green': '#2ca02c',    # Set 1 green
    'yellow': '#bcbd22',   # Set 1 olive (readable yellow)
    'purple': '#9467bd',   # Set 1 purple
    'orange': '#ff7f0e',   # Set 1 orange
    'brown': '#8c564b',    # Set 1 brown
    'pink': '#e377c2',     # Set 1 pink
    'gray': '#7f7f7f',     # Set 1 gray
    'cyan': '#17becf',     # Set 1 cyan
}

def get_model_name(experiment_name):
    """
    Maps experiment folder name to a readable model name.
    e.g. qwen0_5I_all_metrics -> Qwen2.5-0.5B-Instruct
         qwen7_all_metrics -> Qwen2.5-7B
    """
    # Remove suffix
    clean_name = experiment_name.replace('_all_metrics', '')
    # Remove prefix
    clean_name = clean_name.replace('qwen', '')
    
    is_instruct = False
    if clean_name.endswith('I'):
        is_instruct = True
        clean_name = clean_name[:-1]
    
    # Replace underscore with dot for size (specifically converts 0_5 to 0.5)
    size = clean_name.replace('_', '.')
    
    model_name = f"Qwen2.5-{size}B"
    if is_instruct:
        model_name += "-Instruct"
        
    return model_name
