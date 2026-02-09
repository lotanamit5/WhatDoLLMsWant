import os
import pandas as pd
import numpy as np
from src.pref_models import fit_bradley_terry

def calculate_bt_convergence(experiment_name, base_path='experiments', method='ppl'):
    """
    Calculates the convergence of Bradley-Terry weights for a given experiment.
    
    Args:
        experiment_name (str): Name of the experiment folder.
        base_path (str): Path to the experiments directory.
        method (str): Metric to use for determining the winner. 
                      One of ['ppl', 'single', 'group'].
                      Default is 'ppl'.
        
    Returns:
        pd.DataFrame: DataFrame containing weights for each color at each iteration.
    """
    # Check for cached file
    cache_file_path = os.path.join(base_path, experiment_name, f'bt_convergence_{method}.csv')
    if os.path.exists(cache_file_path):
        print(f"Loading cached convergence data from {cache_file_path}")
        return pd.read_csv(cache_file_path)

    file_path = os.path.join(base_path, experiment_name, 'scores.csv')
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    
    # Determine winner based on the selected method (Higher Score = Winner)
    if method == 'ppl':
        # Scores are negative loss (higher is better)
        col_a, col_b = 'score_ppl_a', 'score_ppl_b'
    elif method == 'single':
        col_a, col_b = 'score_single_a', 'score_single_b'
    elif method == 'group':
        col_a, col_b = 'score_group_a', 'score_group_b'
    else:
        raise ValueError(f"Unknown method strategy: {method}")

    # Determine winner (Higher score wins for all current metrics)
    df['winner'] = np.where(df[col_a] > df[col_b], df['item_a'], df['item_b'])
    
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
        ranking, _ = fit_bradley_terry(subset_df, colors)
        
        # Create a dictionary of weights for this iteration
        weights = {'iteration': i, 'num_templates': i}
        for _, row in ranking.iterrows():
            # Convert log-weights (beta) to "probabilities" (e^beta)
            weights[row['Color']] = np.exp(row['BT_Score'])
            
        weights_history.append(weights)
            
    result_df = pd.DataFrame(weights_history)
    
    # Save to cache
    result_df.to_csv(cache_file_path, index=False)
    print(f"Saved convergence data to {cache_file_path}")
    
    return result_df




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

# Color Mappings
better_color_map = {
    'Red': '#d62728',      # Set 1 red
    'Blue': '#1f77b4',     # Set 1 blue
    'Green': '#2ca02c',    # Set 1 green
    'Yellow': '#bcbd22',   # Set 1 olive (readable yellow)
    'Purple': '#9467bd',   # Set 1 purple
    'Orange': '#ff7f0e',   # Set 1 orange
    'Brown': '#8c564b',    # Set 1 brown
    'Pink': '#e377c2',     # Set 1 pink
    'Gray': '#7f7f7f',     # Set 1 gray
    'Cyan': '#17becf',     # Set 1 cyan
}

colors_map = {
    "red": "Red",
    "blue": "Blue",
    "green": "Green",
    "yellow": "Yellow",
    "purple": "Purple",
    "orange": "Orange",
    "pink": "Pink",
    "black": "Black",
    "white": "lightgrey",
    "brown": "Brown",
    "gray": "Gray"
}

colors_es_map = {
    "Rojo": "Red",
    "Azul": "Blue",
    "Verde": "Green",
    "Amarillo": "Yellow",
    "Morado": "Purple"
}

colors_zh_map = {
    "红色": "Red",
    "蓝色": "Blue",
    "绿色": "Green",
    "黄色": "Yellow",
    "紫色": "Purple",
}

stocks_map = {
    "Apple": "Gray",
    "Microsoft": "Cyan",
    "Google": "Yellow",
    "Amazon": "Orange",
    "Tesla": "Red",
    "Nvidia": "Green",
    "Meta": "Blue"
}

tickers_map = {
    "AAPL": "Gray",
    "MSFT": "Cyan",
    "GOOG": "Yellow",
    "AMZN": "Orange",
    "TSLA": "Red",
    "NVDA": "Green",
    "META": "Blue"
}