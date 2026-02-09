import datetime
import os
import json
import itertools
import pandas as pd

from model import get_scores_many_options, load_model, get_perplexity_scores, get_single_token_prob
from pref_models import fit_bradley_terry

def collect_metrics_for_template(model, tokenizer, colors, template):
    results = []
    is_instruct = "Instruct" in getattr(model.config, "_name_or_path", "")

    for color_a, color_b in itertools.permutations(colors, 2):
        prompt = template.format(A=color_a, B=color_b)
        if is_instruct and tokenizer.chat_template:
             messages = [{"role": "user", "content": prompt}]
             try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
             except Exception as e:
                # Fallback if chat template issues
                pass
        
        # 1. Single Token Probability
        # We check " {Option}" (space + option) as the standard single token representation
        score_single_a, score_single_b = get_single_token_prob(model, tokenizer, prompt, " " + color_a, " " + color_b)
        
        # 2. Group Probability (via get_winner)
        # We discard the winner return value as requested
        score_group_a, score_group_b = get_scores_many_options(model, tokenizer, prompt, color_a, color_b)
        
        # 3. Perplexity (via get_choice_via_scoring)
        # Note: Lower perplexity/loss is better.
        score_ppl_a, score_ppl_b = get_perplexity_scores(model, tokenizer, prompt, color_a, color_b)
        
        row = {
            "item_a": color_a,
            "item_b": color_b,
            "template": template,
            "score_single_a": score_single_a,
            "score_single_b": score_single_b,
            "score_group_a": score_group_a,
            "score_group_b": score_group_b,
            "score_ppl_a": score_ppl_a,
            "score_ppl_b": score_ppl_b
        }
        results.append(row)

    return pd.DataFrame(results)

def collect_preference_data(model_name, alternatives, templates, exp_name=None):
    if exp_name is None:
        exp_name = "data_collection_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment directory
    dir_path = os.path.join("experiments", exp_name)
    os.makedirs(dir_path, exist_ok=True)
    config = {
        "model_name": model_name,
        "alternatives": alternatives,
        "templates": templates,
    }
    with open(os.path.join(dir_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"Loading model {model_name}...")
    model, tokenizer = load_model(model_name)
    results = pd.DataFrame()
    
    for i, template in enumerate(templates):
        if i % 10 == 0:
            print(f"Processing template {i+1}/{len(templates)}...")
        df_results = collect_metrics_for_template(model, tokenizer, alternatives, template)
        results = pd.concat([results, df_results], ignore_index=True)
        
    # Save results
    dir_path = os.path.join("experiments", exp_name)
    os.makedirs(dir_path, exist_ok=True)
    results.to_csv(os.path.join(dir_path, f"scores.csv"), index=False)
    
    print(f"Data collection complete. Scored {len(results)} comparisons.")
    
    return results

def fit_pref_models(exp_name, colors=None):
    """
    Loads scores.csv from experiment folder and fits Bradley-Terry models.
    """
    dir_path = os.path.join("experiments", exp_name)
    file_path = os.path.join(dir_path, "scores.csv")
    
    if not os.path.exists(file_path):
        print(f"Error: Results file not found at {file_path}")
        return

    print(f"Loading results from {file_path}...")
    results_df = pd.read_csv(file_path)
    
    # Infer colors if not provided
    if colors is None:
        colors = sorted(list(set(results_df['item_a'].unique()) | set(results_df['item_b'].unique())))
        print(f"Inferred {len(colors)} items from data: {colors}")
        
    fit_bt_for_all_metrics(results_df, colors, exp_name)

def fit_bt_for_all_metrics(results_df, colors, exp_name):
    """
    Fits a Bradley-Terry model for each scoring metric (Single Token, Group, Perplexity).
    """
    # Metrics to analyze: (name -> (col_a, col_b))
    metrics = {
        "single": ("score_single_a", "score_single_b"),
        "group": ("score_group_a", "score_group_b"),
        "ppl": ("score_ppl_a", "score_ppl_b")
    }
    
    dir_path = os.path.join("experiments", exp_name)
    
    for metric_name, (col_a, col_b) in metrics.items():
        print(f"Fitting BT model for metric: {metric_name}...")
        df = results_df.copy()
        
        # Determine winner
        # For all our metrics, Higher is Better (Perplexity is negative loss)
        # Handle ties by defaulting to B (unlikely with floats, but good to know)
        df['winner'] = df.apply(lambda row: row['item_a'] if row[col_a] > row[col_b] else row['item_b'], axis=1)
        
        try:
            ranking, _ = fit_bradley_terry(df, colors)
            out_file = os.path.join(dir_path, f"rankings_{metric_name}.csv")
            ranking.to_csv(out_file, index=False)
            print(f"Saved {metric_name} rankings to {out_file}")
            print(f"Top 3 for {metric_name}: {list(ranking['Color'].head(3))}")
        except Exception as e:
            print(f"Error fitting BT for {metric_name}: {e}")
