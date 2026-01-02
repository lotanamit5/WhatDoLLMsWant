import datetime
import os
import itertools
import pandas as pd

from metrics import calculate_kendall_distance
from model import get_winner_logits, load_model
from pref_models import fit_bradley_terry


def run_experiment(model, tokenizer, device, colors, template):
    """Runs all pairwise comparisons across all templates."""
    results = []

    for color_a, color_b in itertools.permutations(colors, 2):
        # We test both orders (A vs B) and (B vs A) to mitigate positional bias
        # Ideally, we average the logits or just record both trials.
        # Here we will treat them as separate matches.

        # Trial 1: A vs B
        prompt = template.format(A=color_a, B=color_b)
        winner, _, _ = get_winner_logits(model, tokenizer, prompt, color_a, color_b, device)
        results.append({
            "player_1": color_a,
            "player_2": color_b,
            "winner": winner,
            "template": template
        })

    return pd.DataFrame(results)

def run_full_experiment(model_name, colors, templates, exp_name=None):
    if exp_name is None:
        exp_name = "experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    model, tokenizer, device = load_model(model_name)
    results = pd.DataFrame()
    rankings = []
    for i, template in enumerate(templates):
        df_results = run_experiment(model, tokenizer, device, colors, template)
        results = pd.concat([results, df_results], ignore_index=True)
        ranking, win_matrix = fit_bradley_terry(results, colors)
        print(f"Current rankings: {list(ranking['Color'])}, calculated on {len(results)} comparisons")
        if len(rankings) > 0:
            kendell_tau = calculate_kendall_distance(rankings[-1]['Color'], ranking['Color'])
            print(f"Current Kendall Tau Distance: {kendell_tau}")
        rankings.append(ranking.assign(iteration=i))
        
    # Save results
    rankings = pd.concat(rankings)
    dir_path = os.path.join("experiments", exp_name)
    os.makedirs(dir_path, exist_ok=True)
    results.to_csv(os.path.join(dir_path, f"{exp_name}_answers.csv"), index=False)
    rankings.to_csv(os.path.join(dir_path, f"{exp_name}_rankings.csv"), index=False)
    
    return results, rankings