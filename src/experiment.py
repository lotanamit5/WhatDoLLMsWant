import datetime
import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt

from metrics import calculate_kendall_distance
from model import get_winner, load_model, get_top_k_tokens, get_choice_via_scoring
from pref_models import fit_bradley_terry

def run_experiment(model, tokenizer, colors, template):
    """Runs all pairwise comparisons across all templates."""
    results = []

    # Check if model is instruct-tuned (simple heuristic or check tokenizer)
    is_instruct = False
    name_or_path = getattr(model.config, "_name_or_path", getattr(model.config, "name_or_path", ""))
    if "Instruct" in name_or_path:
        is_instruct = True

    for color_a, color_b in itertools.permutations(colors, 2):
        # We test both orders (A vs B) and (B vs A) to mitigate positional bias
        # Ideally, we average the logits or just record both trials.
        # Here we will treat them as separate matches.

        # Trial 1: A vs B
        prompt = template.format(A=color_a, B=color_b)

        if is_instruct and tokenizer.chat_template:
             messages = [{"role": "user", "content": prompt}]
             prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        winner, prob_a, prob_b = get_winner(model, tokenizer, prompt, color_a, color_b)
        results.append({
            "item_a": color_a,
            "item_b": color_b,
            "prob_item_a": prob_a,
            "prob_item_b": prob_b,
            "winner": winner,
            "template": template
        })

    return pd.DataFrame(results)


def run_perplexity_on_template(model, tokenizer, colors, template):
    """Runs all pairwise comparisons across all templates (Perplexity Version)."""
    results = []

    # Check if model is instruct-tuned (simple heuristic or check tokenizer)
    is_instruct = False
    name_or_path = getattr(model.config, "_name_or_path", getattr(model.config, "name_or_path", ""))
    if "Instruct" in name_or_path:
        is_instruct = True

    for color_a, color_b in itertools.permutations(colors, 2):
        # We test both orders (A vs B) and (B vs A) to mitigate positional bias

        # Trial 1: A vs B
        prompt = template.format(A=color_a, B=color_b)

        if is_instruct and tokenizer.chat_template:
             messages = [{"role": "user", "content": prompt}]
             prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Log prefills
        # get_choice_via_scoring adds a space before the option
        print(f"DEBUG: Prefill A: '{prompt} {color_a}'")
        print(f"DEBUG: Prefill B: '{prompt} {color_b}'")

        winner, prob_a, prob_b = get_choice_via_scoring(model, tokenizer, prompt, color_a, color_b)
        results.append({
            "item_a": color_a,
            "item_b": color_b,
            "prob_item_a": prob_a,
            "prob_item_b": prob_b,
            "winner": winner,
            "template": template
        })

    return pd.DataFrame(results)


def run_perplexity_experiment(model_name, colors, templates, exp_name=None):
    if exp_name is None:
        exp_name = "perplexity_experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Loading model {model_name} for perplexity experiment...")
    model, tokenizer = load_model(model_name)
    results = pd.DataFrame()
    rankings = []
    
    for i, template in enumerate(templates):
        print(f"Running template {i+1}/{len(templates)}...")
        df_results = run_perplexity_on_template(model, tokenizer, colors, template)
        results = pd.concat([results, df_results], ignore_index=True)
        
        ranking, _ = fit_bradley_terry(results, colors)
        print(f"Current rankings: {list(ranking['Color'])}, calculated on {len(results)} comparisons")
        
        rankings.append(ranking.assign(iteration=i))
        
    # Save results
    rankings = pd.concat(rankings)
    dir_path = os.path.join("experiments", exp_name)
    os.makedirs(dir_path, exist_ok=True)
    results.to_csv(os.path.join(dir_path, f"answers.csv"), index=False)
    rankings.to_csv(os.path.join(dir_path, f"rankings.csv"), index=False)
    
    return results, rankings

def run_full_experiment(model_name, colors, templates, exp_name=None):
    if exp_name is None:
        exp_name = "experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    model, tokenizer = load_model(model_name)
    results = pd.DataFrame()
    rankings = []
    for i, template in enumerate(templates):
        df_results = run_experiment(model, tokenizer, colors, template)
        results = pd.concat([results, df_results], ignore_index=True)
        ranking, _ = fit_bradley_terry(results, colors)
        print(f"Current rankings: {list(ranking['Color'])}, calculated on {len(results)} comparisons")
        if len(rankings) > 0:
            kendall_tau = calculate_kendall_distance(rankings[-1]['Color'], ranking['Color'])
            print(f"Current Kendall Tau Distance: {kendall_tau}")
        rankings.append(ranking.assign(iteration=i))
        
    # Save results
    rankings = pd.concat(rankings)
    dir_path = os.path.join("experiments", exp_name)
    os.makedirs(dir_path, exist_ok=True)
    results.to_csv(os.path.join(dir_path, f"answers.csv"), index=False)
    rankings.to_csv(os.path.join(dir_path, f"rankings.csv"), index=False)
    
    return results, rankings

def measure_top5_occurrences(model_name, colors, templates=None, exp_name=None):
    """
    Measures the occurrences of tokens in the top-5 logits.
    Aggregates tokens by groups: "A", "B", " {color}", "{color}", "Option", "option".
    """
    if templates is None:
        # Avoid circular import or path issues by importing here only if needed
        try:
            from prompts import templates_en
            templates = templates_en
        except ImportError:
            # Fallback for when running from src directly or different setup
            from src.prompts import templates_en
            templates = templates_en

    print(f"Loading model {model_name}...")
    model, tokenizer = load_model(model_name)
    
    print(f"Starting top-5 analysis with {len(templates)} templates and {len(colors)} colors...")

    # counts_freq: Total occurrences at each rank (can sum > N if multiple tokens map to same category)
    # Categories align with user request: {A/B} (no space), _{A/B} (space), _{color}, _{Color}, {color}, {Color}, option, Other
    categories = ["{A/B}", "_{A/B}", "_{color}", "_{Color}", "{color}", "{Color}", "option", "Other"]
    counts_freq = {k: [0]*5 for k in categories}
    
    # counts_presence: Counts of first appearance (binary) to calculate P(Category in Top-k)
    counts_presence = {k: [0]*5 for k in categories}
    
    print(f"Starting top-5 analysis with {len(templates)} templates and {len(colors)} colors...")

    is_instruct = False
    name_or_path = getattr(model.config, "_name_or_path", getattr(model.config, "name_or_path", ""))
    if "Instruct" in name_or_path:
        is_instruct = True

    total_checks = 0
    others = {}
    
    for i, template in enumerate(templates):
        if i % 10 == 0:
            print(f"Processing template {i}/{len(templates)}...")
            
        for color_a, color_b in itertools.permutations(colors, 2):
            prompt = template.format(A=color_a, B=color_b)
            
            if is_instruct and tokenizer.chat_template:
                 messages = [{"role": "user", "content": prompt}]
                 prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            top_tokens = get_top_k_tokens(model, tokenizer, prompt, k=5)
            
            current_colors = {color_a, color_b} 
            current_colors_cap = {c.capitalize() for c in current_colors}
            
            seen_categories = set()
            
            for rank, token in enumerate(top_tokens):
                # Determine category
                category = "Other"
                if token in ["A", "B"]:
                    category = "{A/B}"
                elif token in [" A", " B"]:
                    category = "_{A/B}"
                elif token in ["Option", " Option", "option", " option"]:
                    category = "option"
                elif any(token == f" {c}" for c in current_colors):
                    category = "_{color}"
                elif any(token == f" {c}" for c in current_colors_cap):
                    category = "_{Color}"
                elif any(token == c for c in current_colors):
                    category = "{color}"
                elif any(token == c for c in current_colors_cap):
                    category = "{Color}"
                
                # Freq count: Always increment
                counts_freq[category][rank] += 1

                # Presence count: Increment only if not seen yet in this prompt
                if category not in seen_categories:
                    counts_presence[category][rank] += 1
                    seen_categories.add(category)
                
                if category == "Other":
                     # For 'others' individual tokens, we can just count raw frequency
                     others[token] = others.get(token, 0) + 1
                     
            total_checks += 1
            
    print("Top-5 Analysis Complete.")
    print(f"Total prompt evaluations: {total_checks}")
    print(f"Total tokens checked: {total_checks * 5}")
    print("Counts (Frequency):", counts_freq)
    
    if exp_name:
         dir_path = os.path.join("experiments", exp_name)
         os.makedirs(dir_path, exist_ok=True)
         
         # Save counts
         pd.DataFrame(counts_freq).to_csv(os.path.join(dir_path, "top5_counts_freq.csv"))
         pd.DataFrame(counts_presence).to_csv(os.path.join(dir_path, "top5_counts_presence.csv"))
         
         # For compatibility with notebook, save presence as main counts (fixes "110%" issue for Prop analysis)
         pd.DataFrame(counts_presence).to_csv(os.path.join(dir_path, "top5_counts.csv"))
         
         # Save others
         df_others = pd.Series(others, name="count").sort_values(ascending=False)
         df_others.to_csv(os.path.join(dir_path, "top5_others.csv"))
         
         # Plot - Top 20 "Other" tokens
         if not df_others.empty:
             plt.figure(figsize=(12, 6))
             df_others.head(20).plot(kind='bar', color='salmon')
             plt.title(f"Top 20 'Other' tokens (Model: {model_name})")
             plt.xlabel("Token")
             plt.ylabel("Count")
             plt.xticks(rotation=45, ha='right')
             plt.tight_layout()
             plt.savefig(os.path.join(dir_path, "top5_others_histogram.png"))
             plt.close()

         # Plot - Stacked Bar Chart (Using Freq)
         counts = counts_freq 
         plt.figure(figsize=(12, 7))
         
         categories = list(counts.keys())
         # counts[cat] is a list of 5 ints
         
         bottoms = [0] * len(categories)
         ranks = range(5)
         colors_stack = plt.cm.viridis(pd.Series(ranks) / 4.0) # 5 colors

         for rank in ranks:
             values = [counts[c][rank] for c in categories]
             plt.bar(categories, values, bottom=bottoms, label=f"Rank {rank+1}", color=colors_stack[rank])
             bottoms = [b + v for b, v in zip(bottoms, values)]

         plt.title(f"Top-5 Token Distribution by Rank (Model: {model_name})")
         plt.xlabel("Token Group")
         plt.ylabel("Count")
         plt.legend()
         plt.xticks(rotation=45)
         plt.tight_layout()
         plt.savefig(os.path.join(dir_path, "top5_histogram.png"))
         plt.close()

    return counts
