import itertools
import pandas as pd
from typing import List
from pref_models import fit_bradley_terry
from tqdm import tqdm

from src.agent import Agent

def query_for_single_template(agent, alternatives, template):
    results = []
    for a1, a2 in itertools.permutations(alternatives, 2):
        prompt = template.format(A=a1, B=a2)
        scores = agent.query(prompt, [a1, a2])
        # Store or process scores as needed
        results.append({
            "template": template,
            "item_a": a1,
            "item_b": a2,
            "score_a": scores[0],
            "score_b": scores[1]
        })
    return pd.DataFrame(results)

def query_all(agent, alternatives, task_prompt):
    scores = agent.query(task_prompt, alternatives)
    return pd.DataFrame({
        "template": task_prompt,
        "item_a": alternatives[i],
        "item_b": alternatives[j],
        "score_a": scores[i],
        "score_b": scores[j]  
    } for i, j in itertools.combinations(range(len(alternatives)), 2))

def run_experiment(
    agent: Agent,
    alternatives: List[str],
    templates: List[str],
    exp_name: str, #TODO: change to logger
    pairwise: bool = True,
):        
    pref_data = pd.DataFrame()
    pref_weights = pd.DataFrame()
    for i, template in tqdm(enumerate(templates)): #TODO: add template generator
        # TODO: log for Wandb?
        # 1. Collect new data
        if pairwise:
            current_data = query_for_single_template(agent, alternatives, template)
        else:
            current_data = query_all(agent, alternatives, template)
        current_data['winner'] = current_data.apply(lambda row: row['item_a'] if row['score_a'] > row['score_b'] else row['item_b'], axis=1)
        pref_data = pd.concat([pref_data, current_data.assign(iteration=i)], ignore_index=True)
        
        # 2. Fit new BT model
        # ranking is a dataframe with columns 'Color" and 'BT_Score'
        ranking, _ = fit_bradley_terry(pref_data, alternatives)
        pref_weights = pd.concat([pref_weights, ranking.assign(iteration=i)], ignore_index=True)
        
        pref_data.to_csv(f"experiments/{exp_name}/scores.csv", index=False)
        pref_weights.to_csv(f"experiments/{exp_name}/BT_weights.csv", index=False)
    
    # # TODO: log this
    # pref_data.to_csv(f"experiments/{exp_name}/scores.csv", index=False)
    # pref_weights.to_csv(f"experiments/{exp_name}/BT_weights.csv", index=False)
    
    print("Experiment complete.")
    print("Final Weights:")
    print(ranking)
