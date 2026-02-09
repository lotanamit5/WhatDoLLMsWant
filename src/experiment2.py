import itertools
import pandas as pd
from typing import List
from pref_models import fit_bradley_terry
from tqdm import tqdm

from agent import Agent

def query_for_single_template(agent, alternatives, template):
    results = []
    for a1, a2 in itertools.permutations(alternatives, 2):
        prompt = template.format(A=a1, B=a2)
        scores = agent.query(prompt, [a1, a2])
        # Store or process scores as needed
        results.append({
            "template": template,
            "a1": a1,
            "a2": a2,
            "s1": scores[0],
            "s2": scores[1]
        })
    return pd.DataFrame(results)

def run_experiment(
    agent: Agent,
    alternatives: List[str],
    templates: List[str],
    exp_name: str #TODO: change to logger
):
    pref_data = pd.DataFrame()
    pref_weights = pd.DataFrame()
    for i, template in tqdm(enumerate(templates)): #TODO: add template generator
        # TODO: log for Wandb?
        current_data = query_for_single_template(agent, alternatives, template)
        pref_data = pd.concat([pref_data, current_data.assign(iteration=i)], ignore_index=True)
        # ranking is a dataframe with columns 'Color" and 'BT_Score'
        ranking, _ = fit_bradley_terry(pref_data, alternatives)
        pref_weights = pd.concat([pref_weights, ranking.assign(iteration=i)], ignore_index=True)
    
    # TODO: log this
    pref_data.to_csv(f"experiments/{exp_name}/scores.csv", index=False)
    pref_weights.to_csv(f"experiments/{exp_name}/BT_weights.csv", index=False)
    
    print("Experiment complete.")
    print("Final Weights:")
    print(ranking)
