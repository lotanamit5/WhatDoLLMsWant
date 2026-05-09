import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import expit


def fit_bradley_terry(df, items):
    """
    Fits the Bradley-Terry model using Maximum Likelihood Estimation.
    P(i beats j) = exp(beta_i) / (exp(beta_i) + exp(beta_j))
    """
    n = len(items)
    item_to_idx = {item: i for i, item in enumerate(items)}

    # Initialize betas (scores) to 0
    initial_betas = np.zeros(n)

    # Prepare data for optimization
    # We aggregate wins: wins[i][j] = number of times i beat j
    wins = np.zeros((n, n))
    for _, row in df.iterrows():
        idx_w = item_to_idx[row['winner']]
        # Identifying the loser
        loser = row['item_b'] if row['winner'] == row['item_a'] else row['item_a']
        idx_l = item_to_idx[loser]
        wins[idx_w, idx_l] += 1

    # Negative Log-Likelihood Function
    def neg_log_likelihood(betas):
        nll = 0
        for i in range(n):
            for j in range(n):
                if wins[i, j] > 0:
                    # Log Prob of i beating j
                    prob_i_beats_j = np.exp(betas[i]) / (np.exp(betas[i]) + np.exp(betas[j]))
                    nll -= wins[i, j] * np.log(prob_i_beats_j + 1e-9) # small epsilon
                    nll += (wins[i, j] + wins[j, i]) * np.log(np.exp(betas[i]) + np.exp(betas[j]) + 1e-9)
        return nll

    # Constraint: Sum of betas = 0 (or beta_0 = 0) to fix scale.
    # Here we typically center them or fix one. Let's just minimize.
    res = minimize(neg_log_likelihood, initial_betas, method='BFGS')

    # Normalize scores so they sum to 0 for easier interpretation
    final_betas = res.x - np.mean(res.x)

    ranking = pd.DataFrame({
        'Item': items,
        'BT_Score': final_betas
    })

    return ranking, wins

def fit_BT_soft(df, items):
    """
    Fits the Bradley-Terry model using Cross-Entropy on soft probabilities.
    P(i beats j) = exp(beta_i) / (exp(beta_i) + exp(beta_j))
    """    
    n = len(items)
    item_to_idx = {item: i for i, item in enumerate(items)}

    # Initialize betas (scores) to 0
    initial_betas = np.zeros(n)

    # Prepare data for optimization
    # We aggregate soft wins using the LLM's continuous perplexity scores
    soft_wins = np.zeros((n, n))
    for _, row in df.iterrows():
        idx_a = item_to_idx[row['item_a']]
        idx_b = item_to_idx[row['item_b']]
        
        # Soft probability that A beats B based on LLM scores
        p_a_beats_b = expit(row['score_a'] - row['score_b'])
        
        soft_wins[idx_a, idx_b] += p_a_beats_b
        soft_wins[idx_b, idx_a] += (1.0 - p_a_beats_b)

    # Negative Log-Likelihood (Cross-Entropy) Function
    def neg_log_likelihood(betas):
        nll = 0
        for i in range(n):
            for j in range(n):
                if soft_wins[i, j] > 0:
                    # Log Prob of i beating j: beta_i - log(exp(beta_i) + exp(beta_j))
                    log_p_ij = betas[i] - np.logaddexp(betas[i], betas[j])
                    nll -= soft_wins[i, j] * log_p_ij
        return nll

    # Constraint: Sum of betas = 0 (or beta_0 = 0) to fix scale.
    # Here we typically center them or fix one. Let's just minimize.
    res = minimize(neg_log_likelihood, initial_betas, method='BFGS')

    # Normalize scores so they sum to 0 for easier interpretation
    final_betas = res.x - np.mean(res.x)

    ranking = pd.DataFrame({
        'Item': items,
        'BT_Score': final_betas
    })

    return ranking, soft_wins
