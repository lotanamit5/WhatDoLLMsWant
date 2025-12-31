import numpy as np
import pandas as pd
from scipy.optimize import minimize


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
        loser = row['player_2'] if row['winner'] == row['player_1'] else row['player_1']
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
        return nll

    # Constraint: Sum of betas = 0 (or beta_0 = 0) to fix scale.
    # Here we typically center them or fix one. Let's just minimize.
    res = minimize(neg_log_likelihood, initial_betas, method='BFGS')

    # Normalize scores so they sum to 0 for easier interpretation
    final_betas = res.x - np.mean(res.x)

    ranking = pd.DataFrame({
        'Color': items,
        'BT_Score': final_betas
    }).sort_values('BT_Score', ascending=False).reset_index(drop=True)

    return ranking, wins
