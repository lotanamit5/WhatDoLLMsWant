import numpy as np
import pandas as pd

import statsmodels.api as sm

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


# ============================================================================
# OLS Bradley-Terry with positional bias (laptops / feature experiments)
# ============================================================================

def fit_feature_based_bradley_terry(df: pd.DataFrame, 
                                    a_prefix='a_', b_prefix='b_',
                                    score_a_col='score_a', score_b_col='score_b') -> dict:
    """
    Fits a Feature-Based Bradley-Terry model with Positional Bias.
    Automatically detects categorical features and handles dummy encoding.
    """
    n_samples = len(df)
    
    # 1. Target Variable Y (Log-Odds)
    Y = (df[score_a_col] - df[score_b_col]).values
    
    # 2. Automatically detect feature columns
    cols_a = [col for col in df.columns if col.startswith(a_prefix)]
    cols_b = [col.replace(a_prefix, b_prefix) for col in cols_a]
    
    # Extract the base feature names (e.g., 'a_brand' -> 'brand')
    base_features = [col[len(a_prefix):] for col in cols_a]
    
    # 3. Standardize and align the feature sets
    # We rename the columns so both A and B share the exact same column names
    df_a = df[cols_a].rename(columns=dict(zip(cols_a, base_features)))
    df_b = df[cols_b].rename(columns=dict(zip(cols_b, base_features)))
    
    # Combine A and B vertically. This is crucial! 
    # It ensures that pd.get_dummies() sees all possible categories across the entire 
    # dataset and drops the EXACT SAME reference level for both Option A and Option B.
    df_combined = pd.concat([df_a, df_b], axis=0)
    
    # Create dummy variables (dropping the first level to avoid multicollinearity)
    # We cast to float to ensure statsmodels handles it cleanly
    df_combined_dummies = pd.get_dummies(df_combined, drop_first=True, dtype=float)
    
    # Split them back apart into A and B matrices
    dummies_a = df_combined_dummies.iloc[:n_samples]
    dummies_b = df_combined_dummies.iloc[n_samples:]
    
    # 4. Create the Design Matrix X (Features A - Features B)
    X_base = dummies_a.values - dummies_b.values
    feature_names = dummies_a.columns.tolist()
    
    # Add the Intercept for gamma (Positional Bias)
    X = sm.add_constant(X_base)
    
    # 5. Fit the OLS Model
    model = sm.OLS(Y, X)
    results = model.fit()
    
    # 6. Extract Bias
    gamma_bias = results.params[0]
    gamma_pvalue = results.pvalues[0]
    
    # 7. Extract Feature Utilities (Weights)
    # Coefficients start at index 1 (index 0 is the intercept)
    feature_weights = results.params[1:]
    feature_pvalues = results.pvalues[1:]
    
    # Create dictionaries mapping the specific feature category to its learned utility
    feature_scores_dict = {feat: weight for feat, weight in zip(feature_names, feature_weights)}
    feature_pvalues_dict = {feat: pval for feat, pval in zip(feature_names, feature_pvalues)}
    
    # 8. Compile the final dictionary
    output_dict = {
        "feature_scores": feature_scores_dict,
        "feature_pvalues": feature_pvalues_dict,
        "positional_bias_gamma": gamma_bias,
        "gamma_pvalue": gamma_pvalue,
        "model_r_squared": results.rsquared, 
        "f_stat_pvalue": results.f_pvalue    
    }
    print(f"Feature p-values:")
    for feat, pval in feature_pvalues_dict.items():
        print(f"  {feat}: p-value = {pval:.4f}")
    print(f"Positional bias (gamma): {gamma_bias:.4f}, p-value = {gamma_pvalue:.4f}")
    print(f"Model R-squared: {results.rsquared:.4f}, F-statistic p-value: {results.f_pvalue:.4f}")
    
    return output_dict


def fit_item_bradley_terry(df, a_prefix="a_", b_prefix="b_"):
    """
    Fits the feature-based Bradley-Terry model (one additive weight per feature
    level: brand, screen, ram, ...) and then computes each laptop's utility in
    post-processing as the sum of its own feature weights. This assumes the
    features contribute additively (no brand x screen interaction, etc.) - a
    more constrained model than fitting a separate coefficient per laptop.
    """
    feature_scores = fit_feature_based_bradley_terry(df, a_prefix=a_prefix, b_prefix=b_prefix)["feature_scores"]

    cols_a = [col for col in df.columns if col.startswith(a_prefix)]
    cols_b = [col.replace(a_prefix, b_prefix) for col in cols_a]
    base_features = [col[len(a_prefix):] for col in cols_a]

    # Every unique laptop is a unique combination of feature values seen as
    # either option A or option B anywhere in the data
    df_a = df[cols_a].rename(columns=dict(zip(cols_a, base_features)))
    df_b = df[cols_b].rename(columns=dict(zip(cols_b, base_features)))
    unique_laptops = pd.concat([df_a, df_b], axis=0).drop_duplicates()

    item_scores = {}
    for _, row in unique_laptops.iterrows():
        item_label = "-".join(str(row[feat]) for feat in base_features)
        item_scores[item_label] = sum(
            feature_scores.get(f"{feat}_{row[feat]}", 0.0) for feat in base_features
        )

    return item_scores
