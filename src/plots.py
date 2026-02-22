
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_utils import better_color_map


def plot_convergence(df_weights,
                     title="BT Convergence",
                     color_map=None,
                     IQR=False,
                     start_iteration=1):
    """
    :param df_weights: index is 'iteration', columns are alternatives, values are BT probabilities
    """
    df_weights = df_weights.copy()[df_weights.index >= start_iteration]
    
    plt.figure(figsize=(15, 5))
    columns = [c for c in df_weights.columns if c not in ['iteration', 'num_templates']]
    if color_map:
        assert set(columns).issubset(set(color_map.keys())) \
            , f"Data columns do not match color map keys.\nColumns:{columns}\nColor Map Keys: {color_map.keys()}"
    for alternative in columns:
        if color_map:
            color = color_map[alternative]
            plt.plot(df_weights.index, df_weights[alternative], label=alternative, 
                color=better_color_map.get(color, color), linewidth=2)
        else:
            plt.plot(df_weights.index, df_weights[alternative], label=alternative, linewidth=2)
    plt.legend()
    plt.xlabel("Iteration (Templates)")
    plt.ylabel("BT Probability ($e^\\beta$)")
    if IQR:
        all_values = df_weights.values.flatten()
        q1 = np.percentile(all_values, 25)
        q3 = np.percentile(all_values, 75)
        iqr = q3 - q1
        plt.ylim(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_final_weights(df_weights_list,
                       title="BT Probabilities",
                       color_map=None,
                    ):
   all_final_probs = []

   for df in df_weights_list:
      last_iter = df['iteration'].max()
      final_df = df[df['iteration'] == last_iter].copy()
      final_df['prob'] = np.exp(final_df['BT_Score'])
      all_final_probs.append(final_df[['Item', 'prob']])

   summary_df = (
      pd.concat(all_final_probs)
      .groupby('Item')['prob'].agg(['mean', 'std'])
      .reset_index()
      .sort_values('mean', ascending=False)
   )

   plt.figure(figsize=(14, 6))
   items = summary_df['Item']
   means = summary_df['mean']
   stds = summary_df['std'].fillna(0)
   
   x_pos = np.arange(len(items))
   # Handle colors
   colors = 'skyblue'
   if color_map:
      assert set(items).issubset(set(color_map.keys())), \
         f"All items must have a corresponding color in the color_map.\nItems: {items}\nColor Map: {color_map.keys()}"
      colors = [better_color_map[color_map.get(item, 'grey')] for item in items]
      plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10, color=colors)
   else:
      plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)
   
   plt.xticks(x_pos, items, rotation=45, ha='right')
   plt.ylabel('BT Probability ($e^{\\beta}$)')
   plt.title(title)
   plt.tight_layout()
   plt.grid(axis='y', linestyle='--', alpha=0.7)
   plt.show()