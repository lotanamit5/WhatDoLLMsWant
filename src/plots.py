from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


def plot_convergence(history_df, colors, color_map=None):
    # Convert BT weights to probabilities
    # Extract the scores for the colors
    scores = history_df[colors].values
    # Softmax per row (iteration)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    plt.figure(figsize=(12, 7))

    # Map for actual line colors if possible
    color_map = color_map or {
            'red': 'red', 'blue': 'blue', 'green': 'green',
            'yellow': '#FFD700', 'purple': 'purple',
            'pink': '#FFC0CB', 'black': 'black', 'white': 'lightgrey'
        }

    for i, color in enumerate(colors):
        plt.plot(
            history_df['iteration'],
            probs[:, i],
            label=color.capitalize(),
            color=color_map.get(color, 'black'),
            linewidth=2,
            alpha=0.8
        )

    plt.title("Convergence of LLM Color Preferences (Probabilities)")
    plt.xlabel("Number of Prompt Templates (Cumulative)")
    plt.ylabel("Probability (Softmax of BT Score)")
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_weight_comparison(df_rankings):
    if df_rankings.empty:
        print("No rankings data to plot.")
        return

    # Pivot to have methods as columns and items as rows/index
    df_pivot = df_rankings.pivot(index='Color', columns='method', values='BT_Score')
    
    # Convert scores to probabilities (Softmax over the items for each method)
    df_probs = df_pivot.apply(lambda x: np.exp(x) / np.sum(np.exp(x)), axis=0)

    # Plot grouped bar chart
    ax = df_probs.plot(kind='bar', figsize=(10, 6), width=0.8)
    
    plt.title("Comparison of Probabilities by Method")
    plt.xlabel("Item (Color)")
    plt.ylabel("Probability (Softmax of BT Score)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='Method')
    plt.tight_layout()
    plt.show()

def plot_convergence_results(df_results):
    unique_methods = df_results['method'].unique()
    unique_items = df_results['item'].unique()

    # Define a color map covering potential colors
    color_map = {
        'red': 'red', 'blue': 'blue', 'green': 'green',
        'yellow': '#FFD700', 'purple': 'purple',
        'pink': '#FFC0CB', 'black': 'black', 'white': 'lightgrey',
        'orange': 'orange', 'brown': 'brown', 'cyan': 'cyan',
        'gray': 'gray', 'magenta': 'magenta'
    }

    for method in unique_methods:
        # Filter data for the current method
        df_method = df_results[df_results['method'] == method]
        
        # Pivot: index=iteration, columns=item, values=score
        history_df = df_method.pivot(index='iteration', columns='item', values='score').reset_index()
        
        # Identify item columns
        item_cols = [c for c in history_df.columns if c != 'iteration']
        
        # Convert to probabilities
        scores = history_df[item_cols].values
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Update history_df with probabilities
        history_df[item_cols] = probs

        plt.figure(figsize=(10, 6))
        
        # Plot each item's trajectory
        for item in unique_items:
            if item in history_df.columns:
                plt.plot(
                    history_df['iteration'],
                    history_df[item],
                    label=item.capitalize(),
                    color=color_map.get(item, None), # Fallback to default cycle if not in map
                    linewidth=2,
                    alpha=0.8
                )

        plt.title(f"Convergence of Probabilities (Method: {method})")
        plt.xlabel("Number of Templates Used")
        plt.ylabel("Probability (Softmax of BT Score)")
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    plt.show()

def plot_probability_comparison(df, item_col='item', score_col='score', group_col='model', title="Comparison of Probabilities"):
    """
    Plots a grouped bar chart of probabilities derived from BT scores.
    """
    if df.empty:
        print("No data to plot.")
        return

    # Pivot: index=item, columns=group, values=score
    df_pivot = df.pivot(index=item_col, columns=group_col, values=score_col)
    
    # Convert to probabilities (Softmax column-wise)
    # Using apply with axis=0 operates on each column (each model/method)
    df_probs = df_pivot.apply(lambda x: np.exp(x) / np.sum(np.exp(x)), axis=0)
    
    # Reset index for melting or plot directly with pandas
    # Pandas plotting is convenient for bar charts
    ax = df_probs.plot(kind='bar', figsize=(12, 6), width=0.8)
    
    plt.title(title)
    plt.xlabel("Item")
    plt.ylabel("Probability")
    plt.ylim(0, 1) # Probabilities are 0-1
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title=group_col, loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmaps(df, item_col='item', score_col='score', group_col='model'):
    """
    Plots Pearson and Kendall correlation heatmaps for the probabilities.
    """
    
    # Pivot
    df_pivot = df.pivot(index=item_col, columns=group_col, values=score_col)
    
    # Convert to probs
    df_probs = df_pivot.apply(lambda x: np.exp(x) / np.sum(np.exp(x)), axis=0)
    
    # Pearson
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_probs.corr(method='pearson'), annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title("Correlation of Probabilities (Pearson)")
    plt.tight_layout()
    plt.show()
    
    # Kendall (Identical for probs or scores, effectively)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_probs.corr(method='kendall'), annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
    plt.title("Correlation of Rankings (Kendall)")
    plt.tight_layout()
    plt.show()

def plot_scatter_comparison(df, item_col='item', score_col='score', group_col='model', title="Weight Correlation"):
    """
    Plots a scatter plot of probabilities for two models.
    Assumes df contains exactly 2 groups in group_col.
    """
    unique_groups = df[group_col].unique()
    if len(unique_groups) != 2:
        print(f"Scatter plot requires exactly 2 groups, found {len(unique_groups)}: {unique_groups}")
        return

    # Pivot
    df_pivot = df.pivot(index=item_col, columns=group_col, values=score_col)
    
    # Convert to probs
    df_probs = df_pivot.apply(lambda x: np.exp(x) / np.sum(np.exp(x)), axis=0)
    
    col_x, col_y = unique_groups[0], unique_groups[1]
    
    # Calculate correlations on probabilities
    corr_pearson = df_probs.corr(method='pearson').loc[col_x, col_y]
    corr_kendall = df_probs.corr(method='kendall').loc[col_x, col_y]
    
    print(f"Probability Correlation (Pearson): {corr_pearson:.4f}")
    print(f"Rank Correlation (Kendall): {corr_kendall:.4f}")

    plt.figure(figsize=(7, 7))
    plt.scatter(df_probs[col_x], df_probs[col_y], alpha=0.7)
    
    # Annotate points
    for txt in df_probs.index:
         plt.annotate(txt, (df_probs.loc[txt, col_x], df_probs.loc[txt, col_y]), 
                      textcoords="offset points", xytext=(5,5), fontsize=9)
    
    plt.title(f"{title}\n(Probabilities)")
    plt.xlabel(f"{col_x} (Prob)")
    plt.ylabel(f"{col_y} (Prob)")
    
    # Diagonal line (0 to 1)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal') # Keep aspect ratio square to compare x=y
    # Ensure limits are 0-1 or slightly padded
    plt.xlim(0, max(df_probs.max().max() * 1.1, 0.1))
    plt.ylim(0, max(df_probs.max().max() * 1.1, 0.1))
    plt.tight_layout()
    plt.show()
