from matplotlib import pyplot as plt


def plot_convergence(history_df, colors, color_map=None):
    plt.figure(figsize=(12, 7))

    # Map for actual line colors if possible
    color_map = color_map or {
            'red': 'red', 'blue': 'blue', 'green': 'green',
            'yellow': '#FFD700', 'purple': 'purple',
            'pink': '#FFC0CB', 'black': 'black', 'white': 'grey'
        }

    for color in colors:
        plt.plot(
            history_df['iteration'],
            history_df[color],
            label=color.capitalize(),
            color=color_map.get(color, 'black'),
            linewidth=2,
            alpha=0.8
        )

    plt.title("Convergence of LLM Color Preferences (Bradley-Terry Weights)")
    plt.xlabel("Number of Prompt Templates (Cumulative)")
    plt.ylabel("Preference Strength (Log-Odds)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()