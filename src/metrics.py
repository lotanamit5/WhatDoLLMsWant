from scipy.stats import kendalltau

def calculate_kendall_distance(ranking_a, ranking_b):
    """
    Calculates the Normalized Kendall Tau Distance between two lists of items.
    Returns a value between 0.0 (identical) and 1.0 (completely reversed).
    """
    # Ensure both lists have the same items
    assert set(ranking_a) == set(ranking_b), "Rankings must contain the same items"

    # Calculate Tau correlation (-1 to 1)
    tau, _ = kendalltau(ranking_a, ranking_b)

    # Convert correlation to normalized distance
    # Distance = (1 - tau) / 2
    # If tau = 1 (identical), distance = 0
    # If tau = -1 (reversed), distance = 1
    return (1 - tau) / 2