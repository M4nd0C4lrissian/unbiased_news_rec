import pandas as pd
import numpy as np
# Visualize sparsity trends and zero-user counts
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR

from collections import defaultdict



def altered_normalized_bottom_k_with_bias(Bi, k, selection_count, index_set, alpha=0.0):
    """
    Probabilistically adjust selection to promote fuller coverage of users.
    
    Args:
    - Bi: Correlation matrix.
    - k: Number of users to select per user.
    - alpha: Bias adjustment factor (0.0 = no bias, 1.0 = full bias to less-selected users).
    
    Returns:
    - norm_B: Normalized matrix with retained bottom-k values.
    """
    # global selection_count, index_set  # Track selection frequency globally
    
    norm_B = np.zeros_like(Bi, dtype=np.float64)
    
    Bi = -Bi

    for u in range(Bi.shape[0]):
        row = Bi[u]

        # Filter out NaNs and negative values
        valid_mask = ~np.isnan(row)
        filtered_row = row[valid_mask]

        if len(filtered_row) == 0:
            norm_B[u] = np.zeros_like(row)
            print(f'user: {u} has no viable users')
            continue

        # Adjust scores to include selection bias
        original_indices = np.where(valid_mask)[0]
        adjusted_scores = filtered_row.copy()

        for idx, orig_idx in enumerate(original_indices):
            # Adjust scores based on selection count
            adjusted_scores[idx] += alpha * (1 / (1 + selection_count[orig_idx]))

        # Get indices of the bottom-k adjusted values
        retain_ind = np.argsort(-adjusted_scores)[:k]
        retain_val = filtered_row[retain_ind]

        # Update the global selection count
        for ind in original_indices[retain_ind]:
            selection_count[ind] += 1
            index_set.add(ind)

        # Normalize retained values
        s = np.sum(retain_val)
        if s == 0:
            norm_B[u] = np.zeros_like(row)
            print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
            continue

        # Create a new row with only the selected values retained
        b = np.zeros_like(row)
        b[original_indices[retain_ind]] = retain_val
        b = b / s  # Normalize to sum to 1

        norm_B[u] = b

    return norm_B

if __name__ == '__main__':
    user_correlation_matrix = pd.read_csv("src\\data\\baseline_data\\CF\\correlation_matrix.csv").drop(columns=['Unnamed: 0']).to_numpy()
    selection_count = defaultdict(int)
    index_set = set()
    
    #FN - embedding - CPC - 1 
    k = 30
    Bi = altered_normalized_bottom_k_with_bias(user_correlation_matrix, k, selection_count, index_set, alpha=0.1)
    
    pd.DataFrame(Bi).to_csv('src\\data\\baseline_data\\CF\\FN_matrix.csv')
    
    ## user 9 is everyone's furthest neighbor?
    ## First look at the user histogram, then the class to class histogram
    