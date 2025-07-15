import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from KL_divergence import kl_divergence


class_map = {
    'bystanders': 0,
    'solid liberas': 1,
    'oppty democrats': 2,
    'disaffected democrats': 3,
    'devout and diverse': 4,
    'new era enterprisers': 5,
    'market skeptic repub': 6,
    'country first conserv': 7,
    'core conserv': 8
}

number_of_users = [49, 193, 133, 108, 69, 116, 116, 63, 153]
class_thresholds = [sum(number_of_users[:i+1]) for i in range(len(number_of_users))]
classes = list(class_map.keys())

def categorize(indices, thresholds, buckets):
    new_dist = np.zeros_like(buckets)
    for ind in indices:
        for i in range(len(thresholds)):
            if ind < thresholds[i]:
                new_dist[i] += 1
                break
    return new_dist

def get_class_distribution(df, class_name):
    c_dist = np.zeros(len(classes))
    class_ind = class_map[class_name]
    start_idx = sum(number_of_users[:class_ind])
    end_idx = start_idx + number_of_users[class_ind]

    for i in range(start_idx, end_idx):
        member_row = df.iloc[i, :].to_numpy()
        fn_indices = np.where(member_row > 0)[0]
        new_dist = categorize(fn_indices, class_thresholds, c_dist)
        c_dist += new_dist

    c_dist /= c_dist.sum()  # normalize
    return c_dist

# --- Load your two dataframes here ---
df1 = pd.read_csv('src/data/baseline_data/CF/8_FN_matrix.csv').drop(columns=['Unnamed: 0'])
df2 = pd.read_csv('src/data/baseline_data/CF/CPC_8_FN_matrix.csv').drop(columns=['Unnamed: 0'])

# --- Choose the class to visualize ---
target_class = 'core conserv'

# --- Get distributions ---
dist1 = get_class_distribution(df1, target_class)
dist2 = get_class_distribution(df2, target_class)

# --- Plot side-by-side horizontal bars ---
bar_height = 0.3  # thinner bars to reduce vertical space
reversed_classes = classes[::-1]
y_pos = np.arange(len(reversed_classes))
dist1 = dist1[::-1]
dist2 = dist2[::-1]

plt.figure(figsize=(8, 6))

# Updated colors: more contrast and less pastel
plt.barh(y_pos - bar_height/2, dist1, height=bar_height, color="#E9826D", label='Rating Correlation') 
plt.barh(y_pos + bar_height/2, dist2, height=bar_height, color="#84d1f2", label='CPC-Corr')   

# Font size tweaks
plt.yticks(y_pos, reversed_classes, fontsize=12)
# plt.xlabel('FN Density', fontsize=14)
plt.title(f'FN histogram for {target_class}', fontsize=16, loc='left')

# Remove grid and borders
plt.grid(False)
plt.box(False)

# Legend styling
plt.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig(f'src/data/fn_buckets/comparison.png')
plt.close()