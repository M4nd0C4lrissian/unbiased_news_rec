import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

    
class_map = {'bystanders': 0, 'core conserv': 8, 'country first conserv': 7, 'devout and diverse': 4, 'disaffected democrats': 3, 'market skeptic repub': 6, 'new era enterprisers': 5, 'oppty democrats': 2, 'solid liberas': 1}
number_of_users = [49, 193, 133, 108, 69, 116, 116, 63, 153]

class_thresholds = [sum(number_of_users[:i+1]) for i in range(len(number_of_users))]

def categorize(indices, thresholds, buckets):
    for ind in indices:
        for i in range(len(thresholds)):
            if ind < thresholds[i]:
                buckets[i] += 1
                break
    return buckets
                

df = pd.read_csv('src\\data\\baseline_data\\CF\\FN_matrix.csv').drop(columns=['Unnamed: 0'])

classes = [
"bystanders",
"solid liberas",
"oppty democrats",
"disaffected democrats",
"devout and diverse",
"new era enterprisers",
"market skeptic repub",
"country first conserv",
"core conserv",
]

class_buckets = {}

for c in range(len(classes)):
    
    c_dist = np.zeros((len(classes)))
    
    class_ind = class_map[classes[c]]
    
    #true_position
    true_ind = sum(number_of_users[:class_ind])
    
    #iterating through the class members
    for i in range(true_ind, true_ind + number_of_users[class_ind]):
        # extract each of their rows
        member_row = df.iloc[i, :].to_numpy()
        
        # and find the columns where the value is non-zero
        fn_indices = np.where(member_row > 0)[0]
        # and categorize 
        c_dist = categorize(fn_indices, class_thresholds, c_dist)
        
    # normalize to pdf
    c_dist /= sum(c_dist)
    # store
    class_buckets[classes[c]] = c_dist
    
    # Plot
    plt.figure(figsize=(8, 5))
    x_pos = np.arange(len(classes))

    plt.bar(x_pos, c_dist, alpha=0.7, tick_label=classes)
    plt.xticks(rotation='vertical')
    plt.xlabel('Classes')
    plt.ylabel('FN Density')
    plt.title('Normalized Histogram (PDF-like)')
    plt.grid(True, axis='y')

    # Optional legend for just one class being visualized
    plt.legend([f'{classes[c]}'])

    # Save and close
    plt.tight_layout()
    plt.savefig(f'src/data/fn_buckets/{classes[c]}.png')
    plt.close()

