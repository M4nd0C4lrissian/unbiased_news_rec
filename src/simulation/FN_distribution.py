import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from KL_divergence import kl_divergence

    
class_map = {'bystanders': 0, 'core conserv': 8, 'country first conserv': 7, 'devout and diverse': 4, 'disaffected democrats': 3, 'market skeptic repub': 6, 'new era enterprisers': 5, 'oppty democrats': 2, 'solid liberas': 1}
number_of_users = [49, 193, 133, 108, 69, 116, 116, 63, 153]

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

class_thresholds = [sum(number_of_users[:i+1]) for i in range(len(number_of_users))]

def categorize(indices, thresholds, buckets):
    new_dist = np.zeros_like(buckets)
    for ind in indices:
        for i in range(len(thresholds)):
            if ind < thresholds[i]:
                new_dist[i] += 1
                break
    return new_dist
                

df = pd.read_csv('src\\data\\baseline_data\\CF\\8_FN_matrix.csv').drop(columns=['Unnamed: 0'])

class_buckets = {}
class_divergences = {}

base_dist = np.array(number_of_users) / sum(number_of_users)

for c in range(len(classes)):
    
    class_divergences[classes[c]] = []
    
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
        new_dist = categorize(fn_indices, class_thresholds, c_dist)
        
        class_divergences[classes[c]].append(kl_divergence(new_dist / sum(new_dist), base_dist))
        
        c_dist += new_dist
        
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
    plt.savefig(f'src/data/fn_buckets/rating_corr_{classes[c]}.png')
    plt.close()

for c in range(len(classes)):
    print(classes[c], ':')
    df = pd.DataFrame(class_divergences[classes[c]])
    print(df.describe())
    print('-------------------------------------')