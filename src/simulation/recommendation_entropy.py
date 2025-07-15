import numpy as np
import pandas as pd
import math
import ast

def discrete_variance(probs):
    support = np.arange(-2, 3)
    return np.average((support - np.average(support, weights=probs))**2, weights=probs)

def recommendation_entropy(user_recommendations):

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
    
    class_entropy = {}
    class_variances = {}
    
    #keep it simple - get the matrices, sum them up element-wise
    # reshape them, sum up the columns and calculate the entropy
    
    for c in range(len(classes)):
    
        class_ind = class_map[classes[c]]
        
        #true_position
        true_ind = sum(number_of_users[:class_ind])
        
        class_means = []
        variances = []
        
        #iterating through the class members
        for i in range(true_ind, true_ind + number_of_users[class_ind]):
            # extract each of their rows
            member_row = user_recommendations[i]
            member_row = member_row.reshape((14, 5))
            
            total_bias_preferences = [np.sum(member_row[:, q]) for q in range(member_row.shape[1])]
            total_bias_preferences /= sum(total_bias_preferences)
        
            norm_entrop = sum([0 if t_i == 0 else - t_i * math.log2(t_i) for t_i in total_bias_preferences]) / math.log2(len(total_bias_preferences))
            class_means.append(norm_entrop)
            variances.append(discrete_variance(total_bias_preferences))
        
        class_entropy[classes[c]] = {'Average over all topics' : np.mean(class_means), 'std' : np.std(class_means)} 
        class_variances[classes[c]] = {'Average Variance' : np.mean(variances)}
    
    return class_entropy, class_variances

if __name__ == '__main__':
    def parse_numpy_style_array(s):
        # Strip brackets, split on spaces, filter out empty strings, convert to float
        return np.array([float(x) for x in s.strip('[]').split() if x])

    CPC_recs = pd.read_csv(
        'src\\data\\baseline_data\\total_eval\\results\\8_NN_Rating_Rating.csv',
        converters={'topic_bias_matrix': parse_numpy_style_array},
        usecols=['topic_bias_matrix']
    ).to_numpy()
    
    CPC_recs = np.vstack(CPC_recs[:, 0])
    
    class_entropy, class_var = recommendation_entropy(CPC_recs)
    pass

    GCF_NN_recs = pd.read_csv(
        'src\\data\\baseline_data\\total_eval\\results\\8_neighbors_FN_Rating_CPC.csv',
        converters={'topic_bias_matrix': parse_numpy_style_array},
        usecols=['topic_bias_matrix']
    ).to_numpy()
    
    GCF_NN_recs = np.vstack(GCF_NN_recs[:, 0])
    
    gcf_nn_class_entropy, gcf_var = recommendation_entropy(GCF_NN_recs)
    pass
    