

#### KL divergence - sum (p_i * lg (p_i / q_i)), where P is the true interest, and Q is that induced by our recommendations
## what to do when q_i is zero - we will apply Laplace smoothing with alpha = 0.01

# Also try total variation distance - could be more informative
# we may actually want Jensen-Shannon Divergence\
# what might be better is to not do topic specific, so that either model is
# not penalized for diversifying over topics

from scipy.stats import wasserstein_distance
import numpy as np
import pandas as pd
import math

def laplace_smoothing(dist, alpha):
    dist += alpha
    return dist / sum(dist)

def kl_divergence(p, q):
    sum = 0
    for i in range(len(p)):
        sum += p[i] * math.log2(max(p[i], 1e-10) / max(q[i], 1e-10))
    return sum

def total_variation(p, q):
    sum = 0
    for i in range(len(p)):
        sum += abs(p[i] - q[i])
    return sum * 0.5

def js_divergence(p, q):
    m = (p + q) / 2
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def w_distance(p, q):
    support = np.arange(-2, 3)
    return wasserstein_distance(support, support, p, q) / (support[-1] - support[0])

def distribution_stat(user_recommendations, metric, alpha=0.01):

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

    t = pd.read_pickle('testing_1000users.pkl')
    users_choice = np.empty((1000, 70))

    q = 0
    for i in range(len(classes)):
        cl = classes[i]
        for j in range(number_of_users[i]):
            users_choice[q] = np.array(t[cl][j]).flatten()
            q+=1
        
    class_divergence = {}

    for c in range(len(classes)):
        
        class_ind = class_map[classes[c]]
        
        #true_position
        true_ind = sum(number_of_users[:class_ind])
        
        class_means = []
        
        #iterating through the class members
        for i in range(true_ind, true_ind + number_of_users[class_ind]):
            uv_row = users_choice[i]
            uv_row = uv_row.reshape((14, 5))
            
            member_row = user_recommendations[i]
            member_row = member_row.reshape((14, 5))
            # ------------------------------------------
            p = [np.sum(uv_row[:, j]) for j in range(uv_row.shape[1])]
            q = [np.sum(member_row[:, j]) for j in range(member_row.shape[1])]
            
            p /= sum(p)
            q /= sum(q)
            
            class_means.append(metric(p, q))
            # for topic in range(uv_row.shape[1]):
            #     if sum(member_row[topic]) == 0:
            #         continue
                
            #     q = laplace_smoothing(member_row[topic], alpha = alpha)
            #     p = uv_row[topic]
            #     p /= sum(p)
                
            #     class_means.append(metric(p, q))
            
            # retry over all topics
            
                
        class_divergence[classes[c]] = {'Average over topics' : np.mean(class_means), 'Average Deviation' : np.std(class_means)}
    
    return class_divergence
        
if __name__ == '__main__':
    def parse_numpy_style_array(s):
        # Strip brackets, split on spaces, filter out empty strings, convert to float
        return np.array([float(x) for x in s.strip('[]').split() if x])
    
    metric = kl_divergence
    print('Metric: ', metric.__name__)

    name_1 = '8_neighbors_FN_Rating_CPC'

    CPC_recs = pd.read_csv(
        f'src\\data\\baseline_data\\total_eval\\results\\{name_1}.csv',
        converters={'topic_bias_matrix': parse_numpy_style_array},
        usecols=['topic_bias_matrix']
    ).to_numpy()
    
    CPC_recs = np.vstack(CPC_recs[:, 0])
    
    class_divergence = distribution_stat(CPC_recs, metric, alpha=0.000)
    print(name_1)
    print(class_divergence)

    name_2 = '8_NN_Rating_Rating'
    print(name_2)

    GCF_NN_recs = pd.read_csv(
        f'src\\data\\baseline_data\\total_eval\\results\\{name_2}.csv',
        converters={'topic_bias_matrix': parse_numpy_style_array},
        usecols=['topic_bias_matrix']
    ).to_numpy()
    
    GCF_NN_recs = np.vstack(GCF_NN_recs[:, 0])
    
    gcf_nn_class_divergence = distribution_stat(GCF_NN_recs, metric, alpha=0.000)
    print(gcf_nn_class_divergence)
    