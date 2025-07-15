import numpy as np
import pandas as pd
import math

def discrete_variance(probs):
    support = np.arange(-2, 3)
    return np.average((support - np.average(support, weights=probs))**2, weights=probs)

    
class_map = {'bystanders': 0, 'core conserv': 8, 'country first conserv': 7, 'devout and diverse': 4, 'disaffected democrats': 3, 'market skeptic repub': 6, 'new era enterprisers': 5, 'oppty democrats': 2, 'solid liberas': 1}
number_of_users = [49, 193, 133, 108, 69, 116, 116, 63, 153]

class_thresholds = [sum(number_of_users[:i+1]) for i in range(len(number_of_users))]

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
        
class_entropy = {}
class_variances = {}
        
for c in range(len(classes)):
    
    class_ind = class_map[classes[c]]
    
    #true_position
    true_ind = sum(number_of_users[:class_ind])
    
    class_means = []
    variances = []
    
    #iterating through the class members
    for i in range(true_ind, true_ind + number_of_users[class_ind]):
        # extract each of their rows
        member_row = users_choice[i]
        member_row = member_row.reshape((14, 5))
        
        # topic_entropies = []
        # for topic in range(member_row.shape[0]):
        #     #calculate per-topic entropy over bias - we want std_dev over topics too
        #     t = member_row[topic]
        #     # normalize to pdf
        #     t /= sum(t)
            
        #     norm_entrop = sum([- t_i * math.log2(t_i) for t_i in t]) / math.log2(len(t))
        #     topic_entropies.append(norm_entrop)
        
        total_bias_preferences = [np.sum(member_row[:, q]) for q in range(member_row.shape[1])]
        total_bias_preferences /= sum(total_bias_preferences)
        
        norm_entrop = sum([- t_i * math.log2(t_i) for t_i in total_bias_preferences]) / math.log2(len(total_bias_preferences))
        class_means.append(norm_entrop)
        variances.append(discrete_variance(total_bias_preferences))
        # class_std_dev_sum += std
    # so now we take the average of the average and std_dev entropy over topics, across a class - this feels weird 
    class_entropy[classes[c]] = {'Average over topics' : np.mean(class_means), 'Average Deviation' : np.std(class_means)}
    class_variances[classes[c]] = {'Average Variance' : np.mean(variances)}
print(class_entropy)
            

        