from scipy.stats import wasserstein_distance
import numpy as np
import pandas as pd
import math

import ast

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
    return wasserstein_distance(support, support, p, q) #/ (support[-1] - support[0])


def get_previous_interest(user_item_matrix, u_index, topic_vectors, item_polarity):


    prev_ids = np.array(user_item_matrix.iloc[u_index])
    prev_ids = np.where(prev_ids != 0)[0]
    
    prev_topics = np.zeros(70)
    for k in range(prev_ids.shape[0]):
        prev_topics = np.add(prev_topics, np.array(ast.literal_eval(topic_vectors.iloc[prev_ids[k]]['topical_vector'])))

    row = user_item_matrix.iloc[u_index]
    
    mask = (row != 0)
    magnitudes = np.array(row[mask])

    orig = np.zeros((14,5))
    total_mag = np.zeros((14,5))

    # but prev_ids are in order
    for k in range(len(prev_ids)):
        item_id = prev_ids[k]
        
        rated_label = item_polarity.iloc[item_id]['source_partisan_score']
        
        rated_topic = np.array(ast.literal_eval(topic_vectors.iloc[item_id]['topical_vector']))
        
        topic_ind = np.where(rated_topic > 0)[0] // 5
        
        rated_topic = rated_topic.reshape((14,5))
        
        orig += math.ceil(magnitudes[k]) * rated_topic
        
        for top in topic_ind:
            total_mag[top][int(rated_label+2)] += magnitudes[k]
            
    ##TODO
    ### build interactions as so::
    # og_interaction = np.nan_to_num(np.divide(orig, total_mag))
    og_interaction = orig


    ### pre_interest not being built right relative to og_interaction
    # avg_rating = np.sum(og_interaction.flatten()) / M
    pre_interest = np.zeros(14)
    for p in range(14):
        top_int = np.sum(og_interaction[p]) 
        pre_interest[p] = top_int
    #     ### CHANGE HERE
        
    # avg_rating = np.mean(pre_interest[np.where(pre_interest > 0)[0]])
    
    pre_interest /= np.sum(pre_interest)
    # percent_higher = 0.2
    # existing_topics = list(filter(lambda i: pre_interest[i] >= avg_rating * (1 + percent_higher), range(len(pre_interest))))
    existing_topics = list(filter(lambda i: pre_interest[i] != 0, range(len(pre_interest))))
    
    return existing_topics, og_interaction

def class_topic_specific_details(typology, topic_of_focus, user_recommendations, metric, uv, topic_vecs, v_polarity, alpha=0.01):

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

    c = class_map[typology]
    
    class_ind = class_map[classes[c]]
    
    #true_position
    true_ind = sum(number_of_users[:class_ind])
    
    hist_bias = []
    user_hist_count = 0
    
    rec_bias = []
    user_rec_count = 0
    
    topic = topic_of_focus
    
    total_hist_topics = 0
    total_rec_topics = 0
    
    #iterating through the class members
    for i in range(true_ind, true_ind + number_of_users[class_ind]):
        
        ## calculate their historical topics and their reading distribution over them
        
        existing_topics, og_interaction = get_previous_interest(uv, i, topic_vecs, v_polarity)
        
        member_row = user_recommendations[i]
        total_rec_topics += np.sum(member_row)
        total_hist_topics += np.sum(np.array(og_interaction).flatten())
        
        member_row = member_row.reshape((14, 5))
        # ------------------------------------------
        
        #check recommended topics
           
        if sum(member_row[topic]) > 0:
            indices = np.where(member_row[topic] > 0)
            s = 0
            for ind in indices:
                s += (ind - 2) * member_row[topic][ind]
            # then divide by the sum - this is AB 
            
            rec_bias.append(s[0] / sum(member_row[topic])) 
            user_rec_count += 1
                
        # ALSO COUNT HOW MANY RECS IT IS PROVIDING OF THE TOPIC OF FOCUS
                
        if sum(og_interaction[topic]) > 0:
            # look at indices where we have greater than 0, multiply the count there by (index -2)
            indices = np.where(og_interaction[topic] > 0)
            s = 0
            for ind in indices:
                s += (ind - 2) * og_interaction[topic][ind]
            # then divide by the sum - this is AB 
            
            hist_bias.append(s[0] / sum(og_interaction[topic])) 
            user_hist_count += 1
            

            
    print(f'AB of topic {topic_of_focus} in user HISTORY: {np.mean(hist_bias)} with std_dev - {np.std(hist_bias)}. This topic was {user_hist_count / total_hist_topics} of the recommendations.')
    print(f'AB of topic {topic_of_focus} in user RECS: {np.mean(rec_bias)} with std_dev - {np.std(rec_bias)}. This topic was {user_rec_count / total_rec_topics} of the recommendations.')
    return
        
if __name__ == '__main__':
    def parse_numpy_style_array(s):
        # Strip brackets, split on spaces, filter out empty strings, convert to float
        return np.array([float(x) for x in s.strip('[]').split() if x])
    
    test_data = pd.read_csv("src\data\\baseline_data\\baseline_testing_data.csv",skipinitialspace=True)
    user_item_matrix = pd.read_csv('src\data\CF_test_correlation\\user_item_matrix.csv').drop(columns=['Unnamed: 0'])
    
    topic_vectors = test_data[['article_id','topical_vector']]
    item_polarity = test_data[['article_id', 'source_partisan_score']]
    
    
    metric = w_distance
    print('Metric: ', metric.__name__)
    
    # for i in range(14):
    #     topic = i
    
    # 11 good for popular on market skeptic repubs
    
    topic = 8
    typology = 'core conserv'

    name_1 = '8_neighbors_FN_Rating_CPC'

    CPC_recs = pd.read_csv(
        f'src\\data\\baseline_data\\total_eval\\results\\{name_1}.csv',
        converters={'topic_bias_matrix': parse_numpy_style_array},
        usecols=['topic_bias_matrix']
    ).to_numpy()
    
    CPC_recs = np.vstack(CPC_recs[:, 0])
    
    print(name_1)
    class_divergence = class_topic_specific_details(typology, topic,CPC_recs, metric, user_item_matrix, topic_vectors, item_polarity, alpha=0.000)

    print(class_divergence)

    name_2 = '8_NN_Rating_Rating'
    print(name_2)

    GCF_NN_recs = pd.read_csv(
        f'src\\data\\baseline_data\\total_eval\\results\\{name_2}.csv',
        converters={'topic_bias_matrix': parse_numpy_style_array},
        usecols=['topic_bias_matrix']
    ).to_numpy()
    
    GCF_NN_recs = np.vstack(GCF_NN_recs[:, 0])
    
    gcf_nn_class_divergence = class_topic_specific_details(typology, topic, GCF_NN_recs, metric, user_item_matrix, topic_vectors, item_polarity, alpha=0.000)
    # print(gcf_nn_class_divergence)