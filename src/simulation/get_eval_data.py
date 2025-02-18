import numpy as np
import pandas as pd
import ast
import re


def user_choice(uv, iv, ranked=True):
    """
    Given a user vector (uv) and a recommended new, 
    return the probability of user's clicking
    """

    product = simple_doct_product(uv, iv)

    epsilon = 10e-5

    if (product + epsilon) > 1.0:
        vui = 0.99
    else:
        vui = beta_distribution(product)

    # Awared preference
    ita = beta_distribution(0.98)
    pui = vui * ita
    
    rand_num = np.random.random()

    if rand_num < pui:
        return pui, True
    else:
        return pui, False

def beta_distribution(mu, sigma=10 ** -5):
    """
    Sample from beta distribution given the mean and variance. 
    """
    alpha = mu * mu * ((1 - mu) / (sigma * sigma) - 1 / mu)
    beta = alpha * (1 / mu - 1)

    return np.random.beta(alpha, beta)
    
def simple_doct_product(u, v):
    """
    u is user vector, v is item vector
    v should be normalized
    """
    v = [i / (sum(v)) for i in v]

    return np.dot(u, v)

##normalized topic coverage***********

def get_full_stats(user_item_matrix, user_metrics, out_path, users_choice, test_data, number_of_users, classes):
    ## CTR, utility, topic coverage, bias diversity, PE@K
    class_stats = np.zeros((9, 6))
    
    recs = user_metrics['chosen_items']
    topic_vectors = test_data[['article_id','topical_vector']]
    
    item_polarity = test_data[['article_id', 'source_partisan_score']]
    
    curr = 0
    
    for cl in range(len(classes)):
            
        class_users = user_metrics.iloc[curr: curr + number_of_users[i]]
        ind_mask = (class_users!=0).any(axis=1)
        ind_mask = curr + np.where(ind_mask == 1)[0]
        
        curr_class_stats = []
        
        for j in ind_mask:      
            ## CTR, utility, topic coverage, bias diversity, PE@k
            running_total = np.zeros(6)
            
            # items_chosen = list(map(int, re.findall(r'\d+', recs.iloc[j])))
            temp = np.random.randint(0, 4000, 10, int)
            items_chosen = [test_data.iloc[k]['article_id'] for k in temp]
            
            M = len(items_chosen)
            uv = np.array(users_choice.iloc[j])
            
            # CTR, UTILITY
            for item in items_chosen:
                item_topic = np.array(ast.literal_eval(topic_vectors.loc[topic_vectors['article_id'] == int(item)]['topical_vector'].values[0]))
                score, click = user_choice(uv, item_topic)
                
                running_total[1] += score
                if click:
                    running_total[0] += 1
            
           ## topic coverage, bias diversity, PE@k
            
            left_recs = 0
            right_recs = 0
        
            prev_ids = np.array(user_item_matrix.iloc[j])
            prev_ids = np.where(prev_ids != 0)[0]
            
            prev_topics = np.zeros(70)
            for k in range(prev_ids.shape[0]):
                prev_topics = np.add(prev_topics, np.array(ast.literal_eval(topic_vectors.iloc[prev_ids[k]]['topical_vector'])))

            row = user_item_matrix.iloc[j]
            
            mask = (row != 0)
            magnitudes = np.array(row[mask])

            orig = np.zeros((14,5))
            total_mag = np.zeros((14,5))
        
            ## but prev_ids are in order
            for k in range(len(prev_ids)):
                item_id = prev_ids[k]
                
                rated_label = item_polarity.iloc[item_id]['source_partisan_score']
                
                rated_topic = np.array(ast.literal_eval(topic_vectors.iloc[item_id]['topical_vector']))
                
                topic_ind = np.where(rated_topic > 0)[0] // 5
                
                rated_topic = rated_topic.reshape((14,5))
                
                orig += magnitudes[k] * rated_topic
                
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
            
            ###take uv, consider the relative topic interest
            chosen_ids = items_chosen
            
            ## the chosen_ids are not necessarily in order
            topic_hit = 0
            non_zero = 0
            for item_id in range(len(chosen_ids)):
            
                c_id = chosen_ids[item_id]
            
                ## have uv
                
                c_label = int(item_polarity.loc[item_polarity['article_id'] == int(c_id)]['source_partisan_score'].values[0])
                
                if c_label < 0:
                    left_recs += 1
                    non_zero += 1
                if c_label > 0:
                    right_recs += 1
                    non_zero += 1
                
                
                running_total[5] += c_label
                
                
                c_topics = ast.literal_eval(topic_vectors.loc[topic_vectors['article_id'] == int(c_id)]['topical_vector'].values[0])
                
                chosen_topic_indices = np.where(np.array(c_topics) > 0)[0] // 5
                
                mask = np.isin(chosen_topic_indices, existing_topics)
               
                if np.any(mask):
                    diversities = []
                    topic_hit += 1
                    pos = chosen_topic_indices[mask]
                    for p in pos:
                        row = og_interaction[p]
                        row /= sum(row)
                        lab = c_label + 2
                        diversities.append(1 - row[int(lab)])
                    running_total[3] += np.max(diversities) ## I should probably do average here
            
            # print(non_zero)
            # print(M)
            # print('------------------------')
            running_total[5] /= M
            running_total[4] += 1 - np.abs(left_recs / M - right_recs / M)  
            ## CTR, utility, topic coverage, bias diversity, PE@k
            ### percent bias diversity
            running_total[3] = np.nan_to_num(running_total[3] / topic_hit)
                    
            # running_total[2] = running_total[2] / M
            running_total[2] = topic_hit / len(chosen_ids)
            
            running_total[1] = running_total[1] / M
            
            running_total[0] = running_total[0] / M
            
            curr_class_stats.append(running_total)
    
        curr+= number_of_users[cl]

        stats = np.array(curr_class_stats)
        # stats[:, 5] /= number_of_users[cl]
        for q in range(stats.shape[1]):
            stat_mean = np.mean(stats[:, q])
            class_stats[cl][q] = stat_mean
  
        # curr += number_of_users[i]

    pd.DataFrame(class_stats).to_csv(f'src\data\\baseline_data\\total_eval\\metrics\\metrics_random.csv')
    
## CTR, utility, topic coverage, bias diversity
class_distribution = [49, 193, 133, 108, 69, 116, 116, 63, 153]
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

test_data = pd.read_csv("src\data\\baseline_data\\baseline_testing_data.csv",skipinitialspace=True)
user_item_matrix = pd.read_csv('src\data\CF_test_correlation\\user_item_matrix.csv').drop(columns=['Unnamed: 0'])

# holdouts = pd.read_csv('src\data\CF_test_correlation\\holdouts.csv').drop(columns=['Unnamed: 0'])


# user_item_matrix = holdouts.add(user_item_matrix)


users_choice = pd.read_csv("src\data\\baseline_data\\testing_user_choice_vectors.csv").drop(columns=['Unnamed: 0'])

out_paths = ['FN_Embedding_CPC', 'FN_Rating_CPC', 'Joint_Embedding_CPC',  'Joint_Rating_CPC', 'NN_Embedding_CPC', 'NN_Rating_CPC', 'NN_Rating_Rating']

for i in [0]:
    
    
    out_path = out_paths[i]

    user_data = pd.read_csv(f"src\data\\baseline_data\\total_eval\\results\{out_path}.csv").drop(columns=['Unnamed: 0'])

    get_full_stats(user_item_matrix, user_data, out_path, users_choice, test_data, class_distribution, classes)