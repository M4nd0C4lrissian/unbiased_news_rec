import torch
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

import per_user_gcf as GC
import baseline_per_user_gcf as BGCF

from collections import defaultdict
import copy

import warnings
from redo_FN import furthest_neighbours_construct_convolutions
from redo_rating_FN import normalized_top_k_with_bias

## load user choice models - what is the user choice model?
## load correlation matrix
    ## create normalized top k correlation - can do over the same data - one of these will be trained for every user
## load h weights
## load the item-topic matrix, load the item-polarity matrix
## load user_item matrix
## construct rating matrix

## for every user, look at which ratings it already had, 
## perform the convolutions
## take items with the top k scores, and gauge the response / diversity of polarity and topics (also calculate loss for the time being)
## do the same for k random items, and compare 

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

def user_interaction_score(uv, iv, ranked=True):
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

    return pui

def log(user_metrics):
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

    ##class_map = {'bystanders': 0, 'core conserv': 8, 'country first conserv': 7, 'devout and diverse': 4, 'disaffected democrats': 3, 'market skeptic repub': 6, 'new era enterprisers': 5, 'oppty democrats': 2, 'solid liberas': 1}

    number_of_users = [49, 193, 133, 108, 69, 116, 116, 63, 153]

    # user_metrics = pd.read_csv('src\\data\\baseline_data\\total_eval\\all_user_metrics.csv').drop(columns=['Unnamed: 0'])

    curr = 0
    for i in range(len(classes)):
        class_users = user_metrics.iloc[curr: curr + number_of_users[i]]
        class_users = class_users.loc[(class_users!=0).any(axis=1)]
        print(f'Class {classes[i]}----------------- ')
        print('Topic coverage: ')
        print(class_users['topic_hit'].describe())
        print()
        print('Diversity over hits: ')
        print(class_users['diversity'].describe())
        print()
        curr += number_of_users[i]

##TODO
## want to - save logs of the recommendations generated for each user (indices, and flattened 14 x 5)
## add logging / evaluation stuff 

def evaluate(out_path):
    M = 10
    f = 5
    k = 8
    
    item_topic =  pd.read_csv('src\data\\baseline_data\\baseline_testing_data.csv', skipinitialspace=True, usecols=['article_id', 'topical_vector', 'source_partisan_score'])
    item_polarity = pd.read_csv('src\data\\baseline_data\\baseline_testing_data.csv', skipinitialspace=True, usecols=['article_id', 'source_partisan_score'])

    user_item_matrix = pd.read_csv("src\\data\\CF_test_correlation\\user_item_matrix.csv").drop(columns=['Unnamed: 0'])
    holdouts = pd.read_csv("src\\data\\CF_test_correlation\\holdouts.csv").drop(columns=['Unnamed: 0'])
    
    # user_item_matrix = pd.DataFrame(np.add(user_item_matrix.to_numpy(), holdouts.to_numpy()))

    item_list = pd.read_csv('src\data\\baseline_data\\baseline_testing_data.csv', skipinitialspace=True, usecols=['article_id']).to_numpy()

    chosen_per_class_score = np.zeros((9, 5))
    
    chosen_partisan_score = [0, 0, 0, 0, 0]

    chosen_utility_across_classes = [0,0,0,0,0,0,0,0,0]

    # classes = ['bystanders', 'core conserv', 'country first conserv', 'devout and diverse', 'disaffected democrats', 'market skeptic repub', 'new era enterprisers', 'oppty democrats', 'solid liberas']
    
    #  class_map = {'bystanders': 0, 'core conserv': 1, 'country first conserv': 2, 'devout and diverse': 3, 'disaffected democrats': 4, 'market skeptic repub': 5, 'new era enterprisers': 6, 'oppty democrats': 7, 'solid liberas': 8}
    #    number_of_users = [49, 153, 63, 69, 108, 116, 116, 133, 193]
    
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
    
    class_map = {'bystanders': 0, 'core conserv': 8, 'country first conserv': 7, 'devout and diverse': 4, 'disaffected democrats': 3, 'market skeptic repub': 6, 'new era enterprisers': 5, 'oppty democrats': 2, 'solid liberas': 1}
    
    number_of_users = [49, 193, 133, 108, 69, 116, 116, 63, 153]
   

    t = pd.read_pickle('testing_1000users.pkl')


    users_choice = np.empty((1000, 70))
    user_metrics = []
    
    ##make a 9 by 5 by 14 array 
    
    recommendation_stats = np.zeros((9, 14, 5))
    
    original_interaction_stats = np.zeros((9, 14, 5))
    
    user_classes=[]
    
    q = 0
    for i in range(len(classes)):
        cl = classes[i]
        for j in range(number_of_users[i]):
            users_choice[q] = np.array(t[cl][j]).flatten()
            user_classes.append({"class" : cl})
            q+=1
    
    ## users_choice is the original interest model (not needed necessarily)
    user_classes = pd.DataFrame.from_records(user_classes)
    
    with torch.no_grad():
        
                
        selection_count = defaultdict(int)
        index_set = set() 

        for u in range(user_item_matrix.shape[0]):
            
            print('User: ', u)
            
            row = user_item_matrix.iloc[u]
            
            #####HERE
            hold_row = holdouts.iloc[u]
            valid_mask = ((np.array(row) == 0.0) & (np.array(hold_row) == 0.0))
            # valid_mask = (row == 0)
            filtered_row = row[valid_mask]
            available_indices = np.where(valid_mask)[0]
            
            c = user_classes.iloc[u].values[0]
    
            randomly_chosen_items = np.random.choice(available_indices, M, replace=False)
            
            ## original 
            
            mask = (row != 0)
            magnitudes = np.array(row[mask])
            rated_indices = np.where(mask)[0]
            
            orig = np.zeros((14,5))
            total_mag = np.zeros((14,5))
            
            for k in range(len(rated_indices)):
                item_id = item_list[rated_indices[k]]
                
                rated_label = item_polarity.loc[item_polarity['article_id'] == int(item_id)]['source_partisan_score'].values[0]
                
                rated_topic = np.array(ast.literal_eval(item_topic.loc[item_topic['article_id'] == int(item_id)]['topical_vector'].values[0]))
                
                topic_ind = np.where(rated_topic > 0)[0] // 5
                
                rated_topic = rated_topic.reshape((14,5))
                
                orig += magnitudes[k] * rated_topic
                
                for top in topic_ind:
                    total_mag[top][int(rated_label+2)] += 1
            ##TODO
            ### build interactions as so:
            interaction = np.nan_to_num(np.divide(orig, total_mag))
            
            original_interaction_stats[class_map[c]] = interaction
            
            existing_topics = []
            
            for index in range(14):
                if np.sum(original_interaction_stats[class_map[c]][index]) > 0:
                    existing_topics.append(index)
    
            existing_topics = np.array(existing_topics)
            
            ############################################
            
            percent_topic_hit = 0
            diversity_over_hit_topics = 0
            
            randomly_chosen_ids = item_list[randomly_chosen_items]
            chosen_topics = np.zeros(70)
                
            for item_id in range(len(randomly_chosen_ids)):
            
                c_id = randomly_chosen_ids[item_id]
            
                u_choice = users_choice[u]
                
                c_label = int(item_polarity.loc[item_polarity['article_id'] == int(c_id)]['source_partisan_score'].values[0])
                
                c_topics = ast.literal_eval(item_topic.loc[item_topic['article_id'] == int(c_id)]['topical_vector'].values[0])
                
                chosen_topics = np.add(chosen_topics, c_topics)
                
                chosen_topic_indices = np.where(np.array(c_topics) > 0)[0] // 5
                
                mask = np.isin(chosen_topic_indices, existing_topics)

                if np.any(mask):
                    diversities = []
                    percent_topic_hit += 1
                    pos = chosen_topic_indices[mask]
                    for i in pos:
                        row = interaction[i]
                        row /= sum(row)
                        lab = c_label + 2
                        diversities.append(1 - row[int(lab)])
                    diversity_over_hit_topics += np.max(diversities) ## I should probably do average here
                
                
                c_utility = user_interaction_score(u_choice, c_topics)
                
                idx = class_map[c]
                
                chosen_partisan_score[int(c_label+2)] += 1
                
                chosen_per_class_score[idx][c_label+2] += 1
                
                c_stats = np.array(c_topics).reshape((14, 5))
                
                recommendation_stats[idx] += c_stats
                
                chosen_utility_across_classes[idx] += c_utility
                
            ##TODO
            ##NEEDS TO CHANGE FOR ALL USERS - should add the number of recommendations of different types - add a 14 x 5 array
            if percent_topic_hit == 0:
                user_metrics.append({'topic_hit': 0, 'diversity': None, 'chosen_items': randomly_chosen_ids, 'topic_bias_matrix': chosen_topics})
            else:
                user_metrics.append({'topic_hit': percent_topic_hit / len(randomly_chosen_ids), 'diversity': diversity_over_hit_topics / percent_topic_hit, 'chosen_items': randomly_chosen_ids, 'topic_bias_matrix': chosen_topics})
        
        
        pd.DataFrame(user_metrics).to_csv(out_path)
            
        model_performance = np.divide(chosen_utility_across_classes, np.multiply(M, number_of_users))
        
        
        
        # for i in range(recommendation_stats.shape[0]):
            
        #     pd.DataFrame(recommendation_stats[i], columns=['-2', '-1', '0', '1', '2'], index=['abortion', 'environment', 'guns', 'health care', 'immigration', 'LGBTQ', 'racism', 'taxes',
        #       'technology', 'trade', 'trump impeachment', 'us military', 'us 2020 election', 'welfare']).to_csv(f'src\\data\\baseline_data\\recommended\\{classes[i]}.csv')
        # #     # pd.DataFrame(oracle_stats[i], columns=['-2', '-1', '0', '1', '2'], index=['abortion', 'environment', 'guns', 'health care', 'immigration', 'LGBTQ', 'racism', 'taxes',
        # #     #   'technology', 'trade', 'trump impeachment', 'us military', 'us 2020 election', 'welfare']).to_csv(f'src\\data\\results2\\oracle\\{classes[i]}.csv')
            
        #     pd.DataFrame(chosen_per_class_score[i]).to_csv(f'src\\data\\baseline_data\\recommended\\partisan_dist_{classes[i]}.csv')
        # #     # pd.DataFrame(oracle_per_class_score[i]).to_csv(f'src\\data\\results2\\oracle\\partisan_dist_{classes[i]}.csv')
        # #     pass
        
        
        # for i in range(len(classes)):
        #     cl = classes[i]

        #     print(f'{cl}: ')
            
        #     fig, (ax1, ax2) = plt.subplots(1, 2)

        #     arr = recommendation_stats[i]
        #     arr2 = original_interaction_stats[i]

        #     # total = np.sum(arr.flatten())
        #     # print(total)
        #     # arr /= total
            
        #     total = np.sum(arr2.flatten())
        #     arr2 /= total
            
            
        #     fig, axes = plt.subplots(1, 3, figsize=(12, 8), constrained_layout=True)  # Horizontally stacked

        #     # Plot the first heatmap
        #     im1 = axes[0].imshow(arr, cmap='Blues', interpolation='none')
        #     axes[0].set_title(f"Topic Cov: {user_metrics[i]['topic_hit']}, Div: {user_metrics[i]['diversity']}")  # Title for the first subplot
        #     axes[0].set_xticks(np.arange(5))
        #     axes[0].set_xticklabels([-2, -1, 0, 1, 2])
        #     axes[0].set_yticks(np.arange(len(chosen_topic)))
        #     axes[0].set_yticklabels(chosen_topic)

        #     # Plot the second heatmap
        #     im2 = axes[1].imshow(arr2, cmap='Blues', interpolation='none')
        #     axes[1].set_title("User Interest relative to Ratings")  # Title for the second subplot
        #     axes[1].set_xticks(np.arange(5))
        #     axes[1].set_xticklabels([-2, -1, 0, 1, 2])
        #     axes[1].set_yticks(np.arange(len(chosen_topic)))
        #     axes[1].set_yticklabels(['','','','','','','','','','','','','',''])


        #     im3 = axes[2].imshow(total_topic_dist, cmap='Blues', interpolation='none')
        #     axes[2].set_title("Topic Distribution in Item Set")  # Title for the second subplot
        #     axes[1].set_xticks(np.arange(0))
        #     axes[1].set_xticklabels([])
        #     axes[2].set_yticks(np.arange(len(chosen_topic)))
        #     axes[2].set_yticklabels(['','','','','','','','','','','','','',''])


        #     # Add colorbars for both plots
        #     fig.colorbar(im1, ax=axes[0], orientation='vertical', shrink=0.8)
        #     fig.colorbar(im2, ax=axes[1], orientation='vertical', shrink=0.8)
        #     fig.colorbar(im3, ax=axes[2], orientation='vertical', shrink=0.8)

        #     # Save the figure
        #     plt.savefig(f'src\\data\\baseline_data\\graphs\\{cl}.png')

            

        print(f'Random performance across classes: {model_performance}, with bias distribution: {chosen_partisan_score}')
        
        # df = pd.read_csv('src\\data\\landmark_data\\validation_topics_in_embedding_order.csv').iloc[:1000]
        # total = np.zeros(70)

        # for i in range(df.shape[0]):
        #     row = df.iloc[i]
        #     topics = np.array(ast.literal_eval(row['topical_vector']))
            
        #     total += topics
            
        # per_topic = total.reshape((14,5))

        # more_per_topic = np.zeros(14)
        # for j in range(per_topic.shape[0]):
        #     more_per_topic[j] = np.sum(per_topic[j])
            
        # pd.DataFrame(more_per_topic).to_csv('src\\data\\results2\\total_topic_dist.csv')
        # print(list)
        
        return
    
def simple_choice_calc():
    
    classes = ['bystanders', 'core conserv', 'country first conserv', 'devout and diverse', 'disaffected democrats', 'market skeptic repub', 'new era enterprisers', 'oppty democrats', 'solid liberas']
    class_map = {'bystanders': 0, 'core conserv': 1, 'country first conserv': 2, 'devout and diverse': 3, 'disaffected democrats': 4, 'market skeptic repub': 5, 'new era enterprisers': 6, 'oppty democrats': 7, 'solid liberas': 8}
    
    t = pd.read_pickle('1000users.pkl')
    
    for c in range(len(classes)):
        ## 14 x 5
        avg_member = pd.read_csv('src\\data\synthetic_user\\' + classes[c] +'.csv', header=None)
        
        simple_choice = np.zeros((14, 2))
        for i in range(avg_member.shape[0]):
            row = np.array(avg_member.iloc[i])
            
            max_ind = np.argsort(-row)[0]
            max_utility = row[max_ind]
            
            simple_choice[i] = [max_utility, max_ind - 2]
            
        pd.DataFrame(simple_choice).to_csv(f'src\\data\\synthetic_user\\simple_choice_{classes[c]}.csv')
            
        
def oracle_eval():
    pass           
        

##last attempt - sample one user randomly from every class, use user-item matrix to compute interest, compare with the recommended jazzl
        
    
##not really testing recommendation diversity at the individual level - should try this
if __name__ == '__main__':
    
    with warnings.catch_warnings(action="ignore"):
        
        paths = 'random.csv'
        
        print(paths, "---------------------------------------------------")
    
        out_path = f'src\\data\\baseline_data\\total_eval\\results\\{paths}'
        
        evaluate(out_path)
        
        user_metrics = pd.read_csv(out_path).drop(columns=['Unnamed: 0'])
    
        log(user_metrics)
        
    # correlation_matrix = pd.read_csv("src\\data\\baseline_data\\CF\\rating_correlation_matrix.csv").drop(columns=['Unnamed: 0']).to_numpy()
    # all_weights = pd.read_csv(f'src\\data\\baseline_data\\CF\\per_user\\NN_rating_target_rating_corr_h_5_per_user.csv').drop(columns=['Unnamed: 0'])
    # evaluate('src\\data\\baseline_data\\total_eval\\results\\NN_Rating_Rating.csv', correlation_matrix, all_weights, topk=True)
    
    
    # ##have to change some stuff
    # user_metrics = pd.read_csv('src\\data\\baseline_data\\total_eval\\results\\NN_Rating_Rating.csv').drop(columns=['Unnamed: 0'])
    # log(user_metrics)