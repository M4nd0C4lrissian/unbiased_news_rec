

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

import ast
import matplotlib.pyplot as plt


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
        
        orig += (magnitudes[k]) * rated_topic
        
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

def plot_heatmap(recommendation_stats, original_interaction_stats, cl, user_id):
    
    chosen_topic = [
        "abortion",
        "environment",
        "guns",
        "health care",
        "immigration",
        "LGBTQ",
        "racism",
        "taxes",
        "technology",
        "trade",
        "trump",
        "us military",
        "election",
        "welfare",
    ]

    
    arr = recommendation_stats
    arr2 = original_interaction_stats
    
    # total = np.sum(arr2.flatten())
    # arr2 /= total
    # chosen_topics = chosen_topics.reshape((14, 5))

    fig, axes = plt.subplots(1, 2, figsize=(12, 12), constrained_layout=True)

    im1 = axes[0].imshow(arr, cmap='Blues', interpolation='none')
    axes[0].set_title(f"FNPC Recommendations", fontsize=18)
    axes[0].set_xticks(np.arange(5))
    axes[0].set_xticklabels([-2, -1, 0, 1, 2],  fontsize=18)
    axes[0].set_yticks(np.arange(len(chosen_topic)))
    axes[0].set_yticklabels(chosen_topic, fontsize=18)

    im2 = axes[1].imshow(arr2, cmap='Blues', interpolation='none')
    axes[1].set_title("User Interest relative to Ratings",  fontsize=18)
    axes[1].set_xticks(np.arange(5))
    axes[1].set_xticklabels([-2, -1, 0, 1, 2],  fontsize=18)
    axes[1].set_yticks(np.arange(len(chosen_topic)))
    axes[1].set_yticklabels([''] * len(chosen_topic))

    fig.colorbar(im1, ax=axes[0], orientation='vertical', shrink=0.8)
    fig.colorbar(im2, ax=axes[1], orientation='vertical', shrink=0.8)

    plt.savefig(f'src/data/baseline_data/graphs/{cl}.png')
    plt.close(fig)

def generate(user_recommendations, uv, topic_vecs, v_polarity):

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
    
    # for c in range(len(classes)):
        
    #     class_ind = class_map[classes[c]]
        
    #     #true_position
    #     true_ind = sum(number_of_users[:class_ind])
        
        #iterating through the class members
        # for i in range(true_ind, true_ind + number_of_users[class_ind]):
        
    # i = np.random.choice(range(true_ind, true_ind + number_of_users[class_ind]))
    i = 510
    ## calculate their historical topics and their reading distribution over them
    
    existing_topics, og_interaction = get_previous_interest(uv, i, topic_vecs, v_polarity)
    
    member_row = user_recommendations[i]
    member_row = member_row.reshape((14, 5))
    # ------------------------------------------
    
    plot_heatmap(member_row, og_interaction, 'devout and diverse', i)


    return
        
if __name__ == '__main__':
    def parse_numpy_style_array(s):
        # Strip brackets, split on spaces, filter out empty strings, convert to float
        return np.array([float(x) for x in s.strip('[]').split() if x])
    
    test_data = pd.read_csv("src\data\\baseline_data\\baseline_testing_data.csv",skipinitialspace=True)
    user_item_matrix = pd.read_csv('src\data\CF_test_correlation\\user_item_matrix.csv').drop(columns=['Unnamed: 0'])
    
    topic_vectors = test_data[['article_id','topical_vector']]
    item_polarity = test_data[['article_id', 'source_partisan_score']]

    name_1 = '8_neighbors_FN_Rating_CPC'

    CPC_recs = pd.read_csv(
        f'src\\data\\baseline_data\\total_eval\\results\\{name_1}.csv',
        converters={'topic_bias_matrix': parse_numpy_style_array},
        usecols=['topic_bias_matrix']
    ).to_numpy()
    
    CPC_recs = np.vstack(CPC_recs[:, 0])
    
    class_divergence = generate(CPC_recs, user_item_matrix, topic_vectors, item_polarity)
    print(name_1)
    print(class_divergence)
    