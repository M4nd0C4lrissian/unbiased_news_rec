import torch
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

import per_user_gcf as GC


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

def create_predicted_rating_matrix(f, k, h, correlation_matrix):
 
    np.fill_diagonal(correlation_matrix, 0)
    B = correlation_matrix

    ## was 100 in training - I did not realize 

    Bi = GC.normalized_bottom_k_with_bias(B, k)
    B_i = GC.construct_convolutions_with_user_check(Bi, f)

    # s = torch.nn.Sigmoid()
    # h = s(h)

    item_topic = pd.read_csv('src\\data\\landmark_data\\validation_topics_in_embedding_order.csv')
    item_polarity = pd.read_csv('src\\data\\auto_encoder_training\\validation_data\\validation_partisan_labels.csv')

    user_item_matrix = pd.read_csv('src\\data\\CF\\user_item_matrix.csv').drop(columns=['Unnamed: 0'])


    with torch.no_grad():
        item_list = user_item_matrix.columns
        rating_matrix = torch.empty(1000, 0, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')

        for item_id in range(user_item_matrix.shape[1]):

            # Get rating vector for the item

            x_i = user_item_matrix.iloc[:][str(item_list[item_id])].values
            #print(x_i)
            x_i = torch.tensor(x_i, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
            #print(x_i)


            x_hat = GC.weighted_graph_convolution(x_i, B_i, h)
            # if x_hat.abs().sum().item() == 0:
            #     print('UH oh!')
            #     print(item_id)
            
            if torch.isnan(x_hat).any():
                print("Found NaN in x_hat")
            if torch.isnan(x_i).any():
                print("Found NaN in x_i")
            
            rating_matrix = torch.cat((rating_matrix, x_hat.unsqueeze(0).T), dim = -1)
            
        print(f"rating_matrix shape: {rating_matrix.shape}")

    return rating_matrix


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

def evaluate():
    M = 10
    f = 5
    k = 30
    
    total_topic_dist = np.array(pd.read_csv('src\\data\\results2\\total_topic_dist.csv').drop(columns=['Unnamed: 0']))
    
    item_topic = pd.read_csv('src\\data\\landmark_data\\validation_topics_in_embedding_order.csv')
    item_polarity = pd.read_csv('src\\data\\auto_encoder_training\\validation_data\\validation_partisan_labels.csv')

    user_classes = pd.read_csv('src\\data\\user_space\\user_space_matrix_with_topics.csv').iloc[:]['class']

    user_item_matrix = pd.read_csv('src\\data\\CF\\user_item_matrix.csv').drop(columns=['Unnamed: 0'])

    item_list = user_item_matrix.columns

    oracle_per_class_score = np.zeros((9, 5))
    chosen_per_class_score = np.zeros((9, 5))
    
    oracle_partisan_score = [0,0,0,0, 0]
    chosen_partisan_score = [0, 0, 0, 0, 0]

    oracle_utility_across_classes = [0,0,0,0,0,0,0,0,0]
    chosen_utility_across_classes = [0,0,0,0,0,0,0,0,0]

    classes = ['bystanders', 'core conserv', 'country first conserv', 'devout and diverse', 'disaffected democrats', 'market skeptic repub', 'new era enterprisers', 'oppty democrats', 'solid liberas']
    class_map = {'bystanders': 0, 'core conserv': 1, 'country first conserv': 2, 'devout and diverse': 3, 'disaffected democrats': 4, 'market skeptic repub': 5, 'new era enterprisers': 6, 'oppty democrats': 7, 'solid liberas': 8}
    
    
    number_of_users = [49, 153, 63, 69, 108, 116, 116, 133, 193]

    t = pd.read_pickle('1000users.pkl')


    users_choice = np.empty((1000, 70))
    
    
    ##make a 9 by 5 by 14 array 
    
    recommendation_stats = np.zeros((9, 14, 5))
    oracle_stats = np.zeros((9, 14, 5))
    
    original_interaction_stats = np.zeros((9, 14, 5))
    
    ## topic diversity?
    
    ##using item_topic[]
    
    correlation_matrix = pd.read_csv('src\\data\\user_space\\correlation_matrix.csv').drop(columns=['Unnamed: 0']).to_numpy()
    
    all_weights = pd.read_csv('src\\data\\CF\\per_user\\trained_h_5_per_user.csv').drop(columns=['Unnamed: 0'])
    
    class_oracle_vectors = np.zeros((9,14,2))
    
    q = 0
    for i in range(len(classes)):
        cl = classes[i]
        for j in range(number_of_users[i]):
            users_choice[q] = np.array(t[cl][j]).flatten()
            q+=1
        
        class_oracle_vectors[i] = pd.read_csv(f'src\\data\\synthetic_user\\simple_choice_{cl}.csv').drop(columns=['Unnamed: 0']).to_numpy()
        
    
    list = []
    prev = 0
    for i in range(len(classes)):
        u = np.random.randint(prev, np.sum(prev + number_of_users[i]))
        list.append(u)
        prev += number_of_users[i]
    
    # 
    # ## per user: (topic coverage, diversity correct topics)
    # user_metrics = np.empty((len(list), 2))
    user_metrics = []

    for u in list:
        
        row = user_item_matrix.iloc[u]
        valid_mask = (row == 0)
        filtered_row = row[valid_mask]
        available_indices = np.where(valid_mask)[0]
        
        c = user_classes.iloc[u]
        
        h = torch.tensor(all_weights.iloc[u], dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
        
        predicted_rating_matrix = create_predicted_rating_matrix(f,k, h, correlation_matrix)
        
        predicted_row = predicted_rating_matrix[u].cpu().numpy()
        filtered_pred = predicted_row[available_indices]

        ## top M
        M = min(len(filtered_pred), M)
        if M == 0:
            print('Uh oh! No neighbors!')
            continue
        
        
        retain_ind = np.argsort(-filtered_pred)[:M]
        retain_val = filtered_pred[retain_ind]
        
        ## M random
        
        
        chosen_items = available_indices[retain_ind]
        
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
        "trump impeachment",
        "us military",
        "us 2020 election",
        "welfare",
        ]
        
        
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
                total_mag[top][rated_label+2] += 1
            
        interaction = np.nan_to_num(np.divide(orig, total_mag))
        
        original_interaction_stats[class_map[c]] = interaction
        
        existing_topics = []
        
        for index in range(14):
            if np.sum(original_interaction_stats[class_map[c]][index]) > 0:
                existing_topics.append(index)
 
        existing_topics = np.array(existing_topics)
        
        percent_topic_hit = 0
        diversity_over_hit_topics = 0
        
        # oracle selection
        oracle_vec = class_oracle_vectors[class_map[c]]
        oracle_priority = np.argsort(-oracle_vec.T[0])
        
        oracle_values = oracle_vec[oracle_priority]
        
        j = 0
        count = 0
        done = False
        while j < 1:
            rand_items = available_indices[np.random.choice(len(available_indices), M, replace=False)]
            rand_ids = item_list[rand_items]
            
            for i in range(len(oracle_priority)):
                
                for q in range(len(rand_items)):
                    r_id = rand_ids[q]
                    
                    topic = oracle_priority[i]
                    score = oracle_values[i][1]
                    
                    r_label = item_polarity.loc[item_polarity['article_id'] == int(r_id)]['source_partisan_score'].values[0]
                    r_topics = np.array(ast.literal_eval(item_topic.loc[item_topic['article_id'] == int(r_id)]['topical_vector'].values[0]))
                    
                    topic_ind = np.where(r_topics > 0)[0] // 5
                    
                    if np.any(topic_ind == topic) and int(score) == int(r_label):
                        print(f'Matched for item {r_id} at priority {i} on iteration {count}')
                        oracle_id = r_id
                        j+=1
                        done = True
                        break
            
                if done:
                    break
                    
            count += 1
        
        chosen_ids = item_list[chosen_items]
            
        for item_id in range(len(chosen_ids)):
         
            c_id = chosen_ids[item_id]
         
            u_choice = users_choice[u]
            
            o_label = item_polarity.loc[item_polarity['article_id'] == int(oracle_id)]['source_partisan_score'].values[0]
            c_label = item_polarity.loc[item_polarity['article_id'] == int(c_id)]['source_partisan_score'].values[0]
            
            o_topics = ast.literal_eval(item_topic.loc[item_topic['article_id'] == int(oracle_id)]['topical_vector'].values[0])
            c_topics = ast.literal_eval(item_topic.loc[item_topic['article_id'] == int(c_id)]['topical_vector'].values[0])
            
            
            
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
                    diversities.append(1 - row[lab])
                diversity_over_hit_topics += np.max(diversities)               
            
            
            o_utility = user_interaction_score(u_choice, o_topics)
            c_utility = user_interaction_score(u_choice, c_topics)
            
            idx = class_map[c]
            
            oracle_partisan_score[o_label+2] += 1
            chosen_partisan_score[c_label+2] += 1
            
            oracle_per_class_score[idx][o_label+2] += 1
            chosen_per_class_score[idx][c_label+2] += 1
            
            o_stats = np.array(o_topics).reshape((14, 5))
            c_stats = np.array(c_topics).reshape((14, 5))
            
            recommendation_stats[idx] += c_stats
            oracle_stats[idx] += o_stats
                
            
            oracle_utility_across_classes[idx] += o_utility
            chosen_utility_across_classes[idx] += c_utility
            
        ##NEEDS TO CHANGE FOR ALL USERS
        user_metrics.append(np.array([percent_topic_hit / len(chosen_ids), diversity_over_hit_topics / percent_topic_hit]))
    
            
    random_performance = np.divide(oracle_utility_across_classes, np.multiply(M, number_of_users))
    model_performance = np.divide(chosen_utility_across_classes, np.multiply(M, number_of_users))
    
    
    for i in range(recommendation_stats.shape[0]):
        
        # pd.DataFrame(recommendation_stats[i], columns=['-2', '-1', '0', '1', '2'], index=['abortion', 'environment', 'guns', 'health care', 'immigration', 'LGBTQ', 'racism', 'taxes',
        #   'technology', 'trade', 'trump impeachment', 'us military', 'us 2020 election', 'welfare']).to_csv(f'src\\data\\results2\\recommended\\{classes[i]}.csv')
        # pd.DataFrame(oracle_stats[i], columns=['-2', '-1', '0', '1', '2'], index=['abortion', 'environment', 'guns', 'health care', 'immigration', 'LGBTQ', 'racism', 'taxes',
        #   'technology', 'trade', 'trump impeachment', 'us military', 'us 2020 election', 'welfare']).to_csv(f'src\\data\\results2\\oracle\\{classes[i]}.csv')
        
        # pd.DataFrame(chosen_per_class_score[i]).to_csv(f'src\\data\\results2\\recommended\\partisan_dist_{classes[i]}.csv')
        # pd.DataFrame(oracle_per_class_score[i]).to_csv(f'src\\data\\results2\\oracle\\partisan_dist_{classes[i]}.csv')
        pass
    
    
    for i in range(len(classes)):
        cl = classes[i]

        print(f'{cl}: ')
        
        fig, (ax1, ax2) = plt.subplots(1, 2)

        arr = recommendation_stats[i]
        arr2 = original_interaction_stats[i]

        # total = np.sum(arr.flatten())
        # print(total)
        # arr /= total
        
        total = np.sum(arr2.flatten())
        arr2 /= total
        
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 8), constrained_layout=True)  # Horizontally stacked

        # Plot the first heatmap
        im1 = axes[0].imshow(arr, cmap='Blues', interpolation='none')
        axes[0].set_title(f"Topic Cov: {user_metrics[i][0]}, Div: {user_metrics[i][1]}")  # Title for the first subplot
        axes[0].set_xticks(np.arange(5))
        axes[0].set_xticklabels([-2, -1, 0, 1, 2])
        axes[0].set_yticks(np.arange(len(chosen_topic)))
        axes[0].set_yticklabels(chosen_topic)

        # Plot the second heatmap
        im2 = axes[1].imshow(arr2, cmap='Blues', interpolation='none')
        axes[1].set_title("User Interest relative to Ratings")  # Title for the second subplot
        axes[1].set_xticks(np.arange(5))
        axes[1].set_xticklabels([-2, -1, 0, 1, 2])
        axes[1].set_yticks(np.arange(len(chosen_topic)))
        axes[1].set_yticklabels(['','','','','','','','','','','','','',''])


        im3 = axes[2].imshow(total_topic_dist, cmap='Blues', interpolation='none')
        axes[2].set_title("Topic Distribution in Item Set")  # Title for the second subplot
        axes[1].set_xticks(np.arange(0))
        axes[1].set_xticklabels([])
        axes[2].set_yticks(np.arange(len(chosen_topic)))
        axes[2].set_yticklabels(['','','','','','','','','','','','','',''])


        # Add colorbars for both plots
        fig.colorbar(im1, ax=axes[0], orientation='vertical', shrink=0.8)
        fig.colorbar(im2, ax=axes[1], orientation='vertical', shrink=0.8)
        fig.colorbar(im3, ax=axes[2], orientation='vertical', shrink=0.8)

        # Save the figure
        plt.savefig(f'src\\data\\graphs\\{cl}.png')

        

    print(f'Model performance across classes: {model_performance}, with bias distribution: {chosen_partisan_score}')
    print(f'Random performance across classes: {random_performance}, with bias distribution: {oracle_partisan_score}')
    
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
    print(list)
    
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
    # create_predicted_rating_matrix()
    evaluate()
    
    # simple_choice_calc()