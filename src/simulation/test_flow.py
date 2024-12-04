import torch
import numpy as np
import pandas as pd
import ast

import graph_convolutions as GC


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

def create_predicted_rating_matrix():
 
    correlation_matrix = pd.read_csv('src\\data\\user_space\\correlation_matrix.csv').drop(columns=['Unnamed: 0']).to_numpy()
    np.fill_diagonal(correlation_matrix, 0)
    B = correlation_matrix / 1000

    f = 5
    ## was 100 in training - I did not realize 
    k = 10

    Bi = GC.normalized_bottom_k_with_bias(B, k)
    B_i = GC.construct_convolutions_with_user_check(Bi, f)

   

    h = torch.tensor(pd.read_csv('src\\data\\CF\\trained_h_5_epoch_4.csv')['0'].to_numpy(), dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
    s = torch.nn.Sigmoid()
    h = s(h)

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
            # if torch.isinf(x_i).any() or torch.isinf(x_hat).any():
            #     print("Found inf in x_i or x_hat")

            # mask = x_i != 0  # Only consider entries where actual ratings are available
            
            ##here - need to find top M rated, and find their embeddings, and scale them by their predicted responses
            
            rating_matrix = torch.cat((rating_matrix, x_hat.unsqueeze(0).T), dim = -1)
            
        print(f"rating_matrix shape: {rating_matrix.shape}")

    pd.DataFrame(rating_matrix.cpu().detach().numpy()).to_csv('src\\data\\predicted_rating_matrix.csv')


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
    M = 8
    item_topic = pd.read_csv('src\\data\\landmark_data\\validation_topics_in_embedding_order.csv')
    item_polarity = pd.read_csv('src\\data\\auto_encoder_training\\validation_data\\validation_partisan_labels.csv')

    user_classes = pd.read_csv('src\\data\\user_space\\user_space_matrix_with_topics.csv').iloc[:]['class']

    user_item_matrix = pd.read_csv('src\\data\\CF\\user_item_matrix.csv').drop(columns=['Unnamed: 0'])

    predicted_rating_matrix = pd.read_csv('src\\data\\predicted_rating_matrix.csv').to_numpy()

    item_list = user_item_matrix.columns

    random_partisan_score = [0,0,0]
    chosen_partisan_score = [0, 0, 0]

    random_utility_across_classes = [0,0,0,0,0,0,0,0,0]
    chosen_utility_across_classes = [0,0,0,0,0,0,0,0,0]

    classes = ['bystanders', 'core conserv', 'country first conserv', 'devout and diverse', 'disaffected democrats', 'market skeptic repub', 'new era enterprisers', 'oppty democrats', 'solid liberas']
    class_map = {'bystanders': 0, 'core conserv': 1, 'country first conserv': 2, 'devout and diverse': 3, 'disaffected democrats': 4, 'market skeptic repub': 5, 'new era enterprisers': 6, 'oppty democrats': 7, 'solid liberas': 8}
    number_of_users = [49, 153, 63, 69, 108, 116, 116, 133, 193]

    t = pd.read_pickle('1000users.pkl')


    users_choice = np.empty((1000, 70))
    q = 0
    for i in range(len(classes)):
        cl = classes[i]
        for j in range(number_of_users[i]):
            users_choice[q] = np.array(t[cl][j]).flatten()
            q+=1

    for u in range(user_item_matrix.shape[0]):
        
        row = user_item_matrix.iloc[u]
        valid_mask = (row > 0)
        filtered_row = row[~valid_mask]
        available_indices = np.where(valid_mask)[0]
        
        c = user_classes.iloc[u]
        
        
        predicted_row = predicted_rating_matrix[u]
        filtered_pred = predicted_row[available_indices]

        ## top M
        retain_ind = np.argsort(-filtered_pred)[:M]
        retain_val = filtered_pred[retain_ind]
        
        ## M random
        rand_items = available_indices[np.random.choice(len(available_indices), M, replace=False)]
        
        chosen_items = available_indices[retain_ind]
        
        
        # retrieve polarity + gauge utility
        rand_ids = item_list[rand_items]
        chosen_ids = item_list[chosen_items]
        
        u_choice = users_choice[u]
        
        for i in range(M):
            r_id = rand_ids[i]
            c_id = chosen_ids[i]
            
            r_label = item_polarity.loc[item_polarity['article_id'] == int(r_id)]['source_partisan_score'].values[0]
            c_label = item_polarity.loc[item_polarity['article_id'] == int(c_id)]['source_partisan_score'].values[0]
            
            if r_label == 0:
                random_partisan_score[0] += 1
            elif r_label == 2:
                random_partisan_score[2] += 1
            else:
                random_partisan_score[1] += 1
                
            if c_label == 0:
                chosen_partisan_score[0] += 1
            elif c_label == 2:
                chosen_partisan_score[2] += 1
            else:
                chosen_partisan_score[1] += 1
            
            r_topics = ast.literal_eval(item_topic.loc[item_topic['article_id'] == int(r_id)]['topical_vector'].values[0])
            c_topics = ast.literal_eval(item_topic.loc[item_topic['article_id'] == int(c_id)]['topical_vector'].values[0])
            
            r_utility = user_interaction_score(u_choice, r_topics)
            c_utility = user_interaction_score(u_choice, c_topics)
            
            idx = class_map[c]
            
            random_utility_across_classes[idx] += r_utility
            chosen_utility_across_classes[idx] += c_utility
            
        
            
    random_performance = np.divide(random_utility_across_classes, np.multiply(M, number_of_users))
    model_performance = np.divide(chosen_utility_across_classes, np.multiply(M, number_of_users))

    print(f'Model performance across classes: {model_performance}, with topic distribution: {chosen_partisan_score}')
    print(f'Random performance across classes: {random_performance}, with topic distribution: {random_partisan_score}')
    
##not really testing recommendation diversity at the individual level - should try this
if __name__ == '__main__':
    create_predicted_rating_matrix()
    evaluate()