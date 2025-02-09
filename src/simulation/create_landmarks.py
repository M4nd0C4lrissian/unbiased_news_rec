import torch
import numpy as np
import pandas as pd
import ast
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from my_work.training import Encoder, Decoder

def user_interaction(uv, iv):
    """
    Given a user vector (uv) and a recommended new, 
    return whether user is gonna click or not
    """

    # iv = recommended_News["topical_vector"]

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
        return True, pui
    else:
        return False, 0

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

if __name__ == '__main__':
    
    bert_dim = 768  # Example BERT embedding size
    intermediate_dim = 256
    encoder_output_dim = 128

    encoder = Encoder(bert_dim, intermediate_dim, encoder_output_dim)
    polarity_decoder = Decoder(encoder_output_dim, intermediate_dim, encoder_output_dim)


    encoder.load_state_dict(torch.load('src\my_work\models\encoder.pt', weights_only=True))
    polarity_decoder.load_state_dict(torch.load('src\my_work\models\polarity_decoder.pt', weights_only=True))

    encoder.eval()
    polarity_decoder.eval()


    classes = ['bystanders', 'core conserv', 'country first conserv', 'devout and diverse', 'disaffected democrats', 'market skeptic repub', 'new era enterprisers', 'oppty democrats', 'solid liberas']

    class_utility = np.zeros((9,70), dtype=np.float64)
    for c in range(len(classes)):
        avg_member = pd.read_csv('src\\data\synthetic_user\\' + classes[c] +'.csv', header=None)
        class_utility[c] = avg_member.to_numpy().flatten()
        
    ##need to do item_sampling, meaning:

    #complete ## INPUT: table schema training items item_topics = (item_id, topical_vector) ordered as our partisan_scores.csv file is relative to item_id (same order as embeddings)

    ## we randomly sample from item_topics, pulling their embeddings from the associated .pt file, until we have a certain number M of 'hits' - interacted-with articles
    ## instead of constantly reloading large pt files, we do uniform sampling across all 32 batches of 1000 for each user type - first we need to load the users and
    ## calculate user choice
    test_data = pd.read_csv('src\data\\baseline_data\\baseline_testing_data.csv')
    test_data = test_data[['article_id', 'topical_vector']]
    #topic_lists = pd.read_csv("src\data\\landmark_data\\topics_in_embedding_order.csv")
    ##Change here once more data
    num_batches = 1

    batch_size = 4000
    num_pos_samples = 20

    source_path = "D:\Bert-Embeddings\\test_data\\"
    type_landmarks = np.zeros((9, encoder_output_dim), dtype = np.float64)
    total_utility_score = np.zeros(9)
    
    text_embedding_file = torch.load(source_path + f"text_embedding_{0}.pt")
    title_embedding_file = torch.load(source_path + f"title_embedding_{0}.pt")
    
    for i in range(1,4):
        text_embedding_file = torch.cat((text_embedding_file, torch.load(source_path + f"text_embedding_{i}.pt")), dim=0)
        title_embedding_file = torch.cat((title_embedding_file, torch.load(source_path + f"title_embedding_{i}.pt")), dim=0)

    for i in range(num_batches):

        print('Loaded embeddings!')

        with torch.no_grad():

            for type in range(9):  
                pos_samples = 0

                while pos_samples < num_pos_samples:
                    r = np.random.randint(0, batch_size)
                    
                    ## r is raw value from 0 to 4000, but we need to get the actual article_id - its equal to the index of the test-set

                    acc = i * batch_size + r

                    topic_vector = ast.literal_eval(test_data.iloc[acc]['topical_vector'])

                    did_interact, utility_score = user_interaction(class_utility[type], topic_vector)

                    if did_interact:
                    
                        print(f'User interacted with a utility score of {utility_score}')

                        total_utility_score[type] += utility_score

                        text = text_embedding_file[r]
                        title = title_embedding_file[r]

                        x2, _ = encoder(torch.cat((title.T.unsqueeze(0), text.T.unsqueeze(0)), dim=-1))
                        polarity_rep = polarity_decoder(x2)

                        print("Inference complete...")
                        ##print(polarity_rep.detach().numpy())
                        type_landmarks[type] = np.add(type_landmarks[type], utility_score * polarity_rep.detach().numpy())

                        pos_samples += 1

                    else:
                        ##print('Did not interact... continuing')
                        continue

    for type in range(9):
        type_landmarks[type] /= total_utility_score[type]

    df = pd.DataFrame(type_landmarks)
    df.to_csv("src\data\\baseline_data\landmarks\\landmark_embeddings.csv")
    print(total_utility_score)


    # we compute their polarity-encoded representation, and perform a weighted average over them according to the user_interaction score
    ## upon doing this for each class, we will have a 64 dimension user profile for each of the 9,
    ## for other users, their 64 dimension embedding will be cast to 9, each element being the distance of their 64-d vector to each landmark 64 d
    
    ##just try normalizing them after