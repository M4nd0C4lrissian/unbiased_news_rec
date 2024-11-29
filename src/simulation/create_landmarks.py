import torch
import numpy as np
import pandas as pd
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from my_work.training import Encoder, Decoder

bert_dim = 60 + 128  # Example BERT embedding size
intermediate_dim = 128
encoder_output_dim = 64

encoder = Encoder(bert_dim, intermediate_dim, encoder_output_dim)
polarity_decoder = Decoder(encoder_output_dim, intermediate_dim, encoder_output_dim)
polarity_free_decoder = Decoder(encoder_output_dim, intermediate_dim, encoder_output_dim)

encoder.load_state_dict(torch.load('src\my_work\models\encoder.pt', weights_only=True))
polarity_decoder.load_state_dict(torch.load('src\my_work\models\polarity_decoder.pt', weights_only=True))
polarity_free_decoder.load_state_dict(torch.load('src\my_work\models\polarity_free_decoder.pt', weights_only=True))

encoder.eval()
polarity_decoder.eval()
polarity_free_decoder.eval()


classes = ['bystanders', 'core conserv', 'country first conserv', 'devout and diverse', 'disaffected democrats', 'market skeptic repub', 'new era enterprisers', 'oppty democrats', 'solid liberas']

landmarks = np.array((9,64), dtype=np.float32)
for c in classes:
    avg_member = pd.read_csv('src\\data\synthetic_user\\' + c +'.csv', header=None)
    print(avg_member.head())
    print(avg_member.describe())
    ##need to be flattened into dimension 70 vectors
    
##need to do item_sampling, meaning:

#complete ## INPUT: table schema training items item_topics = (item_id, topical_vector) ordered as our partisan_scores.csv file is relative to item_id (same order as embeddings)

## we randomly sample from item_topics, pulling their embeddings from the associated .pt file, until we have a certain number M of 'hits' - interacted-with articles
# we compute their polarity-encoded representation, and perform a weighted average over them according to the user_interaction score
## upon doing this for each class, we will have a 64 dimension user profile for each of the 9,
## for other users, their 64 dimension embedding will be cast to 9, each element being the distance of their 64-d vector to each landmark 64 d
    
    
def user_interaction(uv, recommended_News, ranked=True):
    """
    Given a user vector (uv) and a recommended new, 
    return whether user is gonna click or not
    """

    iv = recommended_News["topical_vector"]

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