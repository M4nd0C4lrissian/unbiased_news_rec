import pickle 
import numpy as np
import pandas as pd
import os
import sys
import torch
import ast
from create_landmarks import user_interaction
from sklearn.metrics.pairwise import linear_kernel
from scipy.stats.stats import pearsonr   

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from my_work.training import Encoder, Decoder


## change how this works (load the interactions from the user_item_matrix and then get the associated embedding from the baseline_testing data)

def generate_embedding(user_interactions, text_embedding_file, title_embedding_file, encoder_output_dim, encoder, polarity_decoder, polarity_free_decoder):
    
    source_path = "D:\Bert-Embeddings\\train_data\\"
    
    polarized_sum = np.zeros(encoder_output_dim, dtype = np.float64)
    free_sum = np.zeros(encoder_output_dim, dtype = np.float64)
    total_utility_score = 0
    
    indices = np.where(user_interactions > 0)[0]
    values = user_interactions[indices]
    
    # print(indices)
    # print(values)
    with torch.no_grad():
        
        for i in range(len(values)):
            
            utility_score = values[i]
            
            total_utility_score += utility_score

            text = text_embedding_file[indices[i]]
            title = title_embedding_file[indices[i]]

            x2, _ = encoder(torch.cat((title.T.unsqueeze(0), text.T.unsqueeze(0)), dim=-1))
            polarity_rep = polarity_decoder(x2)
            polarity_free_rep = polarity_free_decoder(x2)

            # print("Inference complete...")

            polarized_sum = np.add(polarized_sum, utility_score * polarity_rep.detach().numpy())
            free_sum = np.add(free_sum, utility_score * polarity_free_rep.detach().numpy())
    
    polarized_sum /= total_utility_score
    free_sum /= total_utility_score

    ##return (embedding, user_item vector)
    return polarized_sum, free_sum

bert_dim = 768 # Example BERT embedding size
intermediate_dim = 256
encoder_output_dim = 128

encoder = Encoder(bert_dim, intermediate_dim, encoder_output_dim)
polarity_decoder = Decoder(encoder_output_dim, intermediate_dim, encoder_output_dim)
polarity_free_decoder = Decoder(encoder_output_dim, intermediate_dim, encoder_output_dim)

encoder.load_state_dict(torch.load('src\my_work\models\encoder.pt', weights_only=True))
polarity_decoder.load_state_dict(torch.load('src\my_work\models\polarity_decoder.pt', weights_only=True))
polarity_free_decoder.load_state_dict(torch.load('src\my_work\models\polarity_free_decoder.pt', weights_only=True))

encoder.eval()
polarity_decoder.eval()
polarity_free_decoder.eval()



##must be in embedding order
##topic_lists = pd.read_csv("src\data\\baseline_data\\baseline_testing_data.csv")     
source_path = "D:\Bert-Embeddings\\test_data\\"

text_embedding_file = torch.load(source_path + f"text_embedding_{0}.pt")
title_embedding_file = torch.load(source_path + f"title_embedding_{0}.pt")

for i in range(1,4):
    text_embedding_file = torch.cat((text_embedding_file, torch.load(source_path + f"text_embedding_{i}.pt")), dim=0)
    title_embedding_file = torch.cat((title_embedding_file, torch.load(source_path + f"title_embedding_{i}.pt")), dim=0)
    


landmarks = pd.read_csv("src\data\\baseline_data\landmarks\landmark_embeddings.csv").drop(['Unnamed: 0'], axis=1)
norm_dist = np.ones(landmarks.iloc[0].size)

for i in range(landmarks.shape[1]):
    col = landmarks.iloc[:][str(i)]
    
    d = col.describe()
    
    max_diff = abs(d['max'] - d['min'])
    norm_dist[i] = max_diff
    
    
pd.DataFrame(norm_dist).to_csv('src\\data\\baseline_data\landmarks\max_dist.csv')


normalized_distance = pd.read_csv('src\\data\\baseline_data\landmarks\\max_dist.csv')['0'].to_numpy()

def dist(a, b, normalized_distance):
    dist = a - b
    n_dist = np.divide(dist, normalized_distance)
    return np.linalg.norm(n_dist)
    # return np.linalg.norm(dist)

# def dist(a, b):
#     return pearsonr(a,b)[0]

def landmark_embedding(landmarks, embedding, normalized_distance):
    return [dist(np.array(lm), embedding, normalized_distance) for _ , lm in landmarks.iterrows()]


## instead of this - we take the interaction data and build the 128 dimension embedding from that

user_item = pd.read_csv('src\data\CF_test_correlation\\user_item_matrix.csv').drop(['Unnamed: 0'], axis=1)


polarity_free_interest_model = pd.DataFrame(columns=['interest model'])

distance_embeddings = np.zeros((1000, 9), dtype=np.float64)

p_free_embeddings = []
polarized_embeddings = []

for user in range(user_item.shape[0]):
    # print('User:' , user)
    user_interactions = user_item.iloc[user]
    
    user_landmark_embedding, polarity_free_embedding = generate_embedding(user_interactions, text_embedding_file, title_embedding_file, encoder_output_dim, encoder, polarity_decoder, polarity_free_decoder)
    # new = landmark_embedding(landmarks, user_landmark_embedding, normalized_distance)
    # distance_embeddings[user] = new
    
    # p_free_embeddings.append({'interest model': polarity_free_embedding})
    polarized_embeddings.append({'interest model': user_landmark_embedding[0]})
    
# polarity_free_interest_model = pd.concat([polarity_free_interest_model, pd.DataFrame.from_records(p_free_embeddings)], ignore_index=True)
pd.DataFrame(polarized_embeddings).to_csv('src\\data\\baseline_data\\CF\\bias_models.csv')
# ## 9d with politype class label

# correlation = np.corrcoef(distance_embeddings)
# distance_embeddings =  pd.DataFrame(distance_embeddings)
# distance_embeddings.to_csv('src\data\\baseline_data\\user_space\\user_space_matrix.csv')

# ##Now - should have a 9d vector for each of our 1000 users, now we have to calculate all of their similarity

# pd.DataFrame(correlation).to_csv('src\data\\baseline_data\\CF\\correlation_matrix.csv')


# ## did not move to baseline data
# polarity_free_interest_model.to_csv('src\\data\\baseline_data\\CF\\interest_models.csv')

##edits - use training data (only 4000 items)
## also create a user-item matrix 


##I'm dumb it was just reloading the data every time