import pickle 
import numpy as np
import pandas as pd
import os
import sys
import torch
import ast
from create_landmarks import user_interaction


    
    
# testing = pd.read_csv("src\\data\\auto_encoder_training\\bert_testing_set.csv")
# mask = pd.read_csv("src\\data\\auto_encoder_training\\testing_mask.csv")

# df = mask[mask.columns[[2, 4]]]

# full_data = pd.merge(testing, df, on='article_id')
# full_data.to_csv("src\\data\\baseline_data\\baseline_testing_data.csv")


# testing_data = pd.read_csv("src\\data\\baseline_data\\baseline_testing_data.csv")
# topical_vectors = pd.read_csv("src\\data\\landmark_data\\testing_topics_in_embedding_order.csv")

# full_data = pd.merge(testing_data, topical_vectors, on='article_id')

# full_data.drop(columns=["Unnamed: 0.1"], inplace=True)
# full_data.drop(columns=["Unnamed: 0_x"], inplace=True)
# full_data.drop(columns=["Unnamed: 0_y"], inplace=True)

# full_data.to_csv("src\\data\\baseline_data\\baseline_testing_data.csv")

# print('wait')

testing_data = pd.read_csv("src\\data\\baseline_data\\baseline_testing_data.csv")
users = pd.read_pickle('testing_1000users.pkl')

class_dist = [49, 193, 133, 108, 69, 116, 116, 63, 153]
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
users_choice = np.empty((1000, 70))

q = 0
for i in range(len(classes)):
    cl = classes[i]
    for j in range(class_dist[i]):
        users_choice[q] = np.array(users[cl][j]).flatten()
        q+=1

pd.DataFrame(users_choice).to_csv("src\\data\\baseline_data\\testing_users_choice.csv")

item_num = 4000

interactions = pd.DataFrame(columns=['user_id', 'article_id', 'rel_timestamp', 'click', 'interest'])

for u in range(users_choice.shape[0]):
    
    if u % 1000 == 0:
        print(f"{u} users simulated, total interactions: {interactions.shape[0]}")
    
    uv = users_choice[u]
    
    num_pos_samples = 20
    pos_samples = 0
    interaction_num = 1
    
    user_interactions = []
    
    while pos_samples < num_pos_samples:
        r = np.random.randint(0, item_num)

        topic_vector = ast.literal_eval(testing_data.iloc[r]['topical_vector'])

        did_interact, utility_score = user_interaction(uv, topic_vector)
        
        user_interactions.append({'user_id' : u, 'article_id' : testing_data.iloc[r]['article_id'], 'rel_timestamp' : interaction_num, 'click' : int(did_interact), 'interest' : utility_score})
        
        if did_interact:
            pos_samples+=1
        
        interaction_num += 1
        
    interactions = pd.concat([interactions, pd.DataFrame(user_interactions)], ignore_index=True)
    
interactions.to_csv("src\\data\\baseline_data\\testing_interactions.csv")
        
        