import pandas as pd
import numpy as np


df = pd.read_csv('src\data\\baseline_data\\testing_interactions.csv')
test_set = pd.read_csv('src\data\\baseline_data\\baseline_testing_data.csv')

user_item_matrix = np.zeros((1000, 4000))
held_out_interactions = np.zeros((1000, 4000))
# id_dictionary = {}

indexing_count = 0
for i in range(1000):
    ## filter all interactions of user i
    
    user_interactions = df[df['user_id'] == i]
    pos_interactions = user_interactions[user_interactions['click'] == 1]
    
    # for q in range(pos_interactions.shape[0]):
        
    #     indexing_count[pos_interactions.iloc[q]['article_id']] = indexing_count
    #     indexing_count += 1
    
    for j in range(pos_interactions.shape[0] // 2):
        item_id = pos_interactions.iloc[j]['article_id']
        ind = test_set[test_set['article_id'] == item_id].index.values[0]
        user_item_matrix[i][ind] = pos_interactions.iloc[j]['interest']
        
    for j in range(pos_interactions.shape[0] // 2, pos_interactions.shape[0]):
        item_id = pos_interactions.iloc[j]['article_id']
        ind = test_set[test_set['article_id'] == item_id].index.values[0]
        held_out_interactions[i][ind] = pos_interactions.iloc[j]['interest']
        
        
pd.DataFrame(user_item_matrix).to_csv('src\data\CF_test_correlation\\user_item_matrix.csv')
pd.DataFrame(held_out_interactions).to_csv('src\data\CF_test_correlation\\holdouts.csv')