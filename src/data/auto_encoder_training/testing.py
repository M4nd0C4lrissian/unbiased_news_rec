import torch
import pandas as pd
import numpy as np
import ast

# data = torch.load('unbiased_news_rec\src\data\\auto_encoder_training\\title_embedding_0.pt')
# print(data[2])

# UI = pd.read_csv("src\\data\\CF\\user_item_matrix.csv")

# print(UI.size)
# print(UI.shape)
# print(UI.head())

# print(UI.describe())

# def process_data(data):
#     data = data.strip("[]")
#     elements = data.split()
#     numbers = list(map(float, elements))
    
#     return numbers

# data = "[-5.50317552e-02 2 3]"

# print(process_data(data))


# x = torch.tensor([[0, 1, 2], [0, 0, 3]])
# condition = x > 1

# # torch.nonzero returns a 2D tensor of indices
# result = torch.nonzero(condition)

# print(result)

# landmarks = pd.read_csv("src\data\\landmark_data\\landmark_embeddings.csv").drop(columns=['Unnamed: 0'])
# norm_dist = np.ones(landmarks.iloc[0].size)

# for i in range(landmarks.shape[1]):
#     col = landmarks.iloc[:][str(i)]
    
#     d = col.describe()
    
#     max_diff = abs(d['max'] - d['min'])
#     norm_dist[i] = max_diff
    
    
# pd.DataFrame(norm_dist).to_csv('src\\data\\landmark_data\\max_dist.csv')

import matplotlib.pyplot as plt



# loss_over_time = pd.read_csv(f'src\\data\\CF\\trained_h_{7}_loss_over_time.csv')['0'].to_numpy()
# plt.scatter(range(len(loss_over_time)), loss_over_time, s=2)
# plt.savefig(f'src\\data\\CF\\trained_h_{7}_loss_over_time.png')
# plt.show()


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
    
# pd.DataFrame(more_per_topic).to_csv('src\\data\\results\\total_topic_dist.csv')

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# print(torch.cuda.get_device_name(device))

# df = pd.read_csv('src\data\\baseline_data\\baseline_testing_data.csv')

# df = df[['article_id', 'source_partisan_score']]

# df.to_csv('src\data\\baseline_data\CF\per_user\\testing_partisan_labels.csv')

# classes = [
# "bystanders",
# "solid liberas",
# "oppty democrats",
# "disaffected democrats",
# "devout and diverse",
# "new era enterprisers",
# "market skeptic repub",
# "country first conserv",
# "core conserv",
# ]

# class_map = {'bystanders': 0, 'core conserv': 8, 'country first conserv': 7, 'devout and diverse': 4, 'disaffected democrats': 3, 'market skeptic repub': 6, 'new era enterprisers': 5, 'oppty democrats': 2, 'solid liberas': 1}

# number_of_users = [49, 193, 133, 108, 69, 116, 116, 63, 153]

# user_metrics = pd.read_csv('src\\data\\baseline_data\\total_eval\\all_user_metrics.csv').drop(columns=['Unnamed: 0'])

# curr = 0
# for i in range(len(classes)):
#     class_users = user_metrics.iloc[curr: curr + number_of_users[i]]
#     print(f'Class {classes[i]}----------------- ')
#     print('Topic coverage: ')
#     print(class_users['topic_hit'].describe())
#     print()
#     print('Diversity over hits: ')
#     print(class_users['diversity'].describe())
#     print()
#     curr += number_of_users[i]
    
#####
### make pure rating correlation matrix

user_item_matrix = pd.read_csv("src\\data\\CF_test_correlation\\user_item_matrix.csv").drop(columns=['Unnamed: 0'])

correlation = np.corrcoef(user_item_matrix)

##pure rating correlation matrix

pd.DataFrame(correlation).to_csv("src\\data\\baseline_data\\CF\\rating_correlation_matrix.csv")