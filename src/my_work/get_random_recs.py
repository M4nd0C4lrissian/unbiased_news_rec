import pandas as pd
import numpy as np
import torch
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from my_work.training import Encoder, Decoder
from my_work.custom_article_embedding_dataset import CustomArticleEmbeddingDataset as CD
import re


# bert_dim = 768  # Example BERT embedding size
# intermediate_dim = 256
# encoder_output_dim = 128

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# encoder = Encoder(bert_dim, intermediate_dim, encoder_output_dim).to(device)
# polarity_free_decoder = Decoder(encoder_output_dim, intermediate_dim, encoder_output_dim).to(device)

# encoder.load_state_dict(torch.load('src\my_work\models\encoder.pt', weights_only=True))
# polarity_free_decoder.load_state_dict(torch.load('src\my_work\models\polarity_free_decoder.pt', weights_only=True))

# encoder.eval()
# polarity_free_decoder.eval()


class_distribution = [49, 193, 133, 108, 69, 116, 116, 63, 153]
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


source_path = "D:\Bert-Embeddings\\test_data\\"
labels_file = "src\data\\baseline_data\CF\per_user\\testing_partisan_labels.csv"

# text_paths = []
# title_paths = []
# for i in range(4):
#     text_paths.append(source_path + f"text_embedding_{i}.pt")
#     title_paths.append(source_path + f"title_embedding_{i}.pt")
    

# dataloader = CD(labels_file, text_paths, title_paths, [0, 4000])


test_data = pd.read_csv("src\data\\baseline_data\\baseline_testing_data.csv",skipinitialspace=True)
user_item_matrix = pd.read_csv('src\data\CF_test_correlation\\user_item_matrix.csv').drop(columns=['Unnamed: 0'])

topic_vectors = test_data[['article_id','topical_vector']]

item_polarity = test_data[['article_id', 'source_partisan_score']]

out_path = 'FN_Embedding_CPC'
out_path = 'NN_Rating_Rating'
user_metrics = pd.read_csv(f"src\data\\baseline_data\\total_eval\\results\{out_path}.csv").drop(columns=['Unnamed: 0'])
users_choice = pd.read_csv("src\data\\baseline_data\\testing_user_choice_vectors.csv").drop(columns=['Unnamed: 0'])




recs = user_metrics['chosen_items']

j = np.random.randint(0, 1000, 1, int)[0]
j = 554

##actual ids
items_chosen = list(map(int, re.findall(r'\d+', recs.iloc[j])))
# temp = np.random.randint(0, 4000, 10, int)
# items_chosen = [test_data.iloc[k]['article_id'] for k in temp]

M = len(items_chosen)
uv = np.array(users_choice.iloc[j])

prev_ids = np.array(user_item_matrix.iloc[j])
prev_ids = np.where(prev_ids != 0)[0]

print('User: ', j)

for p in prev_ids:
    print(test_data.iloc[p]['title'])
    print(test_data.iloc[p]['cls_label'])
    print(test_data.iloc[p]['source_partisan_score'])
    print('-----------------------------------------')

# CTR, UTILITY
for i in range(10):

    print('Recommended:')
    row = test_data.loc[test_data['article_id'] == int(items_chosen[i])]
    print(row['title'].values[0])
    print(row['cls_label'].values)
    print(row['source_partisan_score'].values[0])
    print('-----------------------------------------')

print('Diversity: ', user_metrics.iloc[j]['diversity'])
print('Topic Hit: ', user_metrics.iloc[j]['topic_hit'])
print('User: ', j)
#233
#554