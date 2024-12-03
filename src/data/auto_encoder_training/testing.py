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

import torch

# x = torch.tensor([[0, 1, 2], [0, 0, 3]])
# condition = x > 1

# # torch.nonzero returns a 2D tensor of indices
# result = torch.nonzero(condition)

# print(result)

landmarks = pd.read_csv("src\data\\landmark_data\\landmark_embeddings.csv").drop(columns=['Unnamed: 0'])
norm_dist = np.ones(landmarks.iloc[0].size)

for i in range(landmarks.shape[1]):
    col = landmarks.iloc[:][str(i)]
    
    d = col.describe()
    
    max_diff = abs(d['max'] - d['min'])
    norm_dist[i] = max_diff
    
    
pd.DataFrame(norm_dist).to_csv('src\\data\\landmark_data\\max_dist.csv')