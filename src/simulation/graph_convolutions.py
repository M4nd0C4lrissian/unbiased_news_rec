#### have to consider our goal here

## bad news - I think we will have to build a 1000 by 32000 matrix - to keep track of what items users have interacted with, and to what degree


## given user correlation matrix, and classical CF matrix of 1000 users and logged responses to V items - eventually this will be all 32000 items? (simulated), we can do our graph convolutions
## but - are we not planning on making one of these filters for every user? in that case, let's just do this over the testing data! only 4000 items - if the idea is that this will be computed daily for every user,
# we don't need to prove generalizability using train / test splits, and is just easier to handle

import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import math
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from my_work.training import Encoder, Decoder
from my_work.custom_article_embedding_dataset import CustomArticleEmbeddingDataset as CD

## need to re-confirm that this works (though I'm pretty sure I tested it before)
#change to bottom k
def normalized_bottom_k(Bi, k):
    
    norm_B = np.zeros_like(Bi, dtype=np.float64)
    
    for u in range(Bi.shape[0]):
        row = Bi[u]

        # Filter out NaNs and negative values
        valid_mask = ~np.isnan(row) & (row > 0)
        filtered_row = row[valid_mask]

        # Check if there are enough valid values to select
        if len(filtered_row) == 0:
            norm_B[u] = np.zeros_like(row)
            continue

        # Get indices of the bottom-k values
        retain_ind = np.argsort(filtered_row)[:k]
        retain_val = filtered_row[retain_ind]

        # Calculate the sum of the retained values for normalization
        s = np.sum(retain_val)
        if s == 0:
            norm_B[u] = np.zeros_like(row)
            continue

        # Create a new row with only the top-k values retained
        b = np.zeros_like(row)
        original_indices = np.where(valid_mask)[0][retain_ind]
        b[original_indices] = retain_val

        # Normalize to ensure the sum is between 0 and 1
        b = b / s

        norm_B[u] = b
        
    return norm_B
        

def construct_convolutions(Bi, f):
    # Convert NumPy array to a PyTorch tensor
    Bi_torch = torch.tensor(Bi, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')

    num_rows, num_cols = Bi_torch.shape
    # Initialize a PyTorch tensor to store the results
    tensor = torch.zeros((f, num_rows, num_cols), device=Bi_torch.device)

    # Set the first layer to Bi
    tensor[0] = Bi_torch
    for i in range(1, f):
        # Perform batched matrix multiplication across the third dimension
        tensor[i] = torch.matmul(tensor[i-1].float(), Bi_torch.float())
        tensor[i] = torch.nan_to_num(tensor[i], nan=0.0)

    return tensor

def weighted_graph_convolution(x_i, Bs, h):

  """
    Compute the convolution x_i Bs h.

    Parameters:
    - x_i: A 1 x U rating vector (numpy array).
    - Bs: A k x U x U tensor representing the graph shifts.
    - h: A 1 x k vector of weights.

    Returns:
    - A 1 x U shifted and weighted rating vector.
  """
  x_shifted = torch.stack([torch.matmul(x_i.float(), Bs[k].float()) for k in range(len(h))])  # shape: (k, U)

  weighted_sum = torch.matmul(h.T, x_shifted)  # shape: (1, U)

  return weighted_sum.flatten()

#need to - 
#1. get the top M rated items and their ratings
#2. extract their item_ids (columns of user-item matrix)
#3. using validation_partisan_labels, find the absolute index of the row whose item_id equals the id of the top M items
#4. find their embeddings in one of the 4 pt files (use mod)
#5. weighted sum them

def get_predicted_embedding(x_hat, M, user_item_matrix, encoder_output_dim, partisan_labels, val_data, encoder, polarity_free_decoder, batch_size = 1000):
    item_list = user_item_matrix.columns
    row = x_hat.detach().clone()

    valid_mask = ~np.isnan(row) & (row > 0)
    filtered_row = row[valid_mask]

    # Get indices of the top-M values
    retain_ind = np.argsort(-filtered_row)[:M]
    retain_val = filtered_row[retain_ind]

    # Create a new row with only the top-k values retained
    b = np.zeros_like(row)
    original_indices = np.where(valid_mask)[0][retain_ind]
    b[original_indices] = retain_val
    
    item_ids = item_list[original_indices]
    combined_embedding = np.zeros( encoder_output_dim , dtype=np.float64)
    
    
    for i in range(len(item_ids)):
        
        id = item_ids[i]
        
        ##should be precisely one
        raw_index = int(partisan_labels.loc[partisan_labels['article_id'] == int(id)].index[0])
        
        ## let's assume we're concatenating all 4000 for now   
        # batch_num = raw_index // 1000
        
        # local_index = raw_index % 1000
        
        point = val_data.__getitem__(raw_index)
        
        title = point[0]
        text = point[1]
        
        x2, _ = encoder(torch.cat((title, text), dim=-1))
        polarity_free_rep = polarity_free_decoder(x2)
        
        aggregated_score = b[i]
        combined_embedding = np.add(combined_embedding, aggregated_score * polarity_free_rep.detach().numpy())
            
    return combined_embedding

def process_data(data):
    data = data.strip("[]")
    elements = data.split()
    numbers = list(map(float, elements))
    
    return numbers

def train_weights(matrix, B, M, true_interest_model, partisan_labels, val_data, encoder, polarity_free_decoder, encoder_output_dim = 128, k=10, f=3, lr=0.01, epochs=100):
    
    h = torch.nn.Parameter(torch.rand(f, 1, requires_grad=True))

    optimizer = optim.Adam([h], lr=lr)

    Bi = normalized_bottom_k(B, k)
    B_i = construct_convolutions(Bi, f)
    print(B_i)
    
    item_list = matrix.columns

    for epoch in range(epochs):
        total_loss = 0
        for item_id in range(matrix.shape[1]):

            # Get rating vector for the item

            x_i = matrix.iloc[:][str(item_list[item_id])].values
            #print(x_i)
            x_i = torch.tensor(x_i, dtype=torch.float64)
            #print(x_i)


            x_hat = weighted_graph_convolution(x_i, B_i, h)

            if torch.isnan(x_hat).any():
                print("Found NaN in x_hat")
            if torch.isnan(x_i).any():
                print("Found NaN in x_i")
            # if torch.isinf(x_i).any() or torch.isinf(x_hat).any():
            #     print("Found inf in x_i or x_hat")

            # mask = x_i != 0  # Only consider entries where actual ratings are available
            
            ##here - need to find top M rated, and find their embeddings, and scale them by their predicted responses
            print('Predicting embedding...')
            predicted_user_embedding = torch.tensor(get_predicted_embedding(x_hat, M, matrix, encoder_output_dim, partisan_labels, val_data, encoder, polarity_free_decoder), dtype=torch.float64)
            actual_user_embedding = torch.tensor(process_data(true_interest_model.iloc[item_id]['interest model']), dtype=torch.float64)
            
            loss = torch.mean((actual_user_embedding - predicted_user_embedding) ** 2)
            print('Loss calculated!')
            
            
            # loss = torch.mean((x_i[mask] - x_hat[mask]) ** 2)
            #print(loss.item())
            
            if not math.isnan(loss.item()):
              total_loss += loss.item()
              
            else:
                print('Problem!')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")

    return h.detach().numpy()


bert_dim = 60 + 256  # Example BERT embedding size
intermediate_dim = 256
encoder_output_dim = 128

encoder = Encoder(bert_dim, intermediate_dim, encoder_output_dim)
polarity_free_decoder = Decoder(encoder_output_dim, intermediate_dim, encoder_output_dim)

encoder.load_state_dict(torch.load('src\my_work\models\encoder.pt', weights_only=True))
polarity_free_decoder.load_state_dict(torch.load('src\my_work\models\polarity_free_decoder.pt', weights_only=True))

encoder.eval()
polarity_free_decoder.eval()


val_path = "D:\Bert-Embeddings\\validation_data\\"

labels_file = "src\data\\auto_encoder_training\\validation_data\\validation_partisan_labels.csv"

text_paths = []
title_paths = []
for i in range(4):
    text_paths.append(val_path + f"text_embedding_{i}.pt")
    title_paths.append(val_path + f"title_embedding_{i}.pt")
    

val_dataset = CD(labels_file, text_paths, title_paths, [0, 4000])
    
partisan_labels = pd.read_csv("src\data\\auto_encoder_training\\validation_data\\validation_partisan_labels.csv").drop(columns=['Unnamed: 0'])
    
user_item_matrix = pd.read_csv("src\\data\\CF\\user_item_matrix.csv").drop(columns=['Unnamed: 0'])

#1000 users by 128 interest embedding
true_interest_model = pd.read_csv('src\\data\\CF\\interest_models.csv').drop(columns=['Unnamed: 0'])


user_correlation_matrix = pd.read_csv("src\\data\\user_space\\correlation_matrix.csv").drop(columns=['Unnamed: 0']).to_numpy()
np.fill_diagonal(user_correlation_matrix, 0)


B = user_correlation_matrix
M = 5


print('starting training')
trained_h = train_weights(user_item_matrix, B, M, true_interest_model, partisan_labels, val_dataset, encoder, polarity_free_decoder, encoder_output_dim=128, k = 5, f = 7, lr = 0.001, epochs = 100)

pd.DataFrame(trained_h).to_csv('src\\data\\CF\\trained_h.csv')