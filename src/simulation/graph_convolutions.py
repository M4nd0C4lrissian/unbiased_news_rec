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
# Visualize sparsity trends and zero-user counts
import matplotlib.pyplot as plt

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from my_work.training import Encoder, Decoder
from my_work.custom_article_embedding_dataset import CustomArticleEmbeddingDataset as CD


##issue here - losing certain user's ratings
## need to re-confirm that this works (though I'm pretty sure I tested it before)
#change to bottom k

from collections import defaultdict

# Track the selection frequency of each user
selection_count = defaultdict(int)
index_set = set()

def normalized_bottom_k_with_bias(Bi, k, alpha=0.375):
    """
    Probabilistically adjust selection to promote fuller coverage of users.
    
    Args:
    - Bi: Correlation matrix.
    - k: Number of users to select per user.
    - alpha: Bias adjustment factor (0.0 = no bias, 1.0 = full bias to less-selected users).
    
    Returns:
    - norm_B: Normalized matrix with retained bottom-k values.
    """
    global selection_count  # Track selection frequency globally
    norm_B = np.zeros_like(Bi, dtype=np.float64)

    for u in range(Bi.shape[0]):
        row = Bi[u]

        # Filter out NaNs and negative values
        valid_mask = ~np.isnan(row) & (row > 0)
        filtered_row = row[valid_mask]

        if len(filtered_row) == 0:
            norm_B[u] = np.zeros_like(row)
            print(f'user: {u} has no viable users')
            continue

        # Adjust scores to include selection bias
        original_indices = np.where(valid_mask)[0]
        adjusted_scores = filtered_row.copy()

        for idx, orig_idx in enumerate(original_indices):
            # Adjust scores based on selection count
            adjusted_scores[idx] -= alpha * (1 / (1 + selection_count[orig_idx]))

        # Get indices of the bottom-k adjusted values
        retain_ind = np.argsort(adjusted_scores)[:k]
        retain_val = filtered_row[retain_ind]

        # Update the global selection count
        for ind in original_indices[retain_ind]:
            selection_count[ind] += 1
            index_set.add(ind)

        # Normalize retained values
        s = np.sum(retain_val)
        if s == 0:
            norm_B[u] = np.zeros_like(row)
            print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')
            continue

        # Create a new row with only the selected values retained
        b = np.zeros_like(row)
        b[original_indices[retain_ind]] = 1 - retain_val
        b = b / s  # Normalize to sum to 1

        norm_B[u] = b

    return norm_B

def construct_convolutions_with_user_check(Bi, f):
    """
    Construct graph convolution tensors, log sparsity trends, and check for users with no non-zero values.
    """
    # Convert NumPy array to a PyTorch tensor
    Bi_torch = torch.tensor(Bi, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')

    num_rows, num_cols = Bi_torch.shape
    # Initialize a PyTorch tensor to store the results
    tensor = torch.zeros((f, num_rows, num_cols), device=Bi_torch.device, dtype=torch.float64)

    # Set the first layer to Bi
    tensor[0] = Bi_torch

    # List to store sparsity percentages and zero-user counts for each layer
    sparsity_log = []
    zero_user_counts = []

    for i in range(1, f):
        # Perform batched matrix multiplication across the third dimension
        tensor[i] = torch.matmul(tensor[i-1].float(), Bi_torch.float())
        tensor[i] = torch.nan_to_num(tensor[i], nan=0.0)

        # Calculate sparsity: proportion of non-zero entries
        non_zero_count = torch.count_nonzero(tensor[i])
        total_elements = tensor[i].numel()
        sparsity = 100.0 * (non_zero_count / total_elements)
        sparsity_log.append(sparsity.item())

        # Check for rows (users) with all-zero values
        zero_users = torch.sum(torch.all(tensor[i] == 0, dim=1)).item()
        zero_user_counts.append(zero_users)

        print(f"Layer {i}: Sparsity = {sparsity:.2f}%, Zero Users = {zero_users}/{num_rows}")


    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, f), sparsity_log, marker='o', label="Sparsity")
    # plt.xlabel("Layer")
    # plt.ylabel("Sparsity (%)")
    # plt.title("Sparsity Trends Across Layers")
    # plt.grid(True)
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, f), zero_user_counts, marker='o', color='red', label="Zero Users")
    # plt.xlabel("Layer")
    # plt.ylabel("Number of Zero-User Rows")
    # plt.title("Zero Users Across Layers")
    # plt.grid(True)
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

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
  x_shifted = torch.stack([torch.matmul(x_i, Bs[k]) for k in range(len(h))])  # shape: (k, U)

  weighted_sum = torch.matmul(h.T, x_shifted)  # shape: (1, U)

  return weighted_sum.flatten()

#need to - 
#1. get the top M rated items and their ratings
#2. extract their item_ids (columns of user-item matrix)
#3. using validation_partisan_labels, find the absolute index of the row whose item_id equals the id of the top M items
#4. find their embeddings in one of the 4 pt files (use mod)
#5. weighted sum them


## a few things - need to see where the gradient is falling off (might be fine) AND need to pass into this user vectors, right now we're being passed item rating vectors
def get_predicted_embedding(x_hat, M, user_item_matrix, encoder_output_dim, partisan_labels, val_data, encoder, polarity_free_decoder):
    item_list = user_item_matrix.columns

    # Mask invalid entries
    valid_mask = ~torch.isnan(x_hat) & (x_hat > 0)
    filtered_row = x_hat[valid_mask]

    # Get indices of the top-M values
    top_values, retain_ind = torch.topk(filtered_row, M)

    # Create a new row with only the top-M values retained
    b = torch.zeros_like(x_hat)
    original_indices = torch.nonzero(valid_mask, as_tuple=True)[0][retain_ind]
    b[original_indices] = top_values

    item_ids = item_list[original_indices.cpu().numpy()]  # Assuming item_list is not a tensor
    combined_embedding = torch.zeros(encoder_output_dim, dtype=torch.float64, device=x_hat.device)
    
    
    for i in range(len(item_ids)):
        
        id = item_ids[i]
        
        ##should be precisely one
        raw_index = int(partisan_labels.loc[partisan_labels['article_id'] == int(id)].index[0])
        
        ## let's assume we're concatenating all 4000 for now   
        # batch_num = raw_index // 1000
        
        # local_index = raw_index % 1000
        
        point = val_data.__getitem__(raw_index)
        
        title = point[0].to(x_hat.device)
        text = point[1].to(x_hat.device)
        
        x2, _ = encoder(torch.cat((title.unsqueeze(0), text.unsqueeze(0)), dim=-1))
        polarity_free_rep = polarity_free_decoder(x2)
        
        aggregated_score = b[original_indices[i]]
        combined_embedding += aggregated_score * polarity_free_rep[0]
            
    return combined_embedding

def get_predicted_embedding_batched(x_hat_batch, M, user_item_matrix, encoder_output_dim, partisan_labels, val_data, encoder, polarity_free_decoder, device='cuda', batch_size=100):
    item_list = user_item_matrix.columns
    batch_size_users = x_hat_batch.size(0)

    # Mask invalid entries
    valid_mask_batch = ~torch.isnan(x_hat_batch) & (x_hat_batch > 0)
    combined_embeddings = torch.zeros(batch_size_users, encoder_output_dim, dtype=torch.float64, device=device)

    all_items, all_scores = [], []

    # Collect top-M items for each user in the batch
    for i in range(batch_size_users):
        x_hat = x_hat_batch[i]
        valid_mask = valid_mask_batch[i]

        # Get top-M values and their indices
        filtered_row = x_hat[valid_mask]
        if len(filtered_row) < M:
            print(f"too few for user {i}, only {len(filtered_row)} found but need {M}")
            continue

        top_values, retain_ind = torch.topk(filtered_row, min(M, len(filtered_row)))
        original_indices = torch.nonzero(valid_mask, as_tuple=True)[0][retain_ind]

        item_ids = item_list[original_indices.cpu().numpy()]
        all_items.extend(item_ids)
        all_scores.extend(top_values.tolist())

    # Batch process items through the encoder and decoder
    embeddings = []
    # for batch_start in range(0, len(all_items), batch_size):
    #     batch_end = min(batch_start + batch_size, len(all_items))
    #     batch_items = all_items[batch_start:batch_end]
    
    ##loading data
    titles, texts, scores = [], [], []
    for i in range(len(all_items)):
        item_id = all_items[i]
        
        raw_index = int(partisan_labels.loc[partisan_labels["article_id"] == int(item_id)].index[0])
        point = val_data.__getitem__(raw_index)
        titles.append(point[0].to(device))
        texts.append(point[1].to(device))

    titles = torch.stack(titles)
    texts = torch.stack(texts)
    scores = torch.tensor(all_scores, dtype=torch.float64, device=device)

    # Encoder and decoder predictions
    x2, _ = encoder(torch.cat((titles, texts), dim=-1))
    polarity_free_reps = polarity_free_decoder(x2)
    
    # embeddings.append((polarity_free_reps * scores).sum(dim=0))
    
    for i in range(len(scores)):
        embeddings.append((scores[i] * polarity_free_reps[i]))

    embeddings = torch.cat(embeddings, dim=0)

    # Aggregate embeddings for each user in the batch
    offset = 0
    for i in range(batch_size_users):
        x_hat = x_hat_batch[i]
        valid_mask = valid_mask_batch[i]
        filtered_row = x_hat[valid_mask]
        num_items = min(M, len(filtered_row))

        total_score = scores[offset: offset + num_items].sum(dim=0)
        combined_embeddings[i] = embeddings[offset:offset + num_items].sum(dim=0) / total_score
        
        offset += num_items

    return combined_embeddings

def process_data(data):
    data = data.strip("[]")
    elements = data.split()
    numbers = list(map(float, elements))
    
    return numbers

def __train_weights(matrix, B, M, true_interest_model, partisan_labels, val_data, encoder, polarity_free_decoder, encoder_output_dim = 128, k=10, f=3, lr=0.0001, epochs=100, batch_size = 100):
    
    h = torch.nn.Parameter(torch.rand(f, 1, requires_grad=True, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu'))

    optimizer = optim.Adam([h], lr=lr)
    
    Bi = normalized_bottom_k_with_bias(B, k)
    B_i = construct_convolutions_with_user_check(Bi, f)
    print(B_i)
    
    item_list = matrix.columns

    for epoch in range(epochs):
        total_loss = 0
        rating_matrix = torch.empty(1000, 0, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
        for item_id in range(matrix.shape[1]):

            # Get rating vector for the item

            x_i = matrix.iloc[:][str(item_list[item_id])].values
            #print(x_i)
            x_i = torch.tensor(x_i, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
            #print(x_i)


            x_hat = weighted_graph_convolution(x_i, B_i, h)
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
        
        for batch_start in range(0, matrix.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, matrix.shape[0])
            user_batch = list(range(batch_start, batch_end))

            u_hat_batch = rating_matrix[user_batch, :]
            actual_embeddings = torch.tensor(
                [process_data(true_interest_model.iloc[u]["interest model"]) for u in user_batch],
                dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu'
            )

            # Get predicted embeddings for the batch
            predicted_embeddings = get_predicted_embedding_batched(
                u_hat_batch, M, matrix, encoder_output_dim, partisan_labels, val_data, encoder, polarity_free_decoder, batch_size=batch_size
            )

            # Compute loss for the batch
            loss = torch.mean((actual_embeddings - predicted_embeddings) ** 2)

            # Update weights
            if not torch.isnan(loss).any():
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                print("NaN encountered in loss computation")
                
            print(f'Batch {batch_start} completed, Loss: {loss.item()}')

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")

    return h.detach().numpy()



def train_weights(matrix, B, M, true_interest_model, partisan_labels, val_data, encoder, polarity_free_decoder, encoder_output_dim = 128, k=10, f=3, lr=0.01, epochs=100, batch_size = 100):
    
    h = torch.nn.Parameter(torch.rand(f, 1, requires_grad=True, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu'))

    optimizer = optim.Adam([h], lr=lr)
    
    Bi = normalized_bottom_k_with_bias(B, k)
    B_i = construct_convolutions_with_user_check(Bi, f)
    print(B_i)
    
    item_list = matrix.columns
    
    loss_over_time = []

    for epoch in range(epochs):
        total_loss = 0
        rating_matrix = torch.empty(1000, 0, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
        for item_id in range(matrix.shape[1]):

            # Get rating vector for the item

            x_i = matrix.iloc[:][str(item_list[item_id])].values
            #print(x_i)
            x_i = torch.tensor(x_i, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
            #print(x_i)


            x_hat = weighted_graph_convolution(x_i, B_i, h)
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
        
        for u in range(rating_matrix.size()[0]):
            
            print(f'Computing for user {u}')
            
            u_hat = rating_matrix[u]
            
            valid_mask = ~torch.isnan(u_hat) & (u_hat > 0)
            filtered_row = u_hat[valid_mask]
            if len(filtered_row) < M:
                print(f"too few for user {u}, only {len(filtered_row)} found but need {M}")
                continue
                
        
            ##print('Predicting embedding...')
            predicted_user_embedding = get_predicted_embedding(u_hat, M, matrix, encoder_output_dim, partisan_labels, val_data, encoder, polarity_free_decoder)
            actual_user_embedding = torch.tensor(process_data(true_interest_model.iloc[u]['interest model']), dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            
            loss = torch.mean((actual_user_embedding - predicted_user_embedding) ** 2)
            ##print('Loss calculated!')
            
            # loss = torch.mean((x_i[mask] - x_hat[mask]) ** 2)
            #print(loss.item())
            
            if not math.isnan(loss.item()):
                total_loss += loss.item()
              
            else:
                print('Problem!')

            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"User {u} completed, Loss: {loss.item()}")
            loss_over_time.append(loss.item())

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")
        pd.DataFrame(h.cpu().detach().numpy()).to_csv(f'src\\data\\CF\\trained_h_{f}_epoch_{epoch}.csv')


    return h.cpu().detach().numpy(), loss_over_time


if __name__ == '__main__':

    bert_dim = 768  # Example BERT embedding size
    intermediate_dim = 256
    encoder_output_dim = 128


    device='cuda' if torch.cuda.is_available() else 'cpu'
    encoder = Encoder(bert_dim, intermediate_dim, encoder_output_dim).to(device)
    polarity_free_decoder = Decoder(encoder_output_dim, intermediate_dim, encoder_output_dim).to(device)

    encoder.load_state_dict(torch.load('src\my_work\models\encoder.pt', weights_only=True))
    polarity_free_decoder.load_state_dict(torch.load('src\my_work\models\polarity_free_decoder.pt', weights_only=True))

    encoder.eval()
    polarity_free_decoder.eval()


    val_path = "D:\Bert-Embeddings\\validation_data\\"

    labels_file = "src\data\\auto_encoder_training\\validation_data\\validation_partisan_labels.csv"

    text_paths = []
    title_paths = []
    for i in range(1):
        text_paths.append(val_path + f"text_embedding_{i}.pt")
        title_paths.append(val_path + f"title_embedding_{i}.pt")
        

    val_dataset = CD(labels_file, text_paths, title_paths, [0, 1000])
        
    partisan_labels = pd.read_csv("src\data\\auto_encoder_training\\validation_data\\validation_partisan_labels.csv").drop(columns=['Unnamed: 0'])
        
    user_item_matrix = pd.read_csv("src\\data\\CF\\user_item_matrix.csv").drop(columns=['Unnamed: 0'])

    #1000 users by 128 interest embedding
    true_interest_model = pd.read_csv('src\\data\\CF\\interest_models.csv').drop(columns=['Unnamed: 0'])


    user_correlation_matrix = pd.read_csv("src\\data\\user_space\\correlation_matrix.csv").drop(columns=['Unnamed: 0']).to_numpy()

    np.fill_diagonal(user_correlation_matrix, 0)

    ##hacky
    B = user_correlation_matrix / 1000
    M = 5

    print(B)

    # user_item_matrix = user_item_matrix
    f = 5
    print('starting training')
    k = 100
    lr = 0.001
    epochs = 25

    trained_h, loss_over_time = train_weights(user_item_matrix, B, M, true_interest_model, partisan_labels, val_dataset, encoder, polarity_free_decoder, encoder_output_dim=128, k = k, f = f, lr = lr, epochs = epochs)

    pd.DataFrame(trained_h).to_csv(f'src\\data\\CF\\trained_h_{f}_final.csv')

    plt.scatter(range(len(loss_over_time)), loss_over_time)
    plt.show()
    plt.savefig(f'src\\data\\CF\\trained_h_{f}_loss_over_time.png')

    pd.DataFrame(loss_over_time).to_csv(f'src\\data\\CF\\trained_h_{f}_loss_over_time.csv')