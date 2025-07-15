import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import math
import os
import sys
# Visualize sparsity trends and zero-user counts
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
import copy

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from my_work.training import Encoder, Decoder
from my_work.custom_article_embedding_dataset import CustomArticleEmbeddingDataset as CD

from collections import defaultdict

#################


def lr_schedule(step):

    warmup_steps = 8  # Number of steps to warm up
    total_steps = epochs  # Total training steps
    decay_rate = 0.98    # Exponential decay rate
    if step < warmup_steps:
        return step / warmup_steps  # Linear warmup
    else:
        return decay_rate ** ((step - warmup_steps) / (total_steps - warmup_steps))  # Exponential decay


def altered_normalized_bottom_k_with_bias(Bi, k, selection_count, index_set, alpha=0.0):
    """
    Probabilistically adjust selection to promote fuller coverage of users.
    
    Args:
    - Bi: Correlation matrix.
    - k: Number of users to select per user.
    - alpha: Bias adjustment factor (0.0 = no bias, 1.0 = full bias to less-selected users).
    
    Returns:
    - norm_B: Normalized matrix with retained bottom-k values.
    """
    # global selection_count, index_set  # Track selection frequency globally
    
    norm_B = np.zeros_like(Bi, dtype=np.float32)
    
    Bi = -Bi

    for u in range(Bi.shape[0]):
        row = Bi[u]

        # Filter out NaNs and negative values
        valid_mask = ~np.isnan(row)
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
            adjusted_scores[idx] += alpha * (1 / (1 + selection_count[orig_idx]))

        # Get indices of the bottom-k adjusted values
        retain_ind = np.argsort(-adjusted_scores)[:k]
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
        b[original_indices[retain_ind]] = retain_val
        b = b / s  # Normalize to sum to 1

        norm_B[u] = b

    return norm_B

def normalized_top_k_with_bias(Bi, k, alpha=0.0):
    """
    Probabilistically adjust selection to promote fuller coverage of users.
    
    Args:
    - Bi: Correlation matrix.
    - k: Number of users to select per user.
    - alpha: Bias adjustment factor (0.0 = no bias, 1.0 = full bias to less-selected users).
    
    Returns:
    - norm_B: Normalized matrix with retained bottom-k values.
    """
    # global selection_count  # Track selection frequency globally
    norm_B = np.zeros_like(Bi, dtype=np.float32)
    
    selection_count = defaultdict(int)
    index_set = set()
    
    ## comment out the flip
    # Bi = 1 - abs(Bi)

    for u in range(Bi.shape[0]):
        row = Bi[u]

        # Filter out NaNs and negative values
        valid_mask = ~np.isnan(row)
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

        # Get indices of the top-k adjusted values
        retain_ind = np.argsort(-adjusted_scores)[:k]
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
        b[original_indices[retain_ind]] = retain_val
        b = b / s  # Normalize to sum to 1

        norm_B[u] = b

    return norm_B

def furthest_neighbours_construct_convolutions(closest, furthest, f):
    """
    Construct graph convolution tensors, log sparsity trends, and check for users with no non-zero values.
    """
    # Convert NumPy array to a PyTorch tensor
    Bi_torch = torch.tensor(copy.deepcopy(furthest), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    B_nearest = torch.tensor(copy.deepcopy(closest), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

    num_rows, num_cols = Bi_torch.shape
    # Initialize a PyTorch tensor to store the results
    tensor = torch.zeros((f, num_rows, num_cols), device=Bi_torch.device, dtype=torch.float32)

    # Set the first layer to be the furthest_neighbor correlation
    tensor[0] = Bi_torch

    # List to store sparsity percentages and zero-user counts for each layer
    sparsity_log = []
    zero_user_counts = []

    for i in range(1, f):
        # Perform batched matrix multiplication across the third dimension
        tensor[i] = torch.matmul(tensor[i-1].float(), B_nearest.float())
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

def construct_convolutions(Bi, f):
    # Convert NumPy array to a PyTorch tensor
    Bi_torch = torch.tensor(Bi, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

    num_rows, num_cols = Bi_torch.shape
    # Initialize a PyTorch tensor to store the results
    tensor = torch.zeros((f, num_rows, num_cols), device=Bi_torch.device, dtype=torch.float32)

    # Set the first layer to Bi
    tensor[0] = Bi_torch
    for i in range(1, f):
        # Perform batched matrix multiplication across the third dimension
        tensor[i] = torch.matmul(tensor[i-1], Bi_torch)
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
  x_shifted = torch.stack([torch.matmul(x_i, Bs[k]) for k in range(len(h))])  # shape: (k, U)

  weighted_sum = torch.matmul(h.T, x_shifted)  # shape: (1, U)

  return weighted_sum.flatten()

def multi_weighted_graph_convolution(x_i, Bs, h):

  """
    Compute the convolution x_i Bs h.

    Parameters:
    - x_i: A 1 x U rating vector (numpy array).
    - Bs: A k x U x U tensor representing the graph shifts.
    - h: A 1 x k vector of weights.

    Returns:
    - A 1 x U shifted and weighted rating vector.
  """
  
  x_shifted = torch.stack([torch.matmul(x_i.T, Bs[k]) for k in range(len(h))])

  weighted_sum = torch.tensordot(h.T, x_shifted, dims=([1], [0])).squeeze(axis=0)

  return weighted_sum

#need to - 
#1. get the top M rated items and their ratings
#2. extract their item_ids (columns of user-item matrix)
#3. using validation_partisan_labels, find the absolute index of the row whose item_id equals the id of the top M items
#4. find their embeddings in one of the 4 pt files (use mod)
#5. weighted sum them


## a few things - need to see where the gradient is falling off (might be fine) AND need to pass into this user vectors, right now we're being passed item rating vectors
def get_predicted_embedding(x_hat, M, encoder_output_dim, partisan_labels, val_data, encoder, polarity_free_decoder, item_list):
   

    combined_embedding = torch.zeros(encoder_output_dim, dtype=torch.float32, device=x_hat.device)

    # HERE
    valid_mask = ~torch.isnan(x_hat) & (x_hat != 0)
    filtered_row = x_hat[valid_mask]
    
    if len(filtered_row) < M:
        M = len(filtered_row)
        
        if M == 0:
            print(f'user has no viable users')
            raise ValueError("No viable users")

    # Get indices of the top-M values
    top_values, retain_ind = torch.topk(filtered_row, M)

    # Create a new row with only the top-M values retained
    b = torch.zeros_like(x_hat)
    original_indices = torch.nonzero(valid_mask, as_tuple=True)[0][retain_ind]
    b[original_indices] = top_values

    item_ids = item_list[original_indices]  # Assuming item_list is not a tensor

    
    total_score = 0
    for i in range(len(item_ids)):
        
        id = item_ids[i]
        
        ##ensured that item ordering is mapped directlyto embeddings in this case
        raw_index = int(id)
        
        point = val_data.__getitem__(raw_index)
        
        title = point[0].to(x_hat.device)
        text = point[1].to(x_hat.device)
        
        x2, _ = encoder(torch.cat((title.unsqueeze(0), text.unsqueeze(0)), dim=-1))
        polarity_free_rep = polarity_free_decoder(x2)
        
        aggregated_score = b[original_indices[i]]
        total_score += aggregated_score
        combined_embedding += aggregated_score * polarity_free_rep[0]
            
    return combined_embedding / total_score

def process_data(data):
    data = data.strip("[]")
    elements = data.split()
    numbers = list(map(float, elements))
    
    return numbers

def train_weights_per_user(matrix, rat_targets, train_dataset, most_corr,  B, M, true_interest_model, labels_and_topics, encoder, polarity_free_decoder, encoder_output_dim = 128, k=10, f=3, lr=0.01, epochs=10):
    
    rat_targets = torch.tensor(rat_targets, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    weights_per_user = np.zeros((matrix.shape[0], f))
    matrix = torch.tensor(copy.deepcopy(matrix.to_numpy()), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    item_list = torch.arange(matrix.shape[1], device=matrix.device)
    selection_count = defaultdict(int)
    index_set = set()
    
    #FN - embedding - CPC - 1
    B_furthest = altered_normalized_bottom_k_with_bias(copy.deepcopy(B), k, selection_count, index_set, alpha=0.0)
    
    # selection_count = defaultdict(int)
    # index_set = set() 
    B_nearest = normalized_top_k_with_bias(copy.deepcopy(B), k, alpha=0.0)
    
    B_i = furthest_neighbours_construct_convolutions(B_nearest, B_furthest, f)

    unviable_users = 0
    
    processed = true_interest_model['interest model'].apply(process_data)

    # Step 2: Convert to a 2D list
    data_list = processed.tolist()  # list of flat float lists

    # Step 3: Convert to a PyTorch tensor
    true_interest_model = torch.tensor(data_list, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    
    for user_id in range(matrix.shape[0]):
        
        print(f'Training for user {user_id}')
        
        h = torch.nn.Parameter(torch.rand(f, 1, requires_grad=True, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu'))
        optimizer = optim.SGD([h], lr=lr)
        
        # Scheduler
        # scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
        
        total_loss = 0

        broken_out = False
        
        ratings_target = rat_targets[user_id]
        rating_indices = torch.where(ratings_target > 0)[0]

        for epoch in range(epochs):
            
            ##remake rating matrix w.r.t h

            x = matrix

            ##rating_matrix - 1000 by 4000
            ## FN - embedding
            x_hat = multi_weighted_graph_convolution(x, B_i, h).T
            
            
            if torch.isnan(x_hat).any():
                print("Found NaN in x_hat")
            if torch.isnan(x).any():
                print("Found NaN in x")
            
            #print(f"rating_matrix shape: {rating_matrix.shape}")
            
            u_hat = x_hat[user_id]
                      
            ##print('Predicting embedding...')
            
            try:
                target = ratings_target[rating_indices]
            
            except ValueError as e:
                print("Error : ", e)
                unviable_users+=1
                broken_out = True
                break
                    
            loss = torch.mean((target - u_hat[rating_indices]) ** 2)
            
            # print(loss.item())
            loss.backward(retain_graph=True)
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            
        print(f"Average Epoch {epoch + 1}/{epochs}, Loss: {total_loss / epochs}")

        
        if broken_out:
            weights_per_user[user_id] = np.zeros(f)
            pd.DataFrame(weights_per_user).to_csv(f'src\\data\\baseline_data\\CF\\per_user\\{k}_FN_rating_rating_h_{f}_per_user.csv')

        else:
            weights_per_user[user_id] = h.cpu().detach().numpy().flatten()
            pd.DataFrame(weights_per_user).to_csv(f'src\\data\\baseline_data\\CF\\per_user\\{k}_FN_rating_rating_h_{f}_per_user.csv')
        
        print('Unviable users: ', unviable_users)
    return

if __name__ == '__main__':

    bert_dim = 768  # Example BERT embedding size
    intermediate_dim = 256
    encoder_output_dim = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)

    encoder = Encoder(bert_dim, intermediate_dim, encoder_output_dim).to(device)
    polarity_free_decoder = Decoder(encoder_output_dim, intermediate_dim, encoder_output_dim).to(device)

    encoder.load_state_dict(torch.load('src\my_work\models\encoder.pt', weights_only=True))
    polarity_free_decoder.load_state_dict(torch.load('src\my_work\models\polarity_free_decoder.pt', weights_only=True))

    encoder.eval()
    polarity_free_decoder.eval()


    source_path = "D:\Bert-Embeddings\\test_data\\"

    labels_file = "src\data\\baseline_data\CF\per_user\\testing_partisan_labels.csv"

    # text_embedding_file = torch.load(source_path + f"text_embedding_{0}.pt")
    # title_embedding_file = torch.load(source_path + f"title_embedding_{0}.pt")

    # for i in range(1,4):
    #     text_embedding_file = torch.cat((text_embedding_file, torch.load(source_path + f"text_embedding_{i}.pt")), dim=0)
    #     title_embedding_file = torch.cat((title_embedding_file, torch.load(source_path + f"title_embedding_{i}.pt")), dim=0)
        
    
    text_paths = []
    title_paths = []
    for i in range(4):
        text_paths.append(source_path + f"text_embedding_{i}.pt")
        title_paths.append(source_path + f"title_embedding_{i}.pt")
        
    
    train_dataset = CD(labels_file, text_paths, title_paths, [0, 4000])
    
    labels_and_topics = pd.read_csv('src\data\\baseline_data\\baseline_testing_data.csv', skipinitialspace=True, usecols=['article_id', 'topical_vector', 'source_partisan_score'])
        
    user_item_matrix = pd.read_csv("src\\data\\CF_test_correlation\\user_item_matrix.csv").drop(columns=['Unnamed: 0'])
    
    holdouts = pd.read_csv("src\\data\\CF_test_correlation\\holdouts.csv").drop(columns=['Unnamed: 0'])
    
    # rat_targets = np.sum([np.array(holdouts.values), np.array(user_item_matrix.values)], axis=0)
    rat_targets = np.array(holdouts.values)

    #1000 users by 128 interest embedding
    true_interest_model = pd.read_csv('src\\data\\baseline_data\\CF\\interest_models.csv').drop(columns=['Unnamed: 0'])

    user_correlation_matrix = pd.read_csv("src\\data\\baseline_data\\CF\\rating_correlation_matrix.csv").drop(columns=['Unnamed: 0']).to_numpy()    

    # Define the learning rate schedule
    ##hacky
    
    np.fill_diagonal(user_correlation_matrix, 0)
    
    most_corr = copy.deepcopy(user_correlation_matrix)
    B = user_correlation_matrix
    M = 10
    f = 5
    epochs = 40
    ##each user has 10 logged interactions

    ##change to using labels_and_topics
    print('starting training')
    trained_h = train_weights_per_user(user_item_matrix, rat_targets, train_dataset, most_corr, B, M, true_interest_model, labels_and_topics, encoder, polarity_free_decoder, encoder_output_dim=128, k = 8, f = f, lr = 1, epochs = epochs)
