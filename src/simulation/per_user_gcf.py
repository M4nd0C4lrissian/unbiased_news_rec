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

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from my_work.training import Encoder, Decoder
from my_work.custom_article_embedding_dataset import CustomArticleEmbeddingDataset as CD

from collections import defaultdict

selection_count = defaultdict(int)
index_set = set()

def normalized_bottom_k_with_bias(Bi, k, alpha=0.0):
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
    
    Bi = 1 - abs(Bi)

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

def construct_convolutions(Bi, f):
    # Convert NumPy array to a PyTorch tensor
    Bi_torch = torch.tensor(Bi, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')

    num_rows, num_cols = Bi_torch.shape
    # Initialize a PyTorch tensor to store the results
    tensor = torch.zeros((f, num_rows, num_cols), device=Bi_torch.device, dtype=torch.float64)

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

#need to - 
#1. get the top M rated items and their ratings
#2. extract their item_ids (columns of user-item matrix)
#3. using validation_partisan_labels, find the absolute index of the row whose item_id equals the id of the top M items
#4. find their embeddings in one of the 4 pt files (use mod)
#5. weighted sum them


## a few things - need to see where the gradient is falling off (might be fine) AND need to pass into this user vectors, right now we're being passed item rating vectors
def get_predicted_embedding(x_hat, M, user_item_matrix, encoder_output_dim, partisan_labels, val_data, encoder, polarity_free_decoder, batch_size = 1000):
    item_list = user_item_matrix.columns

    combined_embedding = torch.zeros(encoder_output_dim, dtype=torch.float64, device=x_hat.device)

    # HERE
    valid_mask = ~torch.isnan(x_hat) & (x_hat != 0)
    filtered_row = x_hat[valid_mask]
    
    if len(filtered_row) < M:
        M = len(filtered_row)
        
        if M == 0:
            print(f'user has no viable users')
            return combined_embedding

    # Get indices of the top-M values
    top_values, retain_ind = torch.topk(filtered_row, M)

    # Create a new row with only the top-M values retained
    b = torch.zeros_like(x_hat)
    original_indices = torch.nonzero(valid_mask, as_tuple=True)[0][retain_ind]
    b[original_indices] = top_values

    item_ids = item_list[original_indices.cpu().numpy()]  # Assuming item_list is not a tensor

    
    total_score = 0
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
        total_score += aggregated_score
        combined_embedding += aggregated_score * polarity_free_rep[0]
            
    return combined_embedding / total_score

def process_data(data):
    data = data.strip("[]")
    elements = data.split()
    numbers = list(map(float, elements))
    
    return numbers

def train_weights_per_user(matrix, B, M, true_interest_model, partisan_labels, val_data, encoder, polarity_free_decoder, encoder_output_dim = 128, k=10, f=3, lr=0.01, epochs=10):
    
    # s = torch.nn.ReLU()
    
    weights_per_user = pd.read_csv(f'src\\data\\CF\\per_user\\trained_h_{f}_per_user.csv').drop(columns=['Unnamed: 0']).to_numpy()
    # weights_per_user = np.zeros((matrix.shape[0], f))

    Bi = normalized_bottom_k_with_bias(B, k)
    B_i = construct_convolutions_with_user_check(Bi, f)
    ##print(B_i)
    
    item_list = matrix.columns
    
    for user_id in range(280, matrix.shape[0]):
        
        print(f'Training for user {user_id}')
        
        h = torch.nn.Parameter(torch.rand(f, 1, requires_grad=True, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu'))
        optimizer = optim.SGD([h], lr=lr)
        
        # Scheduler
        scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
        
        total_loss = 0
        

        for epoch in range(epochs):
            
            rating_matrix = torch.empty(1000, 0, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
            ##remake rating matrix w.r.t h
            for item_id in range(matrix.shape[1]):

            # Get rating vector for the item

                x_i = matrix.iloc[:][str(item_list[item_id])].values
                x_i = torch.tensor(x_i, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')

                x_hat = weighted_graph_convolution(x_i, B_i, h)
                if torch.isnan(x_hat).any():
                    print("Found NaN in x_hat")
                if torch.isnan(x_i).any():
                    print("Found NaN in x_i")
                
                rating_matrix = torch.cat((rating_matrix, x_hat.unsqueeze(0).T), dim = -1)
            
            ##print(f"rating_matrix shape: {rating_matrix.shape}")
            
            u_hat = rating_matrix[user_id]
        
            ##print('Predicting embedding...')
            predicted_user_embedding = get_predicted_embedding(u_hat, M, matrix, encoder_output_dim, partisan_labels, val_data, encoder, polarity_free_decoder)
            actual_user_embedding = torch.tensor(process_data(true_interest_model.iloc[user_id]['interest model']), dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            
            loss = torch.mean((actual_user_embedding - predicted_user_embedding) ** 2)
            
            if not math.isnan(loss.item()):
                total_loss += loss.item()
                print(loss.item())
              
            else:
                print('Problem!')

            loss.backward(retain_graph=True)
            ##print(f'Gradient: {h.grad}')
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        print(f"Average Epoch {epoch + 1}/{epochs}, Loss: {total_loss / epochs}")


        weights_per_user[user_id] = h.cpu().detach().numpy().flatten()
        pd.DataFrame(weights_per_user).to_csv(f'src\\data\\CF\\per_user\\trained_h_{f}_per_user.csv')
        
        
    return weights_per_user

if __name__ == '__main__':

    bert_dim = 768  # Example BERT embedding size
    intermediate_dim = 256
    encoder_output_dim = 128

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    
    

    # Define the learning rate schedule


    ##hacky
    B = user_correlation_matrix
    M = 10
    f = 5
    epochs = 40
    ##each user has 10 logged interactions

    def lr_schedule(step):
    
        warmup_steps = 8  # Number of steps to warm up
        total_steps = epochs  # Total training steps
        decay_rate = 0.98    # Exponential decay rate
        if step < warmup_steps:
            return step / warmup_steps  # Linear warmup
        else:
            return decay_rate ** ((step - warmup_steps) / (total_steps - warmup_steps))  # Exponential decay


    print('starting training')
    trained_h = train_weights_per_user(user_item_matrix, B, M, true_interest_model, partisan_labels, val_dataset, encoder, polarity_free_decoder, encoder_output_dim=128, k = 30, f = f, lr = 0.05, epochs = epochs)

    pd.DataFrame(trained_h).to_csv(f'src\\data\\CF\\per_user\\trained_h_{f}_per_user.csv')