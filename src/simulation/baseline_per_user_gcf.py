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
    # global selection_count  # Track selection frequency globally
    
    selection_count = defaultdict(int)
    index_set = set()
    
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

#################

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
    
    norm_B = np.zeros_like(Bi, dtype=np.float64)
    
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
    norm_B = np.zeros_like(Bi, dtype=np.float64)
    
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

def construct_convolutions_with_user_check(Bi, f):
    """
    Construct graph convolution tensors, log sparsity trends, and check for users with no non-zero values.
    """
    # Convert NumPy array to a PyTorch tensor
    Bi_torch = torch.tensor(copy.deepcopy(Bi), dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')

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
  I = 4000
  
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
            raise ValueError("No viable users")

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
    
    # s = torch.nn.ReLU()
    
    # weights_per_user = pd.read_csv(f'src\\data\\baseline_data\\CF\\per_user\\trained_h_{f}_per_user.csv').drop(columns=['Unnamed: 0']).to_numpy()
    
    # weights_per_user = np.pad(weights_per_user, ((0, 1000 - weights_per_user.shape[0]), (0, 0)), mode='constant', constant_values=0)
    # weights_per_user = pd.read_csv(f'src\\data\\baseline_data\\CF\\per_user\\FN_embedding_CPC_h_{f}_per_user.csv').drop(columns=['Unnamed: 0']).to_numpy()
    # weights_per_user2 = pd.read_csv(f'src\\data\\baseline_data\\CF\\per_user\\FN_rating_target_CPC_h_{f}_per_user.csv').drop(columns=['Unnamed: 0']).to_numpy()
    # weights_per_user3 = pd.read_csv(f'src\\data\\baseline_data\\CF\\per_user\\NN_rating_target_CPC_h_{f}_per_user.csv').drop(columns=['Unnamed: 0']).to_numpy()
    # weights_per_user4 = pd.read_csv(f'src\\data\\baseline_data\\CF\\per_user\\NN_embedding_CPC_h_{f}_per_user.csv').drop(columns=['Unnamed: 0']).to_numpy()
    
    # weights_per_user3 = np.zeros((matrix.shape[0], f))
    # weights_per_user2 = np.zeros((matrix.shape[0], f))
    # weights_per_user = np.zeros((matrix.shape[0], f))
    # weights_per_user4 = np.zeros((matrix.shape[0], f))
    
    selection_count = defaultdict(int)
    index_set = set()
    
    #FN - embedding - CPC - 1
    Bi = normalized_bottom_k_with_bias(copy.deepcopy(B), k)
    B_i = construct_convolutions_with_user_check(Bi, f)
    
    selection_count = defaultdict(int)
    index_set = set() 
    
    ##FN - rating - CPC - 2
    Mi = altered_normalized_bottom_k_with_bias(copy.deepcopy(B), k, selection_count, index_set, alpha=0.1)
    M_i = construct_convolutions_with_user_check(Mi, f)
    
    #NN - rating - CPC - 3
    Ni = normalized_top_k_with_bias(copy.deepcopy(most_corr), k)
    N_i = construct_convolutions_with_user_check(Ni, f)
    
    ##NN - embedding - CPC - 4
    Ei = normalized_top_k_with_bias(copy.deepcopy(most_corr), k)
    E_i = construct_convolutions_with_user_check(Ei, f)

    unviable_users = 0
    
    for user_id in range(326, matrix.shape[0]):
        
        print(f'Training for user {user_id}')
        
        h = torch.nn.Parameter(torch.rand(f, 1, requires_grad=True, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu'))
        optimizer = optim.SGD([h], lr=lr)
        
        h2 = torch.nn.Parameter(torch.rand(f, 1, requires_grad=True, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu'))
        optimizer2 = optim.Adam([h2], lr=0.5)
        
        h3 = torch.nn.Parameter(torch.rand(f, 1, requires_grad=True, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu'))
        optimizer3 = optim.Adam([h3], lr=0.5)
        
        h4 = torch.nn.Parameter(torch.rand(f, 1, requires_grad=True, dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu'))
        optimizer4 = optim.Adam([h4], lr=lr)
        
        # Scheduler
        scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
        scheduler2 = LambdaLR(optimizer2, lr_lambda=lr_schedule)
        scheduler3 = LambdaLR(optimizer3, lr_lambda=lr_schedule)
        scheduler4 = LambdaLR(optimizer4, lr_lambda=lr_schedule)
        
        total_loss = 0
        total_loss2 = 0
        total_loss3 = 0
        total_loss4 = 0
        
        ## do I know how to do this? - should the targets really just be predicting only the true ratings on the things we have ratings for - yeah I guess
        ratings_target = np.array(rat_targets[user_id])
        rating_indices = np.where(ratings_target > 0)[0]

        broken_out = False

        for epoch in range(epochs):
            
            ##remake rating matrix w.r.t h

            x = torch.tensor(copy.deepcopy(matrix.to_numpy()), dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
            x2 = torch.tensor(copy.deepcopy(matrix.to_numpy()), dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
            x3 = torch.tensor(copy.deepcopy(matrix.to_numpy()), dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
            x4 = torch.tensor(copy.deepcopy(matrix.to_numpy()), dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')

            ##rating_matrix - 1000 by 4000
            ## FN - embedding
            x_hat = multi_weighted_graph_convolution(x, B_i, h).T
            
            #FN
            x_hat_2 = multi_weighted_graph_convolution(x2, M_i, h2).T
            
            #NN
            x_hat_3 = multi_weighted_graph_convolution(x3, N_i, h3).T
            
            ## NN - embedding
            x_hat_4 = multi_weighted_graph_convolution(x4, E_i, h4).T
            
            if torch.isnan(x_hat).any():
                print("Found NaN in x_hat")
            if torch.isnan(x).any():
                print("Found NaN in x")
            
            #print(f"rating_matrix shape: {rating_matrix.shape}")
            
            u_hat = x_hat[user_id]
        
            ##print('Predicting embedding...')
            
            try:
                predicted_user_embedding = get_predicted_embedding(u_hat, M, matrix, encoder_output_dim, labels_and_topics, train_dataset, encoder, polarity_free_decoder)
            
            except ValueError as e:
                print("Error : ", e)
                unviable_users+=1
                broken_out = True
                break
            
            else:
                actual_user_embedding = torch.tensor(process_data(true_interest_model.iloc[user_id]['interest model']), dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
                
                loss = torch.mean((actual_user_embedding - predicted_user_embedding) ** 2)
                
                if not math.isnan(loss.item()):
                    total_loss += loss.item()
                    # print(loss.item())
                
                else:
                    print('Problem!')

                loss.backward(retain_graph=True)
                ##print(f'Gradient: {h.grad}')
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            #FN ########################################################
            target = torch.tensor(ratings_target[rating_indices], dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            u_hat2 = x_hat_2[user_id]
            
            loss2 = torch.mean((target - u_hat2[rating_indices]) ** 2)
            
            if not math.isnan(loss2.item()):
                total_loss2 += loss2.item()
                # print(loss2.item())
              
            else:
                print('Problem!')

            loss2.backward(retain_graph=True)
            ##print(f'Gradient: {h.grad}')
            optimizer2.step()
            scheduler2.step()
            optimizer2.zero_grad()
            
            #NN ########################################################
            target = torch.tensor(ratings_target[rating_indices], dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
            
            u_hat3 = x_hat_3[user_id]

            loss3 = torch.mean((target - u_hat3[rating_indices]) ** 2)
            
            if not math.isnan(loss3.item()):
                total_loss3 += loss3.item()
                # print(loss3.item())
              
            else:
                print('Problem!')

            loss3.backward(retain_graph=True)
            ##print(f'Gradient: {h.grad}')
            optimizer3.step()
            scheduler3.step()
            optimizer3.zero_grad()
            
            #NN - embedding #############################################################
            
            u_hat4 = x_hat_4[user_id]
            
            try:
                predicted_user_embedding = get_predicted_embedding(u_hat4, M, matrix, encoder_output_dim, labels_and_topics, train_dataset, encoder, polarity_free_decoder)
            
            except ValueError as e:
                print("Error : ", e)
                unviable_users+=1
                broken_out = True
                break
            
            else:
                actual_user_embedding = torch.tensor(process_data(true_interest_model.iloc[user_id]['interest model']), dtype=torch.float64, device='cuda' if torch.cuda.is_available() else 'cpu')
                
                loss4 = torch.mean((actual_user_embedding - predicted_user_embedding) ** 2)
                
                if not math.isnan(loss4.item()):
                    total_loss4 += loss4.item()
                    # print(loss4.item())
                
                else:
                    print('Problem!')

                loss4.backward(retain_graph=True)
                ##print(f'Gradient: {h.grad}')
                optimizer4.step()
                scheduler4.step()
                optimizer4.zero_grad()

        print(f"Average Epoch {epoch + 1}/{epochs}, Loss: {total_loss / epochs}")
        print(f"FN: {epoch + 1}/{epochs}, Loss: {total_loss2 / epochs}")
        print(f"NN: {epoch + 1}/{epochs}, Loss: {total_loss3 / epochs}")
        print(f"NN + embedding: {epoch + 1}/{epochs}, Loss: {total_loss4 / epochs}")
        
        # if broken_out:
        #     weights_per_user[user_id] = np.zeros(f)
        #     pd.DataFrame(weights_per_user).to_csv(f'src\\data\\baseline_data\\CF\\per_user\\FN_embedding_CPC_h_{f}_per_user.csv')
            
        #     weights_per_user2[user_id] = np.zeros(f)
        #     pd.DataFrame(weights_per_user2).to_csv(f'src\\data\\baseline_data\\CF\\per_user\\FN_rating_target_CPC_h_{f}_per_user.csv')
            
        #     weights_per_user3[user_id] = np.zeros(f)
        #     pd.DataFrame(weights_per_user3).to_csv(f'src\\data\\baseline_data\\CF\\per_user\\NN_rating_target_CPC_h_{f}_per_user.csv')
            
        #     weights_per_user4[user_id] = np.zeros(f)
        #     pd.DataFrame(weights_per_user4).to_csv(f'src\\data\\baseline_data\\CF\\per_user\\NN_embedding_CPC_h_{f}_per_user.csv')


        # weights_per_user[user_id] = h.cpu().detach().numpy().flatten()
        # pd.DataFrame(weights_per_user).to_csv(f'src\\data\\baseline_data\\CF\\per_user\\FN_embedding_CPC_h_{f}_per_user.csv')
        
        # weights_per_user2[user_id] = h2.cpu().detach().numpy().flatten()
        # pd.DataFrame(weights_per_user2).to_csv(f'src\\data\\baseline_data\\CF\\per_user\\FN_rating_target_CPC_h_{f}_per_user.csv')
        
        # weights_per_user3[user_id] = h3.cpu().detach().numpy().flatten()
        # pd.DataFrame(weights_per_user3).to_csv(f'src\\data\\baseline_data\\CF\\per_user\\NN_rating_target_CPC_h_{f}_per_user.csv')
         
        # weights_per_user4[user_id] = h4.cpu().detach().numpy().flatten()
        # pd.DataFrame(weights_per_user4).to_csv(f'src\\data\\baseline_data\\CF\\per_user\\NN_embedding_CPC_h_{f}_per_user.csv')
        
        print('Unviable users: ', unviable_users)
    return

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


    source_path = "D:\Bert-Embeddings\\test_data\\"

    labels_file = "src\data\\baseline_data\CF\per_user\\testing_partisan_labels.csv"

    text_embedding_file = torch.load(source_path + f"text_embedding_{0}.pt")
    title_embedding_file = torch.load(source_path + f"title_embedding_{0}.pt")

    for i in range(1,4):
        text_embedding_file = torch.cat((text_embedding_file, torch.load(source_path + f"text_embedding_{i}.pt")), dim=0)
        title_embedding_file = torch.cat((title_embedding_file, torch.load(source_path + f"title_embedding_{i}.pt")), dim=0)
        
    
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

    user_correlation_matrix = pd.read_csv("src\\data\\baseline_data\\CF\\correlation_matrix.csv").drop(columns=['Unnamed: 0']).to_numpy()    

    # Define the learning rate schedule
    ##hacky
    
    np.fill_diagonal(user_correlation_matrix, 0)
    
    most_corr = copy.deepcopy(user_correlation_matrix)
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


    ##change to using labels_and_topics
    print('starting training')
    trained_h = train_weights_per_user(user_item_matrix, rat_targets, train_dataset, most_corr, B, M, true_interest_model, labels_and_topics, encoder, polarity_free_decoder, encoder_output_dim=128, k = 30, f = f, lr = 0.1, epochs = epochs)
