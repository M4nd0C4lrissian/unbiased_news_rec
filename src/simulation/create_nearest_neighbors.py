import pickle 
import numpy as np
import pandas as pd
import os
import sys
import torch
import ast
from create_landmarks import user_interaction
from sklearn.metrics.pairwise import linear_kernel

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from my_work.training import Encoder, Decoder

def generate_user_content_embedding(topic_lists, source_path, uv, encoder_output_dim, encoder, polarity_decoder, polarity_free_decoder):        
            
    ##Load testing data next
    num_batches = 4
    batch_size = 1000
    num_pos_samples = 10
    item_num = 4000

    polarized_sum = np.zeros(encoder_output_dim, dtype = np.float64)
    free_sum = np.zeros(encoder_output_dim, dtype = np.float64)
    user_item_vector = [0] * item_num
    total_utility_score = 0

    for i in range(num_batches):

        text_embedding_file = torch.load(source_path + f"text_embedding_{i}.pt")
        title_embedding_file = torch.load(source_path + f"title_embedding_{i}.pt")

        print('Loaded embeddings!')

        with torch.no_grad():

            pos_samples = 0

            while pos_samples < num_pos_samples:
                r = np.random.randint(0, batch_size)

                acc = i * batch_size + r
 
                topic_vector = ast.literal_eval(topic_lists.iloc[acc]['topical_vector'])

                did_interact, utility_score = user_interaction(uv, topic_vector)

                if did_interact:

                    print(f'User interacted with a utility score of {utility_score}')

                    total_utility_score += utility_score

                    text = text_embedding_file[r]
                    title = title_embedding_file[r]

                    x2, _ = encoder(torch.cat((title.T, text.T), dim=-1))
                    polarity_rep = polarity_decoder(x2)
                    polarity_free_rep = polarity_free_decoder(x2)

                    print("Inference complete...")

                    polarized_sum = np.add(polarized_sum, utility_score * polarity_rep.detach().numpy())
                    free_sum = np.add(free_sum, utility_score * polarity_free_rep.detach().numpy())

                    user_item_vector[acc] = utility_score

                    pos_samples += 1

                else:
                    print('Did not interact... continuing')
                    continue

    polarized_sum /= total_utility_score
    free_sum /= total_utility_score

    ##return (embedding, user_item vector)
    return polarized_sum, free_sum, user_item_vector

bert_dim = 60 + 256  # Example BERT embedding size
intermediate_dim = 256
encoder_output_dim = 128

encoder = Encoder(bert_dim, intermediate_dim, encoder_output_dim)
polarity_decoder = Decoder(encoder_output_dim, intermediate_dim, encoder_output_dim)
polarity_free_decoder = Decoder(encoder_output_dim, intermediate_dim, encoder_output_dim)

encoder.load_state_dict(torch.load('src\my_work\models\encoder.pt', weights_only=True))
polarity_decoder.load_state_dict(torch.load('src\my_work\models\polarity_decoder.pt', weights_only=True))
polarity_free_decoder.load_state_dict(torch.load('src\my_work\models\polarity_free_decoder.pt', weights_only=True))

encoder.eval()
polarity_decoder.eval()
polarity_free_decoder.eval()

landmarks = pd.read_csv("src\data\\landmark_data\\landmark_embeddings.csv").drop(['Unnamed: 0'], axis=1)

##must be in embedding order
topic_lists = pd.read_csv("src\data\\landmark_data\\validation_topics_in_embedding_order.csv")     
data_source_path = "D:\Bert-Embeddings\\validation_data\\"

def dist(a, b):
    return np.linalg.norm(a-b)

def landmark_embedding(landmarks, embedding):
    return [dist(np.array(lm), embedding) for _ , lm in landmarks.iterrows()]

t = pd.read_pickle('1000users.pkl')
classes = ['bystanders', 'core conserv', 'country first conserv', 'devout and diverse', 'disaffected democrats', 'market skeptic repub', 'new era enterprisers', 'oppty democrats', 'solid liberas']

item_ids = topic_lists['article_id'].values

dummy_data = np.zeros((1000, len(item_ids)))

user_item_matrix = pd.DataFrame(dummy_data, columns=item_ids)

user_space_matrix = pd.DataFrame(columns=['embedding', 'class'])

polarity_free_interest_model = pd.DataFrame(columns=['interest model'])

distance_embeddings = np.zeros((1000, 9), dtype=np.float64)

q = 0
for i in range(len(classes)):
    c = classes[i]
    class_users = np.array(t[c])
    
    user_class_embeddings = []
    p_free_embeddings = []
    ## for each generated user of that class
    for j in range(class_users.shape[0]):
        
        ##need to sample our data from test set - for every user, according to their decision matrix 
        user = np.array(class_users[i])
        user_utility = user.flatten()
        
        user_landmark_embedding, polarity_free_embedding, user_item_vector = generate_user_content_embedding(topic_lists, data_source_path, user_utility, encoder_output_dim, encoder, polarity_decoder, polarity_free_decoder)
        
        ##need to drop unnamed from landmarks
        new = landmark_embedding(landmarks, user_landmark_embedding)
        
        user_item_matrix.iloc[q] = user_item_vector
        distance_embeddings[q] = new
        q += 1
        
        p_free_embeddings.append({'interest model': polarity_free_embedding})
        user_class_embeddings.append({'embedding': new, 'class': c})
    
    user_space_matrix = pd.concat([user_space_matrix, pd.DataFrame.from_records(user_class_embeddings)], ignore_index=True)
    polarity_free_interest_model = pd.concat([polarity_free_interest_model, pd.DataFrame.from_records(p_free_embeddings)], ignore_index=True)
        
user_space_matrix.to_csv('src\data\\user_space\\user_space_matrix_with_topics.csv')
distance_embeddings =  pd.DataFrame(distance_embeddings)
distance_embeddings.to_csv('src\data\\user_space\\user_space_matrix.csv')

user_item_matrix.to_csv('src\data\\CF\\user_item_matrix.csv')

##Now - should have a 9d vector for each of our 1000 users, now we have to calculate all of their similarity

dot_product_matrix = linear_kernel(distance_embeddings)

# Convert back to a DataFrame (optional, for easier interpretation)
dot_product_df = pd.DataFrame(dot_product_matrix)
dot_product_df.to_csv('src\data\\user_space\\correlation_matrix.csv')

polarity_free_interest_model.to_csv('src\\data\\CF\\interest_models.csv')

##edits - use training data (only 4000 items)
## also create a user-item matrix 
