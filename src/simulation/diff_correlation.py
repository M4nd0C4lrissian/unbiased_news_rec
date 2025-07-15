import pandas as pd
import numpy as np
import math

def cos_sim(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def magnitude_weighted_cos_sim(u,v, alpha=0.1):
    return cos_sim(u, v) * math.exp(- alpha * np.linalg.norm(u - v, ord=1))

def parse_numpy_style_array(s):
    # Strip brackets, split on spaces, filter out empty strings, convert to float
    return np.array([float(x) for x in s.strip('[]').split() if x])

CPC = pd.read_csv('src\\data\\baseline_data\\CF\\bias_models.csv', index_col=0,
                usecols=['interest model'])
temp = []

for i in range(CPC.shape[0]):
    temp.append(parse_numpy_style_array(CPC.iloc[i].name))
    
CPC = np.array(temp)

correlation_matrix = np.zeros((CPC.shape[0], CPC.shape[0]))

##
alpha = 0.00
##

for i in range(CPC.shape[0]):
    curr_vec = CPC[i]
    for j in range(CPC.shape[0]):
        correlation_matrix[i][j] = magnitude_weighted_cos_sim(curr_vec, CPC[j], alpha)
        

pd.DataFrame(correlation_matrix).to_csv(f"src\\data\\baseline_data\\CF\\pure_embedding_correlation_matrix.csv")