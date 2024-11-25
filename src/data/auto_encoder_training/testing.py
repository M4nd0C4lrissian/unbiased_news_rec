import torch
import pandas as pd
import numpy as np

data = torch.load('unbiased_news_rec\src\data\\auto_encoder_training\\title_embedding_0.pt')
print(data[2])