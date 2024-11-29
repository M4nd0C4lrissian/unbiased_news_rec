import torch
import numpy as np
import pandas as pd
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from my_work.training import Encoder, Decoder

bert_dim = 60 + 128  # Example BERT embedding size
intermediate_dim = 128
encoder_output_dim = 64

encoder = Encoder(bert_dim, intermediate_dim, encoder_output_dim)
polarity_decoder = Decoder(encoder_output_dim, intermediate_dim, encoder_output_dim)
polarity_free_decoder = Decoder(encoder_output_dim, intermediate_dim, encoder_output_dim)

encoder.load_state_dict(torch.load('src\my_work\models\encoder.pt', weights_only=True))
polarity_decoder.load_state_dict(torch.load('src\my_work\models\polarity_decoder.pt', weights_only=True))
polarity_free_decoder.load_state_dict(torch.load('src\my_work\models\polarity_free_decoder.pt', weights_only=True))

encoder.eval()
polarity_decoder.eval()
polarity_free_decoder.eval()


classes = ['bystanders', 'core conserv', 'country first conserv', 'devout and diverse', 'disaffected democrats', 'market skeptic repub', 'new era enterprisers', 'oppty democrats', 'solid liberals']

landmarks = np.array((9,64), dtype=np.float32)
for c in classes:
    avg_member = pd.read_csv('synthetic_user\\' + c +'.csv')
    print(avg_member.head())
    print(avg_member.describe())