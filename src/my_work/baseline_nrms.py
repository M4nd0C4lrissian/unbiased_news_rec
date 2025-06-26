import pandas as pd
import random
from transformers import BertTokenizer
# tf.compat.v1.disable_v2_behavior()
from recommenders.models.newsrec.models.nrms import NRMSModel
# from recommenders.models.newsrec.models.nrms import NRMSConfig
from recommenders.datasets import Dataset


# Set a random seed
random_seed = 42
random.seed(random_seed)

# Set a random seed for PyTorch (for GPU as well)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def get_encoding(text, tokenizer):
  text_encoding = tokenizer.batch_encode_plus( text,# List of input texts
      padding=True,              # Pad to the maximum sequence length
      truncation=True,           # Truncate to the maximum sequence length if necessary
      return_tensors='pt',      # Return PyTorch tensors
      add_special_tokens=True    # Add special tokens CLS and SEP
  )

  return text_encoding

interactions = pd.read_csv("src\\data\\baseline_data\\training_interactions.csv").drop(columns=['interest'])
training_news = pd.read_csv("src\\data\\baseline_data\\baseline_training_data.csv").drop(columns=['source_partisan_score', 'topical_vector', 'text'])

test_interactions = pd.read_csv("src\\data\\baseline_data\\testing_interactions.csv").drop(columns=['interest'])
test_news = pd.read_csv("src\\data\\baseline_data\\baseline_testing_data.csv").drop(columns=['source_partisan_score', 'topical_vector', 'text'])

training_news['title'] = get_encoding(training_news['title'], tokenizer)
test_news['title'] = get_encoding(test_news['title'], tokenizer)

## I'm not sure our analysis is correct, I think our test data should be 

train = Dataset(
    interactions = interactions,
    items=training_news,
    user_col='user_id',
    item_col='article_id',
    timestamp_col='rel_timestamp',
    response_col='click'
)

test = Dataset(
    interactions = test_interactions,
    items=test_news,
    user_col='user_id',
    item_col='article_id',
    timestamp_col='rel_timestamp',
    response_col='click'
)

config = NRMSConfig(
    n_users = 8000,
    n_items = 32000,
    word_emb_dim = 128,
    num_attention_heads=8,
    head_dim=16,
    max_title_length=40,
    history_length = 50,
    pretrained_word_emb="glove.6B.100d.txt",
    batch_size=128,
    epochs=10
)
model = NRMSModel(config)
model.fit(train)
model.evaluate(test)