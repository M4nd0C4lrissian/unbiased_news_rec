import pandas as pd
import numpy as np

def process(training_data_path, label_data_path, common_col, new_col):
    train_data = pd.read_csv(training_data_path)
    label_data = pd.read_csv(label_data_path)

    labels = np.zeros([train_data.shape[0],2], dtype=np.int64)
    for i in range(train_data.shape[0]):
        row = train_data.iloc[i]
        id = row[common_col]
        
        label_row = label_data.loc[label_data[common_col] == id]
        new_val = label_row[new_col]
        labels[i][0], labels[i][1] = int(id), int(new_val)

    return labels


training_data_path = 'unbiased_news_rec\\src\\data\\auto_encoder_training\\bert_training_set.csv'
label_data_path = 'unbiased_news_rec\\src\\data\\auto_encoder_training\\training_mask.csv'

common_col = 'article_id'
new_col = 'source_partisan_score'

labels = process(training_data_path, label_data_path, common_col, new_col)

df = pd.DataFrame(labels, columns=['article_id', 'source_partisan_score'])
df.to_csv('unbiased_news_rec\\src\\data\\auto_encoder_training\\partisan_labels.csv')