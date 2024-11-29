import pandas as pd
import numpy as np
import ast

def process(training_data_path, label_data_path, common_col, new_col):
    train_data = pd.read_csv(training_data_path)
    label_data = pd.read_csv(label_data_path)

    labels = pd.DataFrame()
    for i in range(train_data.shape[0]):
        row = train_data.iloc[i]
        id = row[common_col]
        
        label_row = label_data.loc[label_data[common_col] == id]
        new_val = label_row[new_col]
        labels = pd.concat([labels, pd.DataFrame.from_records([{ common_col: id, new_col: ast.literal_eval(new_val.values[0]) }])], ignore_index=True)

    return labels


if __name__ == '__main__':

    # training_data_path = 'src\\data\\auto_encoder_training\\bert_validation_set.csv'
    # label_data_path = 'src\\data\\auto_encoder_training\\validation_mask.csv'

    # common_col = 'article_id'
    # new_col = 'source_partisan_score'

    # labels = process(training_data_path, label_data_path, common_col, new_col)

    # df = pd.DataFrame(labels, columns=['article_id', 'source_partisan_score'])
    # df.to_csv('src\\data\\auto_encoder_training\\validation_partisan_labels.csv')
    
    
    training_data_path = 'src\\data\\auto_encoder_training\\bert_training_set.csv'
    label_data_path = 'src\\data\\landmark_data\\item_topic_vector.csv'
    
    common_col = 'article_id'
    new_col = 'topical_vector'
    
    labels = process(training_data_path, label_data_path, common_col, new_col)
    
    # df = pd.DataFrame(labels, columns=[common_col, new_col])
    labels.to_csv('src\\data\\landmark_data\\topics_in_embedding_order.csv')
    