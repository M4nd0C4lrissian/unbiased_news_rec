import os
import pandas as pd
from torch.utils.data import Dataset
import torch
##from torchvision.io import read_image


def __normalizeLabels__(label):
    if label <= -1:
        label = [1, 0, 0]
    elif label >= 1:
        label = [1, 0, 1]
    else:
        label = [0, 1, 0]

    return torch.tensor(label)


class CustomArticleEmbeddingDataset(Dataset):
    def __init__(self, labels_file, text_embedding_file, title_embedding_file, slice, transform = __normalizeLabels__):
        self.all_article_labels = pd.read_csv(labels_file)
        self.current_article_labels = self.all_article_labels.iloc[slice[0] : slice[1], ]
        self.text_embeddings = torch.load(text_embedding_file)
        self.title_embeddings = torch.load(title_embedding_file)
        self.transform = transform


    def __len__(self):
        return self.current_article_labels.shape[0]

    def __getitem__(self, idx):
  
        ##issue - labels_file is of length 40000, while title and text are of length 1000 
        text = self.text_embeddings[idx]
        title = self.title_embeddings[idx]
        label = self.transform(self.current_article_labels.iloc[idx]['source_partisan_score'])
        
        return (title.T, text.T, label)
    
    def update_dataset(self, text_embedding_file, title_embedding_file, slice):
        self.text_embeddings = torch.load(text_embedding_file)
        self.title_embeddings = torch.load(title_embedding_file)
        self.current_article_labels = self.all_article_labels[slice[0] : slice[1]]

## looking to return title_embedding, text_embedding, source_partisan_score
## using text_embedding.pt file, title embedding.pt file, and partisan labels (associated in order of positional index)

if __name__ == '__main__':

    labels_file = "unbiased_news_rec\src\data\\auto_encoder_training\partisan_labels.csv"
    text_embedding_file = "unbiased_news_rec\src\data\\auto_encoder_training\\text_embedding_0.pt"
    title_embedding_file = "unbiased_news_rec\src\data\\auto_encoder_training\\title_embedding_0.pt"

    C = CustomArticleEmbeddingDataset(labels_file, text_embedding_file, title_embedding_file, [0 , 1000])

    print(C.__getitem__(2))