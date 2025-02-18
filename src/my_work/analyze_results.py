import pandas as pd


def log(user_metrics):
    classes = [
    "bystanders",
    "solid liberas",
    "oppty democrats",
    "disaffected democrats",
    "devout and diverse",
    "new era enterprisers",
    "market skeptic repub",
    "country first conserv",
    "core conserv",
    ]

    ##class_map = {'bystanders': 0, 'core conserv': 8, 'country first conserv': 7, 'devout and diverse': 4, 'disaffected democrats': 3, 'market skeptic repub': 6, 'new era enterprisers': 5, 'oppty democrats': 2, 'solid liberas': 1}

    number_of_users = [49, 193, 133, 108, 69, 116, 116, 63, 153]

    # user_metrics = pd.read_csv('src\\data\\baseline_data\\total_eval\\all_user_metrics.csv').drop(columns=['Unnamed: 0'])

    curr = 0
    for i in range(len(classes)):
        class_users = user_metrics.iloc[curr: curr + number_of_users[i]]
        class_users = class_users.loc[(class_users!=0).any(axis=1)]
        print(f'Class {classes[i]}----------------- ')
        print('Topic coverage: ', class_users['topic_hit'].mean())
        print('Diversity over hits: ', class_users['diversity'].mean())
        curr += number_of_users[i]


paths = ['src/data/baseline_data/total_eval/results/FN_Embedding_CPC.csv', 'src/data/baseline_data/total_eval/results/FN_Rating_CPC.csv',
        'src/data/baseline_data/total_eval/results/NN_Embedding_CPC.csv', 'src/data/baseline_data/total_eval/results/NN_Rating_CPC.csv']

for p in paths:
    
    user_metrics = pd.read_csv(p).drop(columns=['Unnamed: 0'])
     
    log(user_metrics)