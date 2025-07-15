import numpy as np
import pandas as pd


paths = ['8_neighbors_FN_Rating_CPC']
columns = ['CTR', 'Utility', '% Topic Coverage', '% Bias Diversity', 'PE@10']

list = []
for i in range(len(paths)):
    
    df = pd.read_csv(f"src\data\\baseline_data\\total_eval\\metrics\\metrics_{paths[i]}.csv").drop(columns=['Unnamed: 0'])
    
    dict = {}
    for j in range(df.shape[1]):
        mean = df.iloc[:, j].mean()
        dict[columns[j]] = str(round(mean, 2))
    
    list.append(dict)

pd.DataFrame.from_records(list, index=paths).to_csv("src\data\\baseline_data\\total_eval\metrics\\metrics_average.csv")