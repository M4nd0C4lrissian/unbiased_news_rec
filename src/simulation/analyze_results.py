import numpy as np
import pandas as pd

def global_avg():

    number_of_users = [49, 153, 63, 69, 108, 116, 116, 133, 193]

    classes = ['bystanders', 'core conserv', 'country first conserv', 'devout and diverse', 'disaffected democrats', 'market skeptic repub', 'new era enterprisers', 'oppty democrats', 'solid liberas']
        
    df = pd.read_csv('src\\data\\total_eval\\all_user_metrics.csv')

    class_stats = np.zeros((9, 2))
    offset = 0
    for c in range(len(classes)):
        for u in range(number_of_users[c]):
            user = df.iloc[offset + u]
            topic_cov = user['topic_hit']
            div = user['diversity']
            class_stats[c][0] += topic_cov
            class_stats[c][1] += div        
            
        offset += number_of_users[c]
        class_stats[c] /= number_of_users[c]
        

    pd.DataFrame(class_stats).to_csv('src\\data\\total_eval\\per_class_metrics.csv')
    
def in_depth_per_class():
    classes = ['bystanders', 'core conserv', 'country first conserv', 'devout and diverse', 'disaffected democrats', 'market skeptic repub', 'new era enterprisers', 'oppty democrats', 'solid liberas']
    number_of_users = [49, 153, 63, 69, 108, 116, 116, 133, 193]
    df = pd.read_csv('src\\data\\total_eval\\all_user_metrics.csv')
    offset = 0
    for c in range(len(classes)):
        class_users = df.iloc[offset:offset + number_of_users[c]]
        
        class_users.to_csv(f'src\\data\\total_eval\\{classes[c]}.csv')
        offset += number_of_users[c]
        
if __name__ == '__main__':
    in_depth_per_class()