import numpy as np

def weighted_avg(weights, vals):
    return np.dot(weights, vals)

class_dist = [49, 193, 133, 108, 69, 116, 116, 63, 153]
weights = np.array(class_dist) / sum(class_dist)

print("Avg: ", weighted_avg(weights, vals=[0.45, 0.36, 0.41, 0.40, 0.48, 0.43, 0.49, 0.43, 0.39]))