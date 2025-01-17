import numpy as np



def cos_sim_list(list1, list2):
    vec1 = np.array(list1)
    vec2 = np.array(list2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0, norm1, norm2
    
    return dot_product / (norm1 * norm2), norm1, norm2
