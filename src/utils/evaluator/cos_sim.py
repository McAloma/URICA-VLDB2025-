import sys 
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
import numpy as np

from src.utils.metadata_wsi.wsi_loader import load_img_url



def cos_sim_list(list1, list2):
    vec1 = np.array(list1)
    vec2 = np.array(list2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0, norm1, norm2
    
    return dot_product / (norm1 * norm2), norm1, norm2


def cos_sim_url(query_url, retrieval_url, encoder):
    query_image = load_img_url(query_url)
    retrieval_image = load_img_url(retrieval_url)

    query_embedding = encoder.encode_image(query_image)
    retrieval_embedding = encoder.encode_image(retrieval_image)

    dot_product = np.dot(query_embedding, retrieval_embedding)

    norm_a = np.linalg.norm(query_embedding)
    norm_b = np.linalg.norm(retrieval_embedding)

    cosine_similarity = dot_product / (norm_a * norm_b)

    return cosine_similarity