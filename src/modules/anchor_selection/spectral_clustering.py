import torch, cv2, warnings
warnings.filterwarnings("ignore", message="Graph is not fully connected")
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from sklearn.cluster import SpectralClustering
from src.utils.basic.wsi_dataset import WSIImageDataset



def spectral_cluster_selection(query_region, encoder, n=4, step=100, neighbor_distance=1):
    img_w, img_h = query_region.size
    patch_w, patch_h = (224, 224)

    patches, coords = [], []
    for y in range(0, img_h - patch_h, step):
        for x in range(0, img_w - patch_w, step):
            patch = query_region.crop((x, y, x + patch_w, y + patch_h))
            patches.append(patch)
            coords.append((x + patch_w // 2, y + patch_h // 2)) 

    wsi_dataset = WSIImageDataset(patches, encoder.transform)
    dataloader = DataLoader(wsi_dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)
    embeddings = encoder.encode_wsi_patch("query region", dataloader, show=False)
    total_emb = torch.cat(embeddings, dim=0).tolist()

    data = []
    for index in range(len(total_emb)):
        data.append((total_emb[index], coords[index]))

    embeddings = np.array([item[0] for item in data])
    coords = np.array([item[1] for item in data])

    n_patches = len(data)

    overlap_matrix = np.zeros((n_patches, n_patches))
    for i in range(n_patches):
        for j in range(i + 1, n_patches):
            if np.linalg.norm(coords[i] - coords[j]) <= neighbor_distance:
                overlap_matrix[i, j] = 1
                overlap_matrix[j, i] = 1

    similarity_matrix = cosine_similarity(embeddings)
    similarity_grid = similarity_matrix * overlap_matrix

    clustering = SpectralClustering(
        n_clusters=n,
        affinity='precomputed',
        random_state=42
    )
    labels = clustering.fit_predict(similarity_grid)

    centers_data = []
    selected_coords = []
    for cluster_id in range(n):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        cluster_coordinates = coords[cluster_indices]

        cluster_center_data = np.mean(cluster_embeddings, axis=0)
        cluster_center_coords = np.mean(cluster_coordinates, axis=0)

        centers_data.append(cluster_center_data)
        selected_coords.append(cluster_center_coords)
    
    return selected_coords



def visualize_clusters_with_centers(coords, labels, centers, image_width, image_height, save_path=None):
    img = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255  
    for i, coord in enumerate(coords):
        label = labels[i]
        color = plt.colormaps['tab10'](label / (max(labels) + 1))
        color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255)) 

        top_left = (int(coord[0] - 25), int(coord[1] - 25))
        bottom_right = (int(coord[0] + 25), int(coord[1] + 25))
        cv2.rectangle(img, top_left, bottom_right, color, -1) 

    for i, center in enumerate(centers):
        label = labels[i] 
        color = (0, 0, 255) 
        cv2.circle(img, (int(center[0]), int(center[1])), 10, color, -1) 

    if save_path:
        try:
            cv2.imwrite(save_path, img)
            print(f"Image saved to {save_path}")
        except Exception as e:
            print(f"Image saving unsuccessful. Error: {e}")

        
    