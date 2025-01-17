import torch, cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from sklearn.cluster import KMeans
from src.utils.basic.wsi_dataset import WSIImageDataset



def kmeans_cluster_selection(query_region, encoder, n=4, step=100):
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

    encoded_vectors = np.array([d[0] for d in data]) 
    positions = np.array([d[1] for d in data])

    kmeans = KMeans(n_clusters=n, random_state=0)
    kmeans.fit(encoded_vectors)

    labels = kmeans.labels_
    
    selected_coords = []
    for cluster_center in kmeans.cluster_centers_:
        distances = np.linalg.norm(encoded_vectors - cluster_center, axis=1)
        closest_patch_idx = np.argmin(distances)
        selected_coords.append(positions[closest_patch_idx])
    
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

        
    