import sys, torch, cv2, warnings
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
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
            coords.append((x + patch_w // 2, y + patch_h // 2))  # Center of the patch

    wsi_dataset = WSIImageDataset(patches, encoder.transform)
    dataloader = DataLoader(wsi_dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)
    embeddings = encoder.encode_wsi_patch("query region", dataloader, show=False)
    total_emb = torch.cat(embeddings, dim=0).tolist()

    data = []
    for index in range(len(total_emb)):
        data.append((total_emb[index], coords[index]))

    # Step 1: 转换为 NumPy 数组
    embeddings = np.array([item[0] for item in data])
    coords = np.array([item[1] for item in data])

    n_patches = len(data)

     # Step 2: 构造邻接关系矩阵
    overlap_matrix = np.zeros((n_patches, n_patches))
    for i in range(n_patches):
        for j in range(i + 1, n_patches):
            if np.linalg.norm(coords[i] - coords[j]) <= neighbor_distance:
                overlap_matrix[i, j] = 1
                overlap_matrix[j, i] = 1

    # Step 3: 构建相似度矩阵
    similarity_matrix = cosine_similarity(embeddings)
    similarity_grid = similarity_matrix * overlap_matrix

    # Step 4: 执行谱聚类
    clustering = SpectralClustering(
        n_clusters=n,
        affinity='precomputed',
        random_state=42
    )
    labels = clustering.fit_predict(similarity_grid)

    # Step 5: 计算每个簇的中心
    centers_data = []
    selected_coords = []
    for cluster_id in range(n):
        # 找到当前簇的所有点
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_embeddings = embeddings[cluster_indices]
        cluster_coordinates = coords[cluster_indices]

        # 计算中心点 (嵌入和坐标取平均)
        cluster_center_data = np.mean(cluster_embeddings, axis=0)
        cluster_center_coords = np.mean(cluster_coordinates, axis=0)

        centers_data.append(cluster_center_data)
        selected_coords.append(cluster_center_coords)

    # save_path = f"image/clustering/result_spectral.png"
    # visualize_clusters_with_centers(coords, labels, selected_coords, img_w, img_h, save_path)
    
    return selected_coords











def visualize_clusters_with_centers(coords, labels, centers, image_width, image_height, save_path=None):
    # Create a blank image with the specified width and height (white background)
    img = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255  # White background

    # Draw each point as a 50x50 block, colored by its cluster label
    for i, coord in enumerate(coords):
        label = labels[i]
        # Get color from colormap (converted to BGR)
        color = plt.colormaps['tab10'](label / (max(labels) + 1))
        color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))  # Convert RGBA to BGR

        # Draw rectangle centered at (coord[0], coord[1]) with size 50x50
        top_left = (int(coord[0] - 25), int(coord[1] - 25))
        bottom_right = (int(coord[0] + 25), int(coord[1] + 25))
        cv2.rectangle(img, top_left, bottom_right, color, -1)  # -1 means filled

    # Draw the cluster centers as colored dots
    for i, center in enumerate(centers):
        label = labels[i]  # Use the label of the corresponding center
        # Get color from colormap (converted to BGR)
        color = (0, 0, 255)  # Red in BGR
        # Draw circle at the cluster center
        cv2.circle(img, (int(center[0]), int(center[1])), 10, color, -1)  # Radius = 10

    # Optionally save the image
    if save_path:
        try:
            cv2.imwrite(save_path, img)
            print(f"Image saved to {save_path}")
        except Exception as e:
            print(f"Image saving unsuccessful. Error: {e}")

        

if __name__ == "__main__":
    from src.utils.basic.encoder import WSI_Image_UNI_Encoder
    from src.utils.metadata_wsi.wsi_loader import load_wsi_region

    encoder = WSI_Image_UNI_Encoder()
    wsi_name, x, y, w, h, level, angle = "241183-21.tiff", 16456, 9632, 1014, 1541, 2, 120
    query_url = f"http://mdi.hkust-gz.edu.cn/wsi/metaservice/api/region/openslide/{wsi_name}"
    _, query_region = load_wsi_region(query_url, x, y, w, h, level, angle)
    query_region.save(f"image/clustering/query_region.png")

    selected_coords = spectral_cluster_selection(query_region, encoder, n=4, step=100)
    print(selected_coords)
    