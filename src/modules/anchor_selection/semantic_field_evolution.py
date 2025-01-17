import sys, torch, random
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from scipy import interpolate
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.distance import cdist

from scipy.interpolate import RegularGridInterpolator

from src.utils.basic.wsi_dataset import WSIImageDataset



# 给你一个二维列表，列表中的每个元素都是一个 1024维的向量代表每个点的语义表征，请你用 python 通过以步骤 k 个组合：

# Step 1: 获得所有 anchor 的语义空间表征，获得平均表征并获得所有 Embedding 在垂直于平均表征的平面上的投影；
# Step 2: 基于相邻点之间的表征，通过cosine相似度，获得所有点的 semantic force 方向；
# Step 3: 通过固定步骤进行 denaturation，每一步把 force 传递到相邻点上，并进行更新；
# Step 4: 基于构建的 field 的散度值最低的 k 个点来选择 target anchor。
# Step 5: 基于选择的的 target anchor 重新投影其他点的 anchor 表征，并在这上面找到相似度最低的 k 个点作为 valid point。

# 其中每个组合带有一个 target anchor 和 k 个 valid point。


def plot_arrow_on_image(image, position_matrix, direction_matrix, output_path, alpha=0.5, arrow_scale=50):
   # 加载图像并转换为 numpy 数组
    img_array = np.array(image)

    # 如果图像是 RGBA 格式，调整 alpha 通道来控制透明度
    if img_array.shape[2] == 4:
        img_array[:, :, 3] = (img_array[:, :, 3] * alpha).astype(np.uint8)
    else:
        # 如果图像是 RGB 格式，则添加 alpha 通道
        img_array = np.dstack([img_array, np.full(img_array.shape[:2], int(255 * alpha))])

    # 提取每个点的 x 和 y 坐标以及方向
    x = position_matrix[:, :, 0]  # x 坐标
    y = position_matrix[:, :, 1]  # y 坐标
    u = direction_matrix[:, :, 0] * arrow_scale  # x 方向的分量（缩放后）
    v = direction_matrix[:, :, 1] * arrow_scale  # y 方向的分量（缩放后）

    # 创建一个图形，并将图像显示在背景
    fig, ax = plt.subplots(figsize=(32, 32))
    ax.imshow(img_array, origin='upper')  # 显示图像，设置为原始坐标系（origin='upper'）

    # 绘制箭头，`quiver` 默认绘制的是有尾巴的箭头
    quiver = ax.quiver(
        x, y, u, v,
        angles='xy', scale_units='xy', scale=1, color='black', width=0.008, headwidth=5, headlength=7
    )

    # 设置坐标轴等
    ax.set_xlim(0, img_array.shape[1])
    ax.set_ylim(img_array.shape[0], 0)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.axis('off')  # 去掉坐标轴

    # 保存结果
    plt.savefig(output_path, bbox_inches='tight', transparent=True)
    plt.close()

def plot_pointers_with_arrows(A, B, image=None, output_path="output.png", alpha=0.5):
    """
    绘制 A 和 B 中的点及从 A 到 B 的矢量箭头。
    
    :param A: 一维列表，每个元素是 (x, y)，表示点的位置
    :param B: 二维列表，每个元素是一个包含 k 个 (x, y) 的列表，表示与 A 中点对应的多个位置
    :param image: 背景图像 (可选)，如果提供则绘制在背景上
    :param output_path: 输出图像文件路径
    """
    # 创建图形
    img_array = np.array(image)
    fig, ax = plt.subplots(figsize=(32, 32))
    
     # 如果图像是 RGBA 格式，调整 alpha 通道来控制透明度
    if img_array.shape[2] == 4:
        img_array[:, :, 3] = (img_array[:, :, 3] * alpha).astype(np.uint8)
    else:
        # 如果图像是 RGB 格式，则添加 alpha 通道
        img_array = np.dstack([img_array, np.full(img_array.shape[:2], int(255 * alpha))])
    
    ax.imshow(img_array, origin='upper')  # 显示图像，设置为原始坐标系（origin='upper'）
    
    # 提取 A 和 B 中的坐标并绘制
    for i, a_point in enumerate(A):
        # 绘制 A 中的红色点
        ax.scatter(a_point[0], a_point[1], color='black', s=50, label='A' if i == 0 else "")
        
        # 绘制 B 中的绿色点和箭头
        for b_point in B[i]:
            ax.scatter(b_point[0], b_point[1], color='green', s=30, label='B' if i == 0 else "")
            # 绘制箭头从 A 到 B
            ax.arrow(
                a_point[0], a_point[1],
                b_point[0] - a_point[0], b_point[1] - a_point[1],
                head_width=0.2, head_length=0.2, fc='blue', ec='blue', alpha=0.7
            )
    
    # 设置图形显示参数
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.legend()
    
    # 保存并显示图形
    plt.savefig(output_path, bbox_inches='tight', transparent=True)
    plt.close()

def plot_circles_with_values(coordinates, values, image=None, distance=224, output_path="output.png", alpha=0.5):
    # 获取坐标的形状，假设它是一个 (m, n, 2) 的数组，其中 m 和 n 是网格的行列数
    m, n, _ = coordinates.shape
    
    # 创建图形
    img_array = np.array(image) if image is not None else np.ones((500, 500, 3), dtype=np.uint8) * 255
    fig, ax = plt.subplots(figsize=(32, 32))
    
    # 如果图像是 RGBA 格式，调整 alpha 通道来控制透明度
    if img_array.shape[2] == 4:
        img_array[:, :, 3] = (img_array[:, :, 3] * alpha).astype(np.uint8)
    else:
        img_array = np.dstack([img_array, np.full(img_array.shape[:2], int(255 * alpha))])
    
    ax.imshow(img_array, origin='upper')  # 显示图像，设置为原始坐标系（origin='upper'）

    # 对每个点绘制红圈
    for i in range(m):
        for j in range(n):
            value = values[i, j]
            if np.isnan(value):  # 跳过 NaN 值
                continue
            
            point = coordinates[i, j]
            
            # 获取当前点的相邻点的最小距离，作为最大半径限制
            # 计算该点与相邻点的距离，排除 NaN 值
            min_distance = np.inf
            if i > 0:
                min_distance = min(min_distance, distance)  # 上
            if i < m - 1:
                min_distance = min(min_distance, distance)  # 下
            if j > 0:
                min_distance = min(min_distance, distance)  # 左
            if j < n - 1:
                min_distance = min(min_distance, distance)  # 右

            max_radius = min_distance / 2  # 最大半径为最小距离的一半
            
            # 根据值调整半径，假设值的大小与半径的比例线性关系
            radius = min(max_radius, value * 0.1)  # 缩放因子根据需要调整
            
            # 创建红色圆圈，大小与值成正比
            circle = patches.Circle((point[0], point[1]), radius, color='red', alpha=0.5)
            ax.add_patch(circle)
    
    # 设置图形显示参数
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.axis('off')  # 去掉坐标轴
    
    # 保存并显示图形
    plt.savefig(output_path, bbox_inches='tight', transparent=True)
    plt.close()

def plot_interpolated_points_and_selected(normalized_values, selected_points, position_matrix, scale_factor, image=None, output_path="output_with_selected.png", max_distance=2, alpha=0.5, step=100):
    # 创建图形
    img_array = np.array(image) if image is not None else np.ones((500, 500, 3), dtype=np.uint8) * 255
    fig, ax = plt.subplots(figsize=(32, 32))
    
    # 如果图像是 RGBA 格式，调整 alpha 通道来控制透明度
    if img_array.shape[2] == 4:
        img_array[:, :, 3] = (img_array[:, :, 3] * 0.5).astype(np.uint8)
    else:
        img_array = np.dstack([img_array, np.full(img_array.shape[:2], int(255 * 0.5))])
    
    ax.imshow(img_array, origin='upper')  # 显示图像，设置为原始坐标系（origin='upper'）

    # 遍历 normalized_values 中的每个点，并绘制到原始图像上
    m, n = normalized_values.shape
    for i in range(m):
        for j in range(n):
        
            value = normalized_values[i, j]
            x, y = position_matrix[i // scale_factor][j // scale_factor]  # 在原始位置矩阵上找到近似的点
            y, x = y + (step // scale_factor ) * (i % scale_factor), x + (step // scale_factor) * (j % scale_factor), 
            
            # 最大半径由 max_distance 决定，半径为 max_distance / 2
            max_radius = max_distance / 2  # 最大半径
            
            # 根据值调整半径，假设值的大小与半径的比例线性关系
            radius = min(max_radius, value*25)  # 缩放因子根据需要调整
            
            # 创建圆圈，大小与值成正比
            circle_color = 'red'
            if (i, j) in selected_points:  # 如果该点是被选择的点，改为绿色
                circle_color = 'green'
            
            circle = patches.Circle((x, y), radius, color=circle_color, alpha=0.5)  # 注意这里，x 和 y 对调
            ax.add_patch(circle)

    # 设置图形显示参数
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.axis('off')  # 去掉坐标轴
    
    # 保存并显示图形
    plt.savefig(output_path, bbox_inches='tight', transparent=True)
    plt.close()









    
def split_image_into_patches(image, patch_size=(224, 224), step=100):
    patches = []
    coords = []
    img_w, img_h = image.size
    patch_w, patch_h = patch_size

    for y in range(0, img_h - patch_h, step):
        cur_patches = []
        cur_coords = []
        for x in range(0, img_w - patch_w, step):
            patch = image.crop((x, y, x + patch_w, y + patch_h))
            cur_patches.append(patch)
            cur_coords.append((x + patch_w // 2, y + patch_h // 2))
        patches.append(cur_patches)
        coords.append(cur_coords)

    return patches, coords

def encode_patches(list_patches, encoder):
    data_embeddings = []
    for patches in list_patches:
        wsi_dataset = WSIImageDataset(patches, encoder.transform)
        dataloader = DataLoader(wsi_dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)
        embeddings = encoder.encode_wsi_patch("query region", dataloader)
        total_emb = torch.cat(embeddings, dim=0).tolist()
        data_embeddings.append(total_emb)

    return data_embeddings


class Semantic_Projection_Field_Denaturation():
    # Step 1: 获取所有 anchor 的语义空间表征，计算平均表征并获得所有嵌入在垂直于平均表征的平面上的投影
    def get_projection_to_plane(self, embeddings):
        n, m, _ = embeddings.shape
        avg_embedding = np.mean(embeddings, axis=(0, 1))  # 计算所有点的平均语义表示
        avg_embedding_unit = avg_embedding / np.linalg.norm(avg_embedding)
        projections = embeddings - np.dot(embeddings, avg_embedding_unit).reshape(n, m, 1) * avg_embedding_unit

        return projections

    # Step 2: 基于相邻点之间的表征，计算相似度，获得所有点的 semantic force 方向
    def compute_semantic_force(self, embeddings):
        n, m, d = embeddings.shape

        def compute_3d_similarity_matrix(A, B, n, m, d):
            A_flat = A.reshape(n * m, d)
            B_flat = B.reshape(n * m, d)
            
            A_norm = np.linalg.norm(A_flat, axis=1, keepdims=False)
            B_norm = np.linalg.norm(B_flat, axis=1, keepdims=False)
    
            dot_product = np.dot(A_flat, B_flat.T)
            diagonal_values = np.diagonal(dot_product)
            
            similarity_matrix = (diagonal_values+1e-6) / (A_norm * B_norm.T) 
            
            return similarity_matrix.reshape(n, m)

        # up
        original_embeddings = embeddings.copy()[1:, :, :]
        up_embeddings = embeddings.copy()[:-1, :, :]
        up_similarity_matrix = compute_3d_similarity_matrix(original_embeddings, up_embeddings, n-1, m, d)
        up_similarity_matrix = np.vstack([np.zeros((1, m)), up_similarity_matrix])

        # down
        original_embeddings = embeddings.copy()[:-1, :, :]
        down_embeddings = embeddings.copy()[1:, :, :]
        down_similarity_matrix = compute_3d_similarity_matrix(original_embeddings, down_embeddings, n-1, m, d)
        down_similarity_matrix = np.vstack([down_similarity_matrix, np.zeros((1, m))])

        # left
        original_embeddings = embeddings.copy()[:, 1:, :]
        left_embeddings = embeddings.copy()[:, :-1, :]
        left_similarity_matrix = compute_3d_similarity_matrix(original_embeddings, left_embeddings, n, m-1, d)
        left_similarity_matrix = np.hstack([np.zeros((n, 1)), left_similarity_matrix])
        
        # right
        original_embeddings = embeddings.copy()[:, :-1, :]
        right_embeddings = embeddings.copy()[:, 1:, :]
        right_similarity_matrix = compute_3d_similarity_matrix(original_embeddings, right_embeddings, n, m-1, d)
        right_similarity_matrix = np.hstack([right_similarity_matrix, np.zeros((n, 1))])

        horizontal_sim_forces = right_similarity_matrix - left_similarity_matrix
        vertical_sim_forces = down_similarity_matrix - up_similarity_matrix
        semantic_forces = np.stack([horizontal_sim_forces, vertical_sim_forces], axis=2)

        min_val = np.min(np.abs(semantic_forces))
        max_val = np.max(np.abs(semantic_forces))
        
        if max_val - min_val != 0:
            semantic_forces = np.sign(semantic_forces) * (np.abs(semantic_forces) - min_val) / (max_val - min_val)

        return semantic_forces

    # Step 3: 进行 denaturation，通过传递 force 到相邻点上，并进行更新
    def forces_evolution(self, forces, steps=5):
        main_field = forces.copy()
        evolve_field = forces.copy() / 2

        for _ in range(steps):
            down_shift = np.roll(evolve_field.copy(), 1, axis=0)
            down_shift[0, :] = 0
            up_shift = np.roll(evolve_field.copy(), -1, axis=0) 
            up_shift[-1, :] = 0
            left_shift = np.roll(evolve_field.copy(), 1, axis=1)
            left_shift[:, 0] = 0
            right_shift = np.roll(evolve_field.copy(), -1, axis=1)
            right_shift[:, -1] = 0

            main_field =  main_field / 2 + (down_shift + up_shift + left_shift + right_shift) / 4
            evolve_field = main_field.copy() / 2

            min_val = np.min(np.abs(main_field))
            max_val = np.max(np.abs(main_field))
            if max_val - min_val != 0: 
                main_field = np.sign(main_field) * (np.abs(main_field) - min_val) / (max_val - min_val)

            min_val = np.min(np.abs(evolve_field))
            max_val = np.max(np.abs(evolve_field))
            if max_val - min_val != 0:
                evolve_field = np.sign(evolve_field) * (np.abs(evolve_field) - min_val) / (max_val - min_val)
        
        return main_field

    # Step 4: 数值 Evolution
    def value_evolution(self, evolve_field, steps=10, kappa=2):
        # 通过模拟流向来取得最大的k个点
        n, m, _ = evolve_field.shape
        initial_sum = n * m * 100  # 记录初始的总和
        evolved_matrix = np.full((n, m), 100)
        evolve_field_x = evolve_field[:, :, 0]
        evolve_field_y = evolve_field[:, :, 1]

        for _ in range(steps):
            change_x = evolved_matrix * evolve_field_x / ((kappa - 1) / (4 * kappa))
            change_y = evolved_matrix * evolve_field_y / ((kappa - 1) / (4 * kappa))

            down_shift = np.maximum(np.roll(change_y.copy(), 1, axis=0), 0)
            down_shift[0, :] = 0
            up_shift = np.maximum(-np.roll(change_y.copy(), -1, axis=0), 0) 
            up_shift[-1, :] = 0
            left_shift = np.maximum(-np.roll(change_x.copy(), -1, axis=1), 0)
            left_shift[:, -1] = 0
            right_shift = np.maximum(np.roll(change_x.copy(), 1, axis=1), 0)
            right_shift[:, 0] = 0

            evolved_matrix = evolved_matrix / kappa + down_shift + up_shift + left_shift + right_shift

            evolved_matrix *= initial_sum / np.sum(evolved_matrix)

        return evolved_matrix
    
    def process_and_select_points(self, position_matrix, value_matrix, scale_factor, k):
        """
        执行整个过程：插值、归一化、选择k个点。
        
        参数:
        - position_matrix: 位置矩阵
        - value_matrix: 对应的值矩阵
        - scale_factor: 插值放大倍数
        - k: 选择的点的数量
        
        返回:
        - 选择的点的坐标
        """
        def interpolate_matrix(data, scale_factor):
            x = np.arange(data.shape[1])
            y = np.arange(data.shape[0])
            
            # 创建 RegularGridInterpolator 插值函数
            interp_func = interpolate.RegularGridInterpolator((y, x), data, method='linear', bounds_error=False, fill_value=0)
            
            # 计算新的插值网格（放大后的网格）
            x_new = np.linspace(0, data.shape[1] - 1, int(data.shape[1] * scale_factor))
            y_new = np.linspace(0, data.shape[0] - 1, int(data.shape[0] * scale_factor))
            
            # 确保 x_new 和 y_new 的大小是匹配的
            grid_x, grid_y = np.meshgrid(x_new, y_new)  # x_new 和 y_new 在这里是按比例扩展的
            data_interpolated = interp_func((grid_y, grid_x))  # 传递 (y_new, x_new)
            
            return data_interpolated

        def normalize_matrix(data, position_matrix=None):
            data_min = np.min(data)
            data_max = np.max(data)
            data_normalized = (data - data_min) / (data_max - data_min)
            
            if position_matrix is not None:
                x_min, y_min = np.min(position_matrix, axis=(0, 1))
                x_max, y_max = np.max(position_matrix, axis=(0, 1))
                position_matrix_normalized = (position_matrix - np.array([x_min, y_min])) / np.array([x_max - x_min, y_max - y_min])
                return data_normalized, position_matrix_normalized
            
            return data_normalized

        def select_k_points(values, positions, k):
            dist_matrix = cdist(positions.reshape(-1, 2), positions.reshape(-1, 2))
            dist_max = np.max(dist_matrix)
            dist_matrix_normalized = dist_matrix / dist_max  # 归一化距离矩阵

            # 贪心选择k个点
            n_points = values.size
            selected_points = [0]  # 从第一个点开始
            while len(selected_points) < k:
                best_point = -1
                best_value = -np.inf
                for i in range(n_points):
                    if i in selected_points:
                        continue
                    # 计算新加入点后的目标函数值
                    selected_points_temp = selected_points + [i]
                    value_avg = np.mean(values.flatten()[selected_points_temp])  # 平均值
                    dist_avg = np.mean([dist_matrix_normalized[x, y] for x in selected_points_temp for y in selected_points_temp if x != y])  # 平均距离
                    total_value = value_avg + dist_avg
                    
                    # 选择目标函数值最大的点
                    if total_value > best_value:
                        best_value = total_value
                        best_point = i
                
                # 将最佳点加入已选择的点
                selected_points.append(best_point)

            # 返回选择的点的二维坐标
            selected_coords = [(point // values.shape[1], point % values.shape[1]) for point in selected_points]
            
            return selected_coords

        def map_to_original_coordinates(selected_points, position_matrix, scale_factor):
            """
            将插值后的选定点映射回原始图像上的位置
            :param selected_points: 插值后选定的点（例如：(0, 1)）
            :param position_matrix: 原始图像的位置矩阵
            :param scale_factor: 插值的缩放因子
            :return: 对应的原始图像位置
            """
            # 通过插值后的点的索引计算插值后的坐标
            selected_y, selected_x = selected_points
            
            # 获取 position_matrix 中的点
            original_y, original_x = position_matrix[selected_y // scale_factor][selected_x // scale_factor]
            
            # 根据 scale_factor 计算原始图像的位置
            # 由于插值是按比例扩展的，原始图像位置通过除以 scale_factor 来反向映射
            original_y += (selected_y % scale_factor) * (position_matrix[0][0][1] - position_matrix[1][0][1])
            original_x += (selected_x % scale_factor) * (position_matrix[0][0][0] - position_matrix[0][1][0])
            
            return (original_x, original_y)
        
        # 插值扩展
        position_interpolated = interpolate_matrix(position_matrix, scale_factor)
        value_interpolated = interpolate_matrix(value_matrix, scale_factor)
        
        # 归一化
        normalized_values, normalized_positions = normalize_matrix(value_interpolated, position_interpolated)
        
        # 选择k个点
        selected_points = select_k_points(normalized_values, normalized_positions, k)

        print(len(position_matrix), len(position_matrix[0]), selected_points)

        original_positions = [map_to_original_coordinates(point, position_matrix, scale_factor) for point in selected_points]
        
        return original_positions, selected_points, normalized_values, normalized_positions













def pre_select_anchor_evolution(query_region, encoder, n=4, step=100, show=True):
    patches, coords = split_image_into_patches(query_region, patch_size=(224, 224), step=step)
    data_embeddings = encode_patches(patches, encoder)  # data包含vector和position
    encoded_vectors = np.array(data_embeddings)
    positions = np.array(coords)

    method = Semantic_Projection_Field_Denaturation()
    if show:
        print("Encoded Vectors Shape:", encoded_vectors.shape)
        print("Positions Shape:", positions.shape)

    projections = method.get_projection_to_plane(encoded_vectors)
    if show:
        print("Projections Shape:", projections.shape)

    semantic_forces = method.compute_semantic_force(projections)
    if show:
        print("Semantic Forces Shape:", semantic_forces.shape)
        plot_arrow_on_image(query_region, positions, semantic_forces, "image/clustering/query_region_forces.png")

    evolved_forces = method.forces_evolution(semantic_forces, steps=3)
    if show:
        print("Denatured Forces Shape:", evolved_forces.shape)
        plot_arrow_on_image(query_region, positions, evolved_forces, "image/clustering/query_region_denatured_forces.png")

    evolved_value_matrix = method.value_evolution(evolved_forces)
    if show:
        print("Evolutio Value matrix Shape:", evolved_value_matrix.shape)
        plot_circles_with_values(positions, evolved_value_matrix, image=query_region, distance=step, output_path="image/clustering/query_region_evolution_values.png", alpha=0.5)

    original_positions, selected_points, normalized_values, normalized_positions = method.process_and_select_points(positions, evolved_value_matrix, scale_factor=3, k=n)
    if show:
        print(original_positions)
        plot_interpolated_points_and_selected(normalized_values, selected_points, positions, scale_factor=3, image=query_region, output_path="image/clustering/query_region_evolution_values_normal.png", max_distance=50, alpha=0.5, step=step)


    return original_positions





if __name__ == "__main__":
    from src.utils.basic.encoder import WSI_Image_UNI_Encoder, WSI_Image_test_Encoder
    encoder = WSI_Image_UNI_Encoder()

    query_region = Image.open("image/clustering/query_region.png")

    # 调用函数生成组合
    target_anchors = pre_select_anchor_evolution(query_region, encoder, n=4)
    print(query_region.size)