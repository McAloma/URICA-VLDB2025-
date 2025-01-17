import sys, os, torch
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.model.load_train_data import Represent_Data_Loader
from src.model.represent_model import Represent_Model



# NOTE: 该测试方法为检查 patch 对的编码在经过距离排序后，相似性的排序情况；
# 理想情况是，按照距离从低到高排序时候，相似性需要从高到低，以此来更好的定位； 
# 因此，我们基于排序情况，检查相似性序列的正序数比例，以此来看是否能够达到定位的效果；
# 从目前的测试结果来看：
# 单个 WSI 上的固定 patch，原始编码的平均正序数比例为：0.4754，编码后的正序数比例为：0.9998
# 单个 WSI 上的随机 patch，原始编码的平均正序数比例为：0.4756，编码后的正序数比例为：0.9997
# 可视化：绘制散点图来看变化情况。

def count_inversions(arr):
    """
    使用归并排序统计逆序数对数量，适用于 PyTorch 张量
    """
    if arr.size(0) < 2:
        return arr, 0
    mid = arr.size(0) // 2
    left, left_inv = count_inversions(arr[:mid])
    right, right_inv = count_inversions(arr[mid:])
    merged, split_inv = merge_and_count(left, right)
    return merged, left_inv + right_inv + split_inv

def merge_and_count(left, right):
    merged = []
    i = j = inv_count = 0
    left_size, right_size = left.size(0), right.size(0)
    left, right = left.tolist(), right.tolist() 

    while i < left_size and j < right_size:
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            inv_count += left_size - i  
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return torch.tensor(merged), inv_count


def represent_testing(target_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = Represent_Data_Loader()

    if not target_dir:
        target_dir = "data/metadata_embedding"
    else:
        target_dir = target_dir

    specified_folder_name = os.listdir(target_dir)
    test_loader = data_loader.get_dataloader(n=1000, batch_size=1000, specified_folder_name=specified_folder_name)

    model = Represent_Model().to(device)
    checkpoint_path = "ckpts/represent_model_shift.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    raw_scores, fix_scores = [], []
    dises = []
    coses_raw = []
    coses_fix = []
    with torch.no_grad(): 
        with tqdm(test_loader, ascii=True) as pbar:
            for (raw1, raw2, pos1, pos2) in pbar:
                raw1, raw2, pos1, pos2 = raw1.to(device), raw2.to(device), pos1.to(device), pos2.to(device)

                feature1 = model.represent(raw1)
                feature2 = model.represent(raw2)

                raw_cos_sin = F.cosine_similarity(raw1, raw2, dim=1).squeeze()
                fix_cos_sim = F.cosine_similarity(feature1, feature2, dim=1).squeeze()
                distances = torch.norm(pos1 - pos2, dim=1) / 224 
                distances = distances.squeeze()

                dises.extend(distances.tolist())
                coses_raw.extend(raw_cos_sin.tolist())
                coses_fix.extend(fix_cos_sim.tolist())

                sorted_indices = torch.argsort(distances)   # 按照距离排序，然后看相似度的情况（从低到高）

                # raw score
                sorted_raw_cos_sim = raw_cos_sin[sorted_indices]
                _, raw_inverse_count = count_inversions(sorted_raw_cos_sim)
                raw_total_pairs = sorted_raw_cos_sim.size(0) * (sorted_raw_cos_sim.size(0) - 1) // 2
                raw_score = (raw_total_pairs - raw_inverse_count) / raw_total_pairs
                raw_scores.append(raw_score)

                # fix score
                sorted_fix_cos_sim = fix_cos_sim[sorted_indices]
                _, inverse_count = count_inversions(sorted_fix_cos_sim)
                total_pairs = sorted_fix_cos_sim.size(0) * (sorted_fix_cos_sim.size(0) - 1) // 2
                fix_score = (total_pairs - inverse_count) / total_pairs
                fix_scores.append(fix_score)

    if target_dir == "data/metadata_embedding_random":
        draw_scatter(dises, coses_raw, "Similarity_distance_mapping_random_raw")
        draw_scatter(dises, coses_fix, "Similarity_distance_mapping_random_fix")
    else:
        draw_scatter(dises, coses_raw, "Similarity_distance_mapping_raw")
        draw_scatter(dises, coses_fix, "Similarity_distance_mapping_fix")

    return sum(raw_scores) / len(raw_scores), sum(fix_scores) / len(fix_scores)


def draw_scatter(distance, sim, name=None):
    if name == None:
        return
    
    x_values = distance
    y_values = sim

    plt.scatter(x_values, y_values, color='blue', marker='.')
    plt.title(f"Scatter Plot of {name}")
    plt.xlabel("Distance")
    plt.ylabel("Y-axis")
    plt.savefig(f"image/{name}.png")
    plt.close()


if __name__ == "__main__":
    avg_raw_score, avg_fix_score = represent_testing()
    print(f"For mesh patch, the Average Raw Score is {avg_raw_score}, the Average Fix Score is {avg_fix_score}")

    target_dir = "data/metadata_embedding_random"
    avg_raw_score, avg_fix_score = represent_testing(target_dir)
    print(f"For Random patch, the Average Raw Score is {avg_raw_score}, the Average Fix Score is {avg_fix_score}")
