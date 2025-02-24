import sys, os, time, math
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
from collections import Counter
import numpy as np

from src.utils.evaluator.cos_sim import cos_sim_list
from src.utils.evaluator.iou import region_retrieval_self_single_IoU
from src.utils.open_wsi.wsi_loader import load_region_png
from src.modules.anchor_selection.selection_all import select_all_anchors
from src.modules.anchor_selection.kmeas_clustering import kmeans_cluster_selection
from src.modules.anchor_selection.spectral_clustering import spectral_cluster_selection



# Region Retrieval 的基本流程是基于 query region 选择一些 tessellation 上的 anchor 作为基本点，并各自出发原始的 patch retriever；
# 基于每个 target anchor 的检索结果，在结果最多的 WSI 和 subtype 上使用其他的 valid anchor 再次触发检索获得 valid point；
# 每个 target anchor retrieved patch 和 valid point 之间构建 pointer，用于后续的 region 验证；
# 验证的方式有两种：一种是 traversal，另一种是 boc，前者是通过计算所有 pointer 的距禮和角度的方差最小的组合，后者是通过将距离和角度分别分组，然后计算方差最小的组合；
# 最后，通过验证的结果，计算 query region 和 region candidate 之间的相似度，返回 top_k 的结果。

class Image2Image_Region_Retriever():
    def __init__(self, basic_retriever, encoder):
        if basic_retriever:
            self.basic_retriever = basic_retriever
        else:
            ValueError("Please select basic retriever.")
        self.encoder = encoder
        self.site = basic_retriever.site

    def region_retrieval(self, query_region, preprocess="spectral", evaluation="boc", top_k=5, step=100, show=False):
        # Step 1: Choosing Anchor
        w, h = query_region.size
        if preprocess == "kmeans":
            target_anchor_pos = kmeans_cluster_selection(query_region, self.encoder, n=4, step=step)
        elif preprocess == "spectral":
            target_anchor_pos = spectral_cluster_selection(query_region, self.encoder, n=4, step=step)
        else:
            target_anchor_pos = select_all_anchors(w, h, step=step)
        if show:
            print(f"1. Select anchor positions: {target_anchor_pos}")
        valid_anchor_pos = [target_anchor_pos.copy() for _ in range(len(target_anchor_pos))]

        # Step 2: Main Retrieval with target anchors
        target_anchor_retrieve_results = {}   
        for (x, y) in target_anchor_pos:
            box = (x-112, y-112, x+112, y+112)
            anchor_image = query_region.crop(box)
            anchor_result = self.basic_retriever.retrieve(anchor_image, top_k=top_k)
            
            target_wsi_names = [res["entity"]["file"] for res in anchor_result]
            target_name = Counter(target_wsi_names).most_common(1)[0][0]
            target_wsi_subtypes = [res["entity"]["subtype"] for res in anchor_result]
            target_subtype = Counter(target_wsi_subtypes).most_common(1)[0][0]

            target_results = [res for res in anchor_result if res["entity"]["file"] == target_name and res["entity"]["subtype"] == target_subtype]
            target_anchor_retrieve_results[(x,y)] = [target_name, target_subtype, target_results] 
        if show:
            lens_valid_result = sum([len(target_anchor_retrieve_results[key][2]) for key in target_anchor_retrieve_results])
            print(f"2. Retrieved target anchor result len: {lens_valid_result}")

        # Step 3: Retrieval with valid anchors
        valid_anchor_retrieve_results = {}      # tuple for hash
        for target, valid_anchors in zip(target_anchor_retrieve_results, valid_anchor_pos):
            cur_name, cur_subtype, _ = target_anchor_retrieve_results[target]     # 解析上面的 target results 的字典
            valid_results = {}
            for (xv, yv) in valid_anchors:
                if xv == target[0] and yv == target[1]:
                    continue
                
                box = (xv-112, yv-112, xv+112, yv+112)
                valid_image = query_region.crop(box)
                valid_retrieve_result = self.basic_retriever.retrieve(valid_image, top_k=top_k)

                valid_retrieve_result = [res for res in valid_retrieve_result if res["entity"]["file"] == cur_name and res["entity"]["subtype"] == cur_subtype]

                valid_results[(xv, yv)] = valid_retrieve_result
            valid_anchor_retrieve_results[target] = valid_results
        if show:
            print(f"3. Retrieved valid anchor results.")

        # Step 4: Pointer Evaluation
        region_candidates = []
        for target_pos in target_anchor_retrieve_results:
            target_name, target_subtype, target_results = target_anchor_retrieve_results[target_pos]
            valid_results = valid_anchor_retrieve_results[target_pos]

            candidate = single_anchor_evaluation_milvus(query_region, target_pos, target_name, target_subtype, target_results, valid_results, mode=evaluation)
            region_candidates += candidate
        if show:
            print(f"4. Retrieved Region candidate numbers: {len(region_candidates)}")

        region_candidates = self.evaluate_region(query_region, region_candidates, top_k=5)
        if show:
            print(f"5. Retrieved Region numbers: {len(region_candidates)}")

        return region_candidates
    
    def evaluate_region(self, query_region, region_candidate, top_k=5):
        png_path_head = f"data/TCGA_thumbnail/{self.site}"
        query_embedding = self.encoder.encode_image(query_region)
        similarity_list = []
        for region_info in region_candidate:
            region_image = load_region_png(region_info, png_path_head=png_path_head)
            if not region_image:
                continue

            region_embedding = self.encoder.encode_image(region_image)

            similarity, _, _ = cos_sim_list(query_embedding, region_embedding)
            similarity_list.append((region_info, similarity))

        top_k_regions = sorted(similarity_list, key=lambda x: x[1], reverse=True)[:top_k]
        
        return [item for item in top_k_regions]
    





from src.modules.pointer_evaluation.single_anchor_evaluation import get_delta, find_min_variance_traversal, find_min_variance_BoC

def single_anchor_evaluation_milvus(query_region, target_pos, target_name, target_subtype, target_results, valid_results, mode="traversal"):
    region_candidate = []
    anchor_x, anchor_y = target_pos
    for res in target_results:
        res_pos = res["entity"]["position"]
        retrieve_x, retrieve_y = res_pos

        delta_list = []     # 保存检索结果的距离和角度变化
        for valid_pos in valid_results:
            valid_anchor_x, valid_anchor_y = valid_pos
            valid_retrieved_results = valid_results[valid_pos]
            for valid_anchor_res in valid_retrieved_results: 
                valid_x, valid_y = valid_anchor_res["entity"]["position"]

                retrieve_vec = (int(valid_x) - int(retrieve_x), int(retrieve_y) - int(valid_y))     # 检索结果构成的 pointer    
                target_vec = (valid_anchor_x - anchor_x, anchor_y - valid_anchor_y)                 # query region 中构成的 pointer

                delta_distance, delta_angle = get_delta(target_vec, retrieve_vec)
                if 0.75 < delta_distance < 1.5:
                    delta_list.append((delta_distance, delta_angle))
        
        if mode == "boc":
            combination = find_min_variance_BoC(delta_list) 
        else:
            combination, _ = find_min_variance_traversal(delta_list) 

        if combination and all(i is not None and i != [] for i in combination):     # 检查 combination 不为空
            width, height = query_region.size

            avg_distence = sum([i[0] for i in combination]) / len(combination)
            avg_angle = sum([i[1] for i in combination]) / len(combination)

            delta_x, delta_y = width // 2 - anchor_x, anchor_y - height // 2
            cos_theta, sin_theta = np.cos(avg_angle), np.sin(avg_angle)

            x_prime = int((delta_x * cos_theta - delta_y * sin_theta) * avg_distence)
            y_prime = int((delta_x * sin_theta + delta_y * cos_theta) * avg_distence)

            region = {
                "position": [
                    int(retrieve_x)+x_prime,
                    int(retrieve_y)-y_prime,
                ],
                "size": [
                    int(width * avg_distence),
                    int(height * avg_distence)
                ],
                "angle": avg_angle * 180 / math.pi,
                "wsi_name":target_name,
                "subtype":target_subtype,
            }

            region_candidate.append(region)
    
    return region_candidate
    







if __name__ == "__main__":
    from src.utils.basic.encoder import WSI_Image_UNI_Encoder, WSI_Image_test_Encoder
    from src.modules.retriever.basic_patch_retriever import Image2Image_Retriever_Milvus

    encoder = WSI_Image_UNI_Encoder()
    # encoder = WSI_Image_test_Encoder()

    site = "hematopoietic"
    basic_retriever = Image2Image_Retriever_Milvus(site, encoder)
    region_retriever = Image2Image_Region_Retriever(basic_retriever, encoder)

    query_info = {
        "position": [
            1500,
            1000
        ],
        "size": [
            1098,
            823
        ],
        "angle": -114,
        "wsi_name":"2fc6dcdf-91e2-4ba1-a80d-ed6fa6247c5b",
        "subtype":"THYM"
    }
    png_path_head = f"data/TCGA_thumbnail/hematopoietic"

    query_region = load_region_png(query_info, png_path_head=png_path_head)
    query_region.save("image/region_retrieval_TCGA/region_query.png")

    start = time.time()
    region_candidate = region_retriever.region_retrieval(query_region, show=True)  # 通过参数控制检索流程
    end = time.time()
    print(f"Total Retrieval Time: {end-start}")

    query_param = (
        query_info["wsi_name"],
        query_info["position"][0],
        query_info["position"][1],
        query_info["size"][0],
        query_info["size"][1],
        query_info["subtype"],
        query_info["angle"]
    )
    for index, (region_info, similarity) in enumerate(region_candidate):
        region_param = (
            region_info["wsi_name"],
            region_info["position"][0],
            region_info["position"][1],
            region_info["size"][0],
            region_info["size"][1],
            region_info["subtype"],
            region_info["angle"]
        )

        print(query_param)
        print(region_param)

        iou_score = region_retrieval_self_single_IoU(query_param, region_param)
        print(f"Res: {region_info} with Sim: {similarity} and IoU: {iou_score}.")
