import sys, os, time
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
from collections import Counter
from qdrant_client.http.models import Filter, FieldCondition

from src.utils.evaluator.cos_sim import cos_sim_list
from src.utils.open_wsi.wsi_loader import load_region_came
from src.modules.anchor_selection.fixed_selection import get_anchor_triple
from src.modules.anchor_selection.kmeas_clustering import kmeans_cluster_selection
from src.modules.anchor_selection.spectral_clustering import spectral_cluster_selection
from src.modules.anchor_selection.semantic_field_evolution import pre_select_anchor_evolution
from src.modules.pointer_evaluation.single_anchor_evaluation import single_anchor_evaluation



# Region Retrieval 的基本流程是基于 query region 选择一些 tessellation 上的 anchor 作为基本点，并各自出发原始的 patch retriever；
# 基于每个 target anchor 的检索结果，在结果最多的 WSI 和 level 上使用其他的 valid anchor 再次触发检索获得 valid point；
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
        self.wsi_file_path = "data/Camelyon"
    
    def valid_positions_basic_retrieval(self, query_region, valid_anchor_pos, target_name, target_level, top_k=20):
        filter_conditions = Filter(
            must=[
                FieldCondition(
                    key="wsi_name",
                    match={"value": target_name}
                ),
                FieldCondition(
                    key="level",
                    match={"value": target_level}
                ),
            ])
        
        valid_x, valid_y = valid_anchor_pos
        box = (valid_x, valid_y, valid_x+224, valid_y+224)      # 左上角坐标
        valid_patch = query_region.crop(box)
        valid_retrieve_result = self.basic_retriever.retrieve(valid_patch, filter_conditions, top_k=top_k)
        
        return valid_retrieve_result
    
    def evaluate_region(self, query_region, region_candidate, top_k=5):
        query_embedding = self.encoder.encode_image(query_region)
        similarity_list = []
        for region in region_candidate:
            
            region_info = {
                "wsi_name": region[0],
                "position": (region[1], region[2]),
                "size": (region[3], region[4]),
                "level": region[5],
                "angle": region[6]
            }

            # NOTE：需要获得 WSI 文件的原始文件名

            region_image = load_region_came(region_info, self.wsi_file_path)
   
            region_embedding = self.encoder.encode_image(region_image)
            similarity, _, _ = cos_sim_list(query_embedding, region_embedding)
            similarity_list.append((region, similarity))

        top_k_regions = sorted(similarity_list, key=lambda x: x[1], reverse=True)[:top_k]
        
        return [item for item in top_k_regions]


    def region_retrieval(self, query_region, preprocess="spectral", evaluation="boc", top_k=10, step=100, show=False):
        # Step 1: Choosing Anchor
        w, h = query_region.size
        if preprocess == "kmeans":
            target_anchor_pos = kmeans_cluster_selection(query_region, self.encoder, n=4, step=step)
            valid_anchor_pos = [target_anchor_pos.copy() for _ in range(len(target_anchor_pos))]
        elif preprocess == "spectral":
            target_anchor_pos = spectral_cluster_selection(query_region, self.encoder, n=4, step=step)
            valid_anchor_pos = [target_anchor_pos.copy() for _ in range(len(target_anchor_pos))]
        elif preprocess == "evolution":
            target_anchor_pos = pre_select_anchor_evolution(query_region, self.encoder, n=4, step=step, show=False)
            valid_anchor_pos = [target_anchor_pos.copy() for _ in range(len(target_anchor_pos))]
        else:
            target_anchor_pos = get_anchor_triple(w, h)
            valid_anchor_pos = [target_anchor_pos.copy() for _ in range(len(target_anchor_pos))]
        if show:
            print(f"1. Select anchor positions: {target_anchor_pos}")

        # Step 2: Main Retrieval with target anchors
        target_anchor_retrieve_results = {}     # tuple for hash
        for (x, y) in target_anchor_pos:
            box = (x, y, x+224, y+224)
            anchor_image = query_region.crop(box)
            anchor_result = self.basic_retriever.retrieve(anchor_image, top_k=top_k)
            
            target_wsi_names = [res.payload["wsi_name"] for res in anchor_result]
            target_name = Counter(target_wsi_names).most_common(1)[0][0]
            target_wsi_levels = [res.payload["level"] for res in anchor_result]
            target_level = Counter(target_wsi_levels).most_common(1)[0][0]
            target_results = [res for res in anchor_result if res.payload["wsi_name"] == target_name and res.payload["level"] == target_level]

            target_anchor_retrieve_results[(x,y)] = [target_name, target_level, target_results]  # 每个 target anchor 都有一个 target level
        if show:
            lens_valid_result = sum([len(target_anchor_retrieve_results[key][2]) for key in target_anchor_retrieve_results])
            print(f"2. Retrieved target anchor result len: {lens_valid_result}")

        # Step 3: Retrieval with valid anchors
        valid_anchor_retrieve_results = {}      # tuple for hash
        for target, valid_anchors in zip(target_anchor_retrieve_results, valid_anchor_pos):
            cur_name, cur_level, _ = target_anchor_retrieve_results[target]     # 解析上面的 target results 的字典
            valid_results = {}
            for (xv, yv) in valid_anchors:
                if xv == target[0] and yv == target[1]:
                    continue
                valid_retrieve_result = self.valid_positions_basic_retrieval(query_region, (xv, yv), cur_name, cur_level) # 通过基础检索获取验证点
                valid_results[(xv, yv)] = valid_retrieve_result
            valid_anchor_retrieve_results[target] = valid_results
        if show:
            print(f"3. Retrieved valid anchor results.")

        # Step 4: Pointer Evaluation
        region_candidates = []
        for target_pos in target_anchor_retrieve_results:
            target_name, target_level, target_results = target_anchor_retrieve_results[target_pos]
            valid_results = valid_anchor_retrieve_results[target_pos]

            candidate = single_anchor_evaluation(query_region, target_pos, target_name, target_level, target_results, valid_results, mode=evaluation)
            region_candidates += candidate
        if show:
            print(f"4. Retrieved Region candidate numbers: {len(region_candidates)}")

        region_candidates = self.evaluate_region(query_region, region_candidates, top_k=5)

        return region_candidates
    


if __name__ == "__main__":
    from src.utils.basic.encoder import WSI_Image_UNI_Encoder, WSI_Image_test_Encoder
    from src.modules.retriever.basic_patch_retriever import Image2Image_Retriever_Qdrant

    # encoder = WSI_Image_UNI_Encoder()
    encoder = WSI_Image_test_Encoder()
    database_path = "data/vector_database_Camelyon"
    basic_retriever = Image2Image_Retriever_Qdrant(encoder, database_path)
    region_retriever = Image2Image_Region_Retriever(basic_retriever, encoder)

    query_info =     {
        "position": [
            13648,
            27050
        ],
        "size": [
            519,
            1886
        ],
        "level": 2,
        "angle": -3,
        "wsi_name": "tumor_016.tif"
    }

    query_region = load_region_came(query_info, wsis_path="data/Camelyon")
    query_region.save("image/region_retrieval/region_query.png")

    start = time.time()
    region_candidate = region_retriever.region_retrieval(query_region, show=True)  # 通过参数控制检索流程
    end = time.time()
    print(f"Total Retrieval Time: {end-start}")

    for index, (region, similarity) in enumerate(region_candidate):
        print(f"Res: {region} with Sim: {similarity}.")