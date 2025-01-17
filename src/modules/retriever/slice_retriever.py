import sys, time
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
from collections import Counter

from src.modules.retriever.basic_patch_retriever import Image2Image_Retriever_Qdrant
from src.modules.anchor_selection.fixed_selection import get_anchor_triple
from src.modules.anchor_selection.kmeas_clustering import kmeans_cluster_selection
from src.modules.anchor_selection.spectral_clustering import spectral_cluster_selection
from src.modules.anchor_selection.semantic_field_evolution import pre_select_anchor_evolution


"""
Slide Retrieval 的基本流程是基于 query region 选择一些 tessellation 上的 anchor 作为基本点，并各自出发原始的 patch retriever；
基于每个 target anchor 的检索结果，在结果最多的 WSI 和 level 即为最后的检索结果。
"""


class Image2Image_Slice_Retriever():
    def __init__(self, basic_retriever, encoder):
        if basic_retriever:
            self.basic_retriever = basic_retriever
        else:
            ValueError("Please select basic retriever.")
        self.encoder = encoder

    def slide_retrieval(self, query_region, preprocess="position", top_k=10, step=100, show=False):
        # Step 1: Choosing Anchor
        w, h = query_region.size
        if preprocess == "kmeans":
            target_anchor_pos = kmeans_cluster_selection(query_region, self.encoder, n=4, step=step)
        elif preprocess == "spectral":
            target_anchor_pos = spectral_cluster_selection(query_region, self.encoder, n=4, step=step)
        elif preprocess == "evolution":
            target_anchor_pos = pre_select_anchor_evolution(query_region, self.encoder, n=4, step=step, show=False)
        else:
            target_anchor_pos = get_anchor_triple(w, h)
        if show:
            print(f"1. Select anchor positions: {target_anchor_pos}")

        # Step 2: Main Retrieval with target anchors
        target_names = []
        for (x, y) in target_anchor_pos:
            box = (x, y, x+224, y+224)
            anchor_image = query_region.crop(box)
            anchor_result = self.basic_retriever.retrieve(anchor_image, top_k=top_k)
            
            target_wsi_names = [res.payload["wsi_name"] for res in anchor_result]
            target_name = Counter(target_wsi_names).most_common(1)[0][0]
            target_count = Counter(target_wsi_names).most_common(1)[0][1]
            target_names.append((target_name, target_count))
        if show:
            print(f"2. Retrieved target anchor result len: {len(target_names)}")

        # Step 3: 按照每个名字的排序进行输出
        target_names.sort(key=lambda x: x[1], reverse=True)
        sorted_names = [name for name, _ in target_names]

        return sorted_names


if __name__ == "__main__":
    from src.utils.open_wsi.wsi_loader import load_wsi_region
    from src.utils.basic.encoder import WSI_Image_UNI_Encoder, WSI_Image_test_Encoder

    # encoder = WSI_Image_UNI_Encoder()
    encoder = WSI_Image_test_Encoder()  # test
    database_path = "data/vector_database_TCGA_sample"
    basic_retriever = Image2Image_Retriever_Qdrant(encoder, database_path)
    region_retriever = Image2Image_Slice_Retriever(basic_retriever, encoder)

    wsi_name, x, y, w, h, level, angle = "241183-21.tiff", 16456, 9632, 1014, 1541, 2, 120
    query_url = f"http://mdi.hkust-gz.edu.cn/wsi/metaservice/api/region/openslide/{wsi_name}"
    _, query_region = load_wsi_region(query_url, x, y, w, h, level, angle)

    start = time.time()
    target_name = region_retriever.slide_retrieval(query_region, show=True)  # 通过参数控制检索流程
    end = time.time()
    print(f"Target Name: {target_name} | Total Retrieval Time: {end-start}")