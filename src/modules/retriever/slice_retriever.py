import time
from collections import Counter

from src.modules.retriever.basic_patch_retriever import Image2Image_Retriever_Qdrant
from src.modules.anchor_selection.fixed_selection import get_anchor_triple
from src.modules.anchor_selection.kmeas_clustering import kmeans_cluster_selection
from src.modules.anchor_selection.spectral_clustering import spectral_cluster_selection


class Image2Image_Slice_Retriever():
    def __init__(self, basic_retriever, encoder):
        if basic_retriever:
            self.basic_retriever = basic_retriever
        else:
            ValueError("Please select basic retriever.")
        self.encoder = encoder

    def slide_retrieval(self, query_region, preprocess="position", top_k=10, step=100, show=False):
        w, h = query_region.size
        if preprocess == "kmeans":
            target_anchor_pos = kmeans_cluster_selection(query_region, self.encoder, n=4, step=step)
        elif preprocess == "spectral":
            target_anchor_pos = spectral_cluster_selection(query_region, self.encoder, n=4, step=step)
        else:
            target_anchor_pos = get_anchor_triple(w, h)
        if show:
            print(f"1. Select anchor positions: {target_anchor_pos}")

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

        target_names.sort(key=lambda x: x[1], reverse=True)
        sorted_names = [name for name, _ in target_names]

        return sorted_names
