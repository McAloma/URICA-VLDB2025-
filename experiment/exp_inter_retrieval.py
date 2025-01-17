import os, sys, argparse, random, math, cv2, time, json
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
from openslide import OpenSlide
from concurrent.futures import ThreadPoolExecutor

from src.utils.open_wsi.backgound import load_wsi_thumbnail, get_region_background_ratio
from src.utils.basic.encoder import WSI_Image_UNI_Encoder
from src.modules.retriever.basic_patch_retriever import Image2Image_Retriever_Qdrant
from src.modules.retriever.region_retrieval_TCGA import Image2Image_Region_Retriever


def sample_query_source(target_file, materials_path, sample_n=50):
    if os.path.exists(materials_path):
        with open(materials_path, "r") as file:
            query_source = json.load(file)
        print(f"Data loaded from {materials_path}.")
    else:
        wsi_dirs = os.listdir(target_file)
        query_source = random.sample(wsi_dirs, sample_n)
        with open(materials_path, "w") as file:
            json.dump(query_source, file, indent=4)
        print(f"Sampled data saved to {materials_path}.")
    return query_source

class DataPreparationProcess:
    def __init__(self, wsi_file_path, wsi_dirs, region_materials_path):
        self.wsi_file_path = wsi_file_path
        self.wsi_dirs = wsi_dirs
        self.materials_path = region_materials_path
        self.max_retries = 5

    def generate_bounded_rotated_box(self, width, height):
        margin = 0.25
        cx = random.uniform(width * margin, width * (1 - margin))
        cy = random.uniform(height * margin, height * (1 - margin))
        max_w = width / 2
        max_h = height / 2
        w = random.uniform(448, min(max_w, 2000))
        h = random.uniform(448, min(max_h, 2000))
        theta = random.uniform(-180, 180)

        corners = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), 
                   (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]
        rotated_corners = [(cx + (x - cx) * math.cos(math.radians(theta)) - 
                            (y - cy) * math.sin(math.radians(theta)), 
                            cy + (x - cx) * math.sin(math.radians(theta)) + 
                            (y - cy) * math.cos(math.radians(theta))) 
                           for x, y in corners]

        min_x = min(p[0] for p in rotated_corners)
        max_x = max(p[0] for p in rotated_corners)
        min_y = min(p[1] for p in rotated_corners)
        max_y = max(p[1] for p in rotated_corners)

        if min_x < 0 or max_x > width or min_y < 0 or max_y > height:
            return self.generate_bounded_rotated_box(width, height)
        return cx, cy, w, h, theta

    def process_wsi(self, wsi_dir, n):
        file_path = os.path.join(self.wsi_file_path, wsi_dir)
        subfile_names = os.listdir(file_path)
        wsi_path = next((os.path.join(file_path, f) for f in subfile_names 
                         if f.lower().endswith(('.svs', '.tiff'))), None)
        if not wsi_path:
            return []
        
        slide = OpenSlide(wsi_path)
        _, background, num_level = load_wsi_thumbnail(slide)
        query_region_infos = []
        
        for level in range(2, num_level):
            width, height = slide.level_dimensions[level]
            for _ in range(n):
                retry_count = 0
                white_pixel_ratio = 1.0

                while white_pixel_ratio > 0.8 and retry_count < self.max_retries:
                    x, y, w, h, angle = self.generate_bounded_rotated_box(width, height)
                    x, y, w, h, angle = int(x), int(y), int(w), int(h), int(angle)
                    region_info = {"position": (x, y), "size": (w, h), 
                                   "level": level, "angle": angle}
                    _, white_pixel_ratio = get_region_background_ratio(slide, background, region_info)
                    retry_count += 1

                if retry_count < self.max_retries:
                    region_info.update({"wsi_dir": wsi_dir, "wsi_name": os.path.basename(wsi_path)})
                    query_region_infos.append(region_info)

        return query_region_infos

    def generation_query_region(self, n=3):
        if os.path.exists(self.materials_path):
            print("Query region info file exists.")
            return

        query_region_infos = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_wsi, wsi_dir, n) for wsi_dir in self.wsi_dirs]
            with tqdm(total=len(futures), desc="Processing WSIs") as pbar:
                for future in futures:
                    query_region_infos.extend(future.result())
                    pbar.update(1)  

        with open(self.materials_path, "w") as file:
            json.dump(query_region_infos, file, indent=4)
        print("Saved query region infos to", self.materials_path)



class Inter_Retrieval_experiment():
    def __init__(self, wsi_file_path, query_materials_path, encoder):
        self.wsi_file_path = wsi_file_path
        self.encoder = encoder
        self.query_materials_path = query_materials_path

    def cosine_similarity(self, list1, list2):
        vec1 = np.array(list1)
        vec2 = np.array(list2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0, norm1, norm2
        
        return dot_product / (norm1 * norm2), norm1, norm2

    def load_region(self, region_info):
        wsi_dir = region_info["wsi_dir"]
        wsi_name = region_info["wsi_name"]
        wsi_path = os.path.join(self.wsi_file_path, wsi_dir, wsi_name)
        slide = OpenSlide(wsi_path)

        target_level = region_info["level"]
        ratio = slide.level_downsamples[target_level]

        w, h = int(region_info["size"][0]), int(region_info["size"][1])
        canvas_size = int(math.sqrt(w**2 + h**2))  # square canvas
        canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))

        # true position with center point
        x = int((region_info["position"][0] - canvas_size // 2) * ratio)
        y = int((region_info["position"][1] - canvas_size // 2) * ratio)
        angle = region_info["angle"]

        psuedo_region = slide.read_region((x, y), target_level, (canvas_size, canvas_size)) 
        canvas.paste(psuedo_region, (0, 0))
        rotated_canvas = canvas.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=False)

        final_crop_left = (rotated_canvas.width - w) // 2
        final_crop_top = (rotated_canvas.height - h) // 2
        region = rotated_canvas.crop(
            (final_crop_left, final_crop_top, final_crop_left + w, final_crop_top + h)
        )

        return region

    def score_region_IoU(self, query_region, region_candidate):
        q_name, q_x, q_y, q_w, q_h, q_level, q_angle = query_region
        rect_query = ((q_x, q_y), (q_w, q_h), q_angle)

        res = []
        for region, _ in region_candidate:
            r_name, r_x, r_y, r_w, r_h, r_level, r_angle = region

            if q_name != r_name or str(q_level) != str(r_level):
                res.append(0)
            else:
                rect_region = ((r_x, r_y), (r_w, r_h), r_angle)
                inter_type, inter_pts = cv2.rotatedRectangleIntersection(rect_query, rect_region)

                if inter_type > 0 and inter_pts is not None:
                    inter_area = cv2.contourArea(inter_pts)
                else:
                    inter_area = 0.0

                area1 = q_w * q_h
                area2 = r_w * r_h
                union_area = area1 + area2 - inter_area
                iou_score = inter_area / union_area

                res.append(iou_score)

        return res
    
    def calculate_at_k(self, scores, k):
        scores = sorted(scores, reverse=True)  # 从大到小排序
        scores = scores[:k] + [0] * (k - len(scores))  # 如果长度不足，补充 0
        return np.mean(scores)

    def main(self, args, region_retriever):
        try:
            with open(self.query_materials_path, "r") as file:
                query_region_infos = json.load(file)
        except FileNotFoundError:
            print("File does not exist.")

        results = []
        for query_info in tqdm(query_region_infos):
            query_region = self.load_region(query_info)

            start = time.time()
            region_candidate = region_retriever.region_retrieval(query_region, 
                                                                preprocess=args.preprocess, 
                                                                evaluation=args.evaluation, 
                                                                step=args.step
                                                                )
            end = time.time()

            query_info_list = [query_info["wsi_name"], *query_info["position"], *query_info["size"], query_info["level"], query_info["angle"]]
            sim_scores = [result[1] for result in region_candidate]
            iou_scores = self.score_region_IoU(query_info_list, region_candidate)

            # print("Query: ", query_info_list)
            # for i, (region, score) in enumerate(region_candidate):
                # print(f"Candidate {i}: {region}, Score: {score}")
            # print("Score", iou_scores, sim_scores, end-start)

            result = {
                "iou_at_1": self.calculate_at_k(iou_scores, 1),
                "sim_at_1": self.calculate_at_k(sim_scores, 1),
                "iou_at_3": self.calculate_at_k(iou_scores, 3),
                "sim_at_3": self.calculate_at_k(sim_scores, 3),
                "iou_at_5": self.calculate_at_k(iou_scores, 5),
                "sim_at_5": self.calculate_at_k(sim_scores, 5),
                "time": end - start
            }
            results.append(result)

        keys = results[0].keys()
        final_resuls = {key: 0 for key in keys}

        # 遍历列表中的每个字典，累加每个键的值
        for result in results:
            for key in keys:
                final_resuls[key] += result[key]

        # 计算平均值
        num_results = len(results)
        for key in final_resuls:
            final_resuls[key] /= num_results

        
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.save_results_to_file({f"preprocess={args.preprocess}, evaluation={args.evaluation}, date={current_date}": final_resuls},
                            filename="experiment/results/self_retrieval_results.txt")
        
    def save_results_to_file(self, results, filename="./results/self_retrieval_results.txt"):
        with open(filename, "a+") as f:
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Experiment Date: {current_date}\n")
            f.write("-" * 50 + "\n")  # 分隔线，增加可读性

            for param_set, metrics in results.items():
                f.write(f"Parameters: {param_set}\n")
                f.write(f"@1 IOU Avg: {metrics['iou_at_1']:.4f}, SIM Avg: {metrics['sim_at_1']:.4f}\n")
                f.write(f"@3 IOU Avg: {metrics['iou_at_3']:.4f}, SIM Avg: {metrics['sim_at_3']:.4f}\n")
                f.write(f"@5 IOU Avg: {metrics['iou_at_5']:.4f}, SIM Avg: {metrics['sim_at_5']:.4f}\n")
                f.write(f"Running Times: {metrics['time']:.4f}\n")
                f.write("\n")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ----------- pathes -----------
    parser.add_argument('--wsi_file_path', type=str, default="data/TCGA_BRCA")
    parser.add_argument('--target_file', type=str, default="data/metadata_embedding_TCGA")
    parser.add_argument('--database_path', type=str, default="data/vector_database_TCGA_sample")
    #  ----------- Parameter -----------
    parser.add_argument('--step', type=int, default=100)
    #  ----------- Module -----------
    parser.add_argument('--preprocess', type=str, default="spectral")
    parser.add_argument('--evaluation', type=str, default="boc")
    args = parser.parse_args() 
    
    args.database_path = "data/vector_database_TCGA"   # 正式实验

    from src.utils.basic.encoder import WSI_Image_UNI_Encoder, WSI_Image_test_Encoder

    encoder = WSI_Image_UNI_Encoder()
    # encoder = WSI_Image_test_Encoder()    # for test

    basic_retriever = Image2Image_Retriever_Qdrant(encoder, args.database_path)
    region_retriever = Image2Image_Region_Retriever(basic_retriever, encoder)

    source_materials_path = "experiment/materials/query_source.json"
    # region_materials_path = "experiment/materials/query_region_infos.json"
    region_materials_path = "experiment/materials/query_region_infos_sample.json"  # small set for exp

    query_source = sample_query_source(args.target_file, source_materials_path, sample_n=50)
    # query_source = [
    #     "b545c7e6-8318-47e1-8ed4-62a5883a1d1c",
    # ]
    data_prepare = DataPreparationProcess(args.wsi_file_path, 
                                          query_source, 
                                          region_materials_path)
    data_prepare.generation_query_region(n=3)

    exp = Inter_Retrieval_experiment(args.wsi_file_path, 
                                     data_prepare.materials_path,
                                     encoder)
    
    # exp.main(args, region_retriever)


    # ---------------------------- Ablation Experiment ----------------------------

    # for preprocess in ["spectral", "kmeans", "others"]:
    # #     for evaluation in ["boc", "traversal"]:
    for preprocess in ["evolution"]:
        for evaluation in ["boc", "traversal"]:
                args.preprocess = preprocess
                args.evaluation = evaluation
                exp.main(args, region_retriever)