import os, argparse, math, cv2, time, json
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
from openslide import OpenSlide

from src.utils.evaluator.cos_sim import cos_sim_list
from src.modules.retriever.basic_patch_retriever import Image2Image_Retriever_Qdrant
from baseline.adjacent_retrieval import Adjacent_Region_Retriever


class Inter_Retrieval_experiment():
    def __init__(self, wsi_file_path, query_materials_path, encoder):
        self.wsi_file_path = wsi_file_path
        self.encoder = encoder
        self.query_materials_path = query_materials_path


    def load_region(self, region_info):
        wsi_dir = region_info["wsi_dir"]
        wsi_name = region_info["wsi_name"]
        wsi_path = os.path.join(self.wsi_file_path, wsi_dir, wsi_name)
        slide = OpenSlide(wsi_path)

        target_level = region_info["level"]
        ratio = slide.level_downsamples[target_level]

        w, h = int(region_info["size"][0]), int(region_info["size"][1])
        canvas_size = int(math.sqrt(w**2 + h**2)) 
        canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))

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
    

    def score_region_sim(self, query_region, target_wsi_name, region_candidate):
        query_embedding = self.encoder.encode_image(query_region)

        wsi_path = ""
        for dirpath, dirnames, filenames in os.walk(self.wsi_file_path):
            if target_wsi_name in filenames:
                dirnames = dirpath.split("/")
                wsi_path = os.path.join(self.wsi_file_path,  dirnames[-1], target_wsi_name)
                break
        if not wsi_path:  
            print(f"Can not find the WSI file in {target_wsi_name}.")
            return 
        slide = OpenSlide(wsi_path)

        sim_scores = []
        for candidate in region_candidate:
            x, y, w, h, level, angle = candidate
            retrieved_region = slide.read_region((x, y), level, (w, h)).convert("RGB")
            retrieved_embeddings = self.encoder.encode_image(retrieved_region)
            sim, _, _ = cos_sim_list(query_embedding, retrieved_embeddings)
            sim_scores.append(sim)

        return sim_scores


    def score_region_IoU(self, query_region, target_wsi_name, region_candidate):
        q_name, q_x, q_y, q_w, q_h, q_level, q_angle = query_region
        rect_query = ((q_x, q_y), (q_w, q_h), q_angle)

        res = []
        for region in region_candidate:
            r_x, r_y, r_w, r_h, r_level, r_angle = region

            if q_name != target_wsi_name or str(q_level) != str(r_level):
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
        scores = sorted(scores, reverse=True)  
        scores = scores[:k] + [0] * (k - len(scores)) 
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
            target_wsi_name, region_candidate, _ = region_retriever.retrieve(query_region)
            end = time.time()

            query_info_list = [query_info["wsi_name"], *query_info["position"], *query_info["size"], query_info["level"], query_info["angle"]]
            sim_scores = [result[1] for result in region_candidate]

            sim_scores = self.score_region_sim(query_region, target_wsi_name, region_candidate)
            iou_scores = self.score_region_IoU(query_info_list, target_wsi_name, region_candidate)

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

        for result in results:
            for key in keys:
                final_resuls[key] += result[key]

        num_results = len(results)
        for key in final_resuls:
            final_resuls[key] /= num_results

        
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.save_results_to_file({f"Regino Retrieval with adjacent method, date={current_date}": final_resuls},
                            filename="experiment/results/self_retrieval_adjacent_results.txt")
        
    def save_results_to_file(self, results, filename="./results/self_retrieval_adjacent_results.txt"):
        with open(filename, "a+") as f:
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Experiment Date: {current_date}\n")
            f.write("-" * 50 + "\n")

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
    args = parser.parse_args() 
    
    args.database_path = "data/vector_database_TCGA"  

    from src.utils.basic.encoder import WSI_Image_UNI_Encoder, WSI_Image_test_Encoder
    encoder = WSI_Image_UNI_Encoder()
    # encoder = WSI_Image_test_Encoder()  

    basic_retriever = Image2Image_Retriever_Qdrant(encoder, args.database_path)
    region_retriever = Adjacent_Region_Retriever(basic_retriever, encoder)

    source_materials_path = "experiment/materials/query_source.json"
    region_materials_path = "experiment/materials/query_region_infos.json"

    exp = Inter_Retrieval_experiment(args.wsi_file_path, 
                                     region_materials_path,
                                     encoder)
    
    exp.main(args, region_retriever)

