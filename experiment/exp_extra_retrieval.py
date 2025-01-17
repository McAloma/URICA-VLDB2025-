import os, sys, argparse, random, math, cv2, time, json
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
from datetime import datetime
import numpy as np
from PIL import Image
from PIL.Image import Resampling
from tqdm import tqdm
from openslide import OpenSlide
from concurrent.futures import ThreadPoolExecutor

from src.utils.evaluator.cos_sim import cos_sim_list
from src.utils.evaluator.mask import calculate_distribution_difference
from src.utils.open_wsi.backgound import load_wsi_thumbnail, get_region_background_ratio
from src.utils.basic.encoder import WSI_Image_UNI_Encoder
from src.modules.retriever.basic_patch_retriever import Image2Image_Retriever_Qdrant
from src.modules.retriever.region_retrieval_Camelyon import Image2Image_Region_Retriever



class DataPreparationProcess:
    def __init__(self, wsi_file_path, materials_path, wsi_names):
        self.wsi_file_path = wsi_file_path
        self.wsi_names = wsi_names
        self.materials_path = materials_path
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

    def process_wsi(self, wsi_name, n):
        wsi_path = os.path.join(self.wsi_file_path, wsi_name)      
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
                    region_info.update({"wsi_name": wsi_name})
                    query_region_infos.append(region_info)

        return query_region_infos

    def generation_query_region(self, n=3):
        if os.path.exists(self.materials_path):
            print("Query region info file exists.")
            return

        query_region_infos = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_wsi, wsi_name, n) for wsi_name in self.wsi_names]
            with tqdm(total=len(futures), desc="Processing WSIs") as pbar:
                for future in futures:
                    query_region_infos.extend(future.result())
                    pbar.update(1)  

        with open(self.materials_path, "w") as file:
            json.dump(query_region_infos, file, indent=4)
        print("Saved query region infos to", self.materials_path)


class Extra_Retrieval_experiment():
    def __init__(self, wsi_file_path, query_materials_path, encoder):
        self.wsi_file_path = wsi_file_path
        self.encoder = encoder
        self.query_materials_path = query_materials_path

    def load_region(self, region_info):
        wsi_name = region_info["wsi_name"]
        wsi_path = os.path.join(self.wsi_file_path, wsi_name)
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
        rotated_canvas = canvas.rotate(-angle, resample=Resampling.BICUBIC, expand=False)

        final_crop_left = (rotated_canvas.width - w) // 2
        final_crop_top = (rotated_canvas.height - h) // 2
        region = rotated_canvas.crop(
            (final_crop_left, final_crop_top, final_crop_left + w, final_crop_top + h)
        )

        return region
    
    # def load_region_mask(self, region):
    #     name, x, y, w, h, level, angle = region
    #     wsi_path = os.path.join(self.wsi_file_path, name)
    #     slide = OpenSlide(wsi_path)

    #     ratio = slide.level_downsamples[-1] // slide.level_downsamples[level]
    #     x, y, w, h = int(x // ratio), int(y // ratio), int(w // ratio), int(h // ratio)

    #     mask_name = name.split(".")[0] + "_evaluation_mask.png"
    #     mask_path = os.path.join(self.wsi_file_path, "mask", mask_name)
    #     with Image.open(mask_path) as img:
    #         if img.size[0] * img.size[1] > Image.MAX_IMAGE_PIXELS:
    #             print(f"警告: {mask_name} 图像过大，无法加载。")
    #             return None  # 或者选择其他方式处理（比如跳过、缩小等）
        
        
    #     mask = Image.open(mask_path).convert('L')
    #     canvas_size = int(math.sqrt(w**2 + h**2))

    #     # 检查 canvas_size 是否超出安全限制
    #     MAX_CANVAS_SIZE = 10000  # 可根据需要调整限制
    #     if canvas_size > MAX_CANVAS_SIZE:
    #         print(f"警告: canvas_size ({canvas_size}) 超出允许范围，调整为 {MAX_CANVAS_SIZE}.")
    #         canvas_size = MAX_CANVAS_SIZE

    #     canvas = Image.new("1", (canvas_size, canvas_size), 255)
    #     cropped = mask.crop((x-canvas_size//2, y-canvas_size//2, x+canvas_size//2, y+canvas_size//2))
    #     canvas.paste(cropped, (0, 0))
    #     rotated_canvas = canvas.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=False)

    #     final_crop_left = (rotated_canvas.width - w) // 2
    #     final_crop_top = (rotated_canvas.height - h) // 2
    #     region_mask = rotated_canvas.crop(
    #         (final_crop_left, final_crop_top, final_crop_left + w, final_crop_top + h)
    #     )
        
    #     return region_mask

    def load_region_mask(self, region):
        name, x, y, w, h, level, angle = region
        wsi_path = os.path.join(self.wsi_file_path, name)
        slide = OpenSlide(wsi_path)

        # 这里的 mask 不一定在最后一层，先要通过 mask 尺寸获得目标 level
        mask_name = name.split(".")[0] + "_evaluation_mask.png"
        mask_path = os.path.join(self.wsi_file_path, "mask", mask_name)
        mask = Image.open(mask_path).convert('L')
        mask_width, mask_height = mask.size

        target_level = -1
        for level, (width, height) in enumerate(slide.level_dimensions):
            if (width, height) == (mask_width, mask_height):
                target_level = level
                break
        
        ratio = slide.level_downsamples[target_level] / slide.level_downsamples[level]
        x, y, w, h = int(x // ratio), int(y // ratio), int(w // ratio), int(h // ratio)

        canvas_size = int(math.sqrt(w**2 + h**2)) 
        canvas = Image.new("1", (canvas_size, canvas_size), 255)
        cropped = mask.crop((x-canvas_size//2, y-canvas_size//2, x+canvas_size//2, y+canvas_size//2))
        canvas.paste(cropped, (0, 0))
        rotated_canvas = canvas.rotate(-angle, resample=Image.BICUBIC, expand=False)

        final_crop_left = (rotated_canvas.width - w) // 2
        final_crop_top = (rotated_canvas.height - h) // 2
        region_mask = rotated_canvas.crop(
            (final_crop_left, final_crop_top, final_crop_left + w, final_crop_top + h)
        )
        
        return region_mask

    def score_region_mPD(self, query_region, retrieved_regions):
        query_mask = self.load_region_mask(query_region)
        res = []
        for region, _ in retrieved_regions:
            retrieved_mask = self.load_region_mask(region)
            mPD_score = calculate_distribution_difference(query_mask, retrieved_mask)
            res.append(mPD_score)

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
            iou_scores = self.score_region_mPD(query_info_list, region_candidate)

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
        self.save_results_to_file({f"preprocess={args.preprocess}, verification={args.evaluation}, date={current_date}": final_resuls},
                            filename="experiment/results/extra_retrieval_results.txt")
        
    def save_results_to_file(self, results, filename="./results/extra_retrieval_results.txt"):
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
    parser.add_argument('--wsi_file_path', type=str, default="data/Camelyon")
    parser.add_argument('--target_file', type=str, default="data/metadata_embedding_Camelyon_100")
    parser.add_argument('--database_path', type=str, default="data/vector_database_Camelyon")
    #  ----------- Parameter -----------
    parser.add_argument('--step', type=int, default=100)
    #  ----------- Module -----------
    parser.add_argument('--preprocess', type=str, default="spectral")
    parser.add_argument('--evaluation', type=str, default="boc")
    args = parser.parse_args() 
    
    args.database_path = "data/vector_database_Camelyon"   # 正式实验

    encoder = WSI_Image_UNI_Encoder()
    basic_retriever = Image2Image_Retriever_Qdrant(encoder, args.database_path)
    region_retriever = Image2Image_Region_Retriever(basic_retriever, encoder)

    materials_path = "experiment/materials/query_region_infos_camelyon.json"
    # materials_path = "experiment/materials/query_region_infos_camelyon_sample.json"

    query_source = [
        "tumor_016.tif", "tumor_017.tif", "tumor_018.tif", "tumor_019.tif", "tumor_020.tif",  
    ]
    data_prepare = DataPreparationProcess(args.wsi_file_path, materials_path, query_source)
    data_prepare.generation_query_region(n=1)
    exp = Extra_Retrieval_experiment(args.wsi_file_path, 
                                     data_prepare.materials_path,
                                     encoder)
    
    # exp.main(args, region_retriever)


    # ---------------------------- Ablation Experiment ----------------------------

    for preprocess in ["spectral", "kmeans", "evolution", "others"]:
    # for preprocess in ["evolution", "others"]:
        for evaluation in ["boc", "traversal"]:
                args.preprocess = preprocess
                args.evaluation = evaluation
                exp.main(args, region_retriever)