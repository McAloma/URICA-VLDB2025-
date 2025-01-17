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
from baseline.thumbnail_retrieval import ImageAlignRetriever


class Extra_Retrieval_experiment():
    def __init__(self, wsi_file_path, query_materials_path, encoder):
        self.wsi_file_path = wsi_file_path
        self.encoder = encoder
        self.query_materials_path = query_materials_path

    def load_region(self, wsi_name, x, y, w, h, theta, level=None):        
        wsi_path = os.path.join(self.wsi_file_path, wsi_name)
        slide = OpenSlide(wsi_path)
        if level:
            target_level = level
        else:
            target_level = slide.level_count - 1
        
        ratio = slide.level_downsamples[target_level]

        w, h = int(w), int(h)
        canvas_size = int(math.sqrt(w**2 + h**2))  # square canvas
        canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))

        # true position with center point
        x = int((x - canvas_size // 2) * ratio)
        y = int((y - canvas_size // 2) * ratio)
        angle = theta

        psuedo_region = slide.read_region((x, y), target_level, (canvas_size, canvas_size)) .convert("RGB")
        canvas.paste(psuedo_region, (0, 0))
        rotated_canvas = canvas.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=False)

        final_crop_left = (rotated_canvas.width - w) // 2
        final_crop_top = (rotated_canvas.height - h) // 2
        region = rotated_canvas.crop(
            (final_crop_left, final_crop_top, final_crop_left + w, final_crop_top + h)
        )

        return region
    
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
    
    # def load_region_mask(self, region):
    #     name, x, y, w, h, level, angle = region
    #     wsi_path = os.path.join(self.wsi_file_path, name)
    #     slide = OpenSlide(wsi_path)

    #     ratio = slide.level_downsamples[-1] // slide.level_downsamples[level]
    #     x, y, w, h = int(x // ratio), int(y // ratio), int(w // ratio), int(h // ratio)

    #     mask_name = name.split(".")[0] + "_evaluation_mask.png"
    #     mask_path = os.path.join(self.wsi_file_path, "mask", mask_name)
        
    #     # 在加载图像前检查尺寸
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

    def score_region_mPD(self, query_region, retrieved_regions):
        # q_name, q_x, q_y, q_w, q_h, q_level, q_angle = query_region
        query_mask = self.load_region_mask(query_region)
        res = []
        for region in retrieved_regions:
            r_name, r_x, r_y, r_w, r_h, r_angle, _, _ = region
            wsi_name = r_name[:-4] + ".tif"
            wsi_path = os.path.join(self.wsi_file_path, wsi_name)
            slide = OpenSlide(wsi_path)

            # wsi_name, wsi_path = r_name, NotImplementedError
            # for dirpath, _, filenames in os.walk(self.wsi_file_path):
            #     if wsi_name in filenames:
            #         wsi_path = os.path.join(dirpath, wsi_name)
            #         break
            # if not wsi_path:
            #     print(f"Can not find the WSI file in {wsi_name}.")
            #     return
                
            # slide = OpenSlide(wsi_path) 
            level = slide.level_count - 1
            retrieved_mask = self.load_region_mask((wsi_name, r_x, r_y, r_w, r_h, level, r_angle))     

            if not retrieved_mask or 0 in query_mask.size or 0 in retrieved_mask.size:
                res.append(0)
            else:
                mPD_score = calculate_distribution_difference(query_mask, retrieved_mask)
                res.append(mPD_score)

        return res
    
    def calculate_at_k(self, scores, k):
        scores = sorted(scores, reverse=True)  # 从大到小排序
        scores = scores[:k] + [0] * (k - len(scores))  # 如果长度不足，补充 0
        return np.mean(scores)

    def main(self, args, folder_path):
        try:
            with open(self.query_materials_path, "r") as file:
                query_region_infos = json.load(file)
        except FileNotFoundError:
            print("File does not exist.")

        results = []
        for query_info in tqdm(query_region_infos):
            q_name = query_info["wsi_name"]
            qx, qy = query_info["position"]
            qw, qh = query_info["size"]
            qlevel = query_info["level"]
            qtheta = query_info["angle"]

            query_info_list = [q_name, qx, qy, qw, qh, qlevel, qtheta]
            query_region = self.load_region(q_name, qx, qy, qw, qh, qtheta, level=qlevel)
            query_embeddings = self.encoder.encode_image(query_region)

            start = time.time()
            retriever = ImageAlignRetriever(query_region, folder_path, encoder)
            align_retrieval_results = retriever.get_top_5_matches()
            end = time.time()

            sim_scores = []
            for result in align_retrieval_results:
                file_name, x, y, w, h, theta, _, _ = result
                file_name = file_name[:-4] + ".tif"
                retrieved_region = self.load_region(file_name, x, y, w, h, theta)
                retrieved_embeddings = self.encoder.encode_image(retrieved_region)
                cos_sim, _, _ = cos_sim_list(query_embeddings, retrieved_embeddings)
                sim_scores.append(cos_sim)

            mPD_scores = self.score_region_mPD(query_info_list, align_retrieval_results)

            result = {
                "mPD_at_1": self.calculate_at_k(mPD_scores, 1),
                "sim_at_1": self.calculate_at_k(sim_scores, 1),
                "mPD_at_3": self.calculate_at_k(mPD_scores, 3),
                "sim_at_3": self.calculate_at_k(sim_scores, 3),
                "mPD_at_5": self.calculate_at_k(mPD_scores, 5),
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
        self.save_results_to_file({f"Extra_region retrieval experiment with align method, date={current_date}": final_resuls},
                            filename="experiment/results/extra_retrieval_align_results.txt")
        
    def save_results_to_file(self, results, filename="./results/extra_retrieval_align_results.txt"):
        with open(filename, "a+") as f:
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Experiment Date: {current_date}\n")
            f.write("-" * 50 + "\n")  # 分隔线，增加可读性

            for param_set, metrics in results.items():
                f.write(f"Parameters: {param_set}\n")
                f.write(f"@1 mPD Avg: {metrics['mPD_at_1']:.4f}, SIM Avg: {metrics['sim_at_1']:.4f}\n")
                f.write(f"@3 mPD Avg: {metrics['mPD_at_3']:.4f}, SIM Avg: {metrics['sim_at_3']:.4f}\n")
                f.write(f"@5 mPD Avg: {metrics['mPD_at_5']:.4f}, SIM Avg: {metrics['sim_at_5']:.4f}\n")
                f.write(f"Running Times: {metrics['time']:.4f}\n")
                f.write("\n")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ----------- pathes -----------
    parser.add_argument('--wsi_file_path', type=str, default="data/Camelyon")
    args = parser.parse_args() 

    from src.utils.basic.encoder import WSI_Image_UNI_Encoder, WSI_Image_test_Encoder

    encoder = WSI_Image_UNI_Encoder()
    # encoder = WSI_Image_test_Encoder()    # for test

    source_materials_path = "experiment/materials/query_source.json"
    region_materials_path = "experiment/materials/query_region_infos_camelyon.json"
    # region_materials_path = "experiment/materials/query_region_infos_camelyon_sample.json"

    exp = Extra_Retrieval_experiment(args.wsi_file_path, 
                                     region_materials_path,
                                     encoder)
    
    folder_path = "data/Camelyon_thumbnail_cache"
    exp.main(args, folder_path)

