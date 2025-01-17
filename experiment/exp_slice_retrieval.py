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
from src.modules.retriever.slice_retriever import Image2Image_Slice_Retriever


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

    def process_wsi(self, wsi_dir):
        file_path = os.path.join(self.wsi_file_path, wsi_dir)
        subfile_names = os.listdir(file_path)
        wsi_path = next((os.path.join(file_path, f) for f in subfile_names 
                         if f.lower().endswith(('.svs', '.tiff'))), None)
        if not wsi_path:
            return []
        
        slide = OpenSlide(wsi_path)
        level = slide.level_count
        width, height = slide.level_dimensions[level-1]

        slice_info = {"wsi_path":wsi_path,
                      "position": (width//2, height//2), 
                      "size": (width, height), 
                      "level": level-1, 
                    }

        return [slice_info]

    def generation_query_slice(self):
        if os.path.exists(self.materials_path):
            print("Query region info file exists.")
            return

        query_slice_infos = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.process_wsi, wsi_dir) for wsi_dir in self.wsi_dirs]
            with tqdm(total=len(futures), desc="Processing WSIs") as pbar:
                for future in futures:
                    query_slice_infos.extend(future.result())
                    pbar.update(1)  

        with open(self.materials_path, "w") as file:
            json.dump(query_slice_infos, file, indent=4)
        print("Saved query region infos to", self.materials_path)



class Inter_Retrieval_experiment():
    def __init__(self, wsi_file_path, query_materials_path, encoder):
        self.wsi_file_path = wsi_file_path
        self.encoder = encoder
        self.query_materials_path = query_materials_path
    
    def calculate_ratio(self, target_name, name_list):
        count_target_name = name_list.count(target_name)
        return count_target_name / len(name_list)

    def main(self, args, slice_retriever):
        try:
            with open(self.query_materials_path, "r") as file:
                query_slice_infos = json.load(file)
        except FileNotFoundError:
            print("File does not exist.")

        results = []
        for query_slice_info in tqdm(query_slice_infos):
            wsi_path = query_slice_info["wsi_path"]
            x, y = query_slice_info["position"]
            target_level = query_slice_info["level"]
            slice_size = query_slice_info["size"]

            slide = OpenSlide(wsi_path)
            query_slice = slide.read_region((0, 0), target_level, slice_size) 
            query_slice = query_slice.convert("RGB")

            start = time.time()
            names = slice_retriever.slide_retrieval(query_slice, 
                                                    preprocess=args.preprocess, 
                                                    step=args.step
                                                )
            end = time.time()

            target_name = wsi_path.split("/")[-1]
            ratio_top_1 = self.calculate_ratio(target_name, names[:1])
            ratio_top_3 = self.calculate_ratio(target_name, names[:3])
            ratio_top_5 = self.calculate_ratio(target_name, names[:5])

            print(target_name, names)
            
            result = {
                "right_at_1": ratio_top_1,
                "right_at_3": ratio_top_3,
                "right_at_5": ratio_top_5,
                "time": end - start
            }

            print(result)

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
                            filename="experiment/results/slice_retrieval_results.txt")
        
    def save_results_to_file(self, results, filename="./results/slice_retrieval_results.txt"):
        with open(filename, "a+") as f:
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Experiment Date: {current_date}\n")
            f.write("-" * 50 + "\n")  # 分隔线，增加可读性

            for param_set, metrics in results.items():
                f.write(f"Parameters: {param_set}\n")
                f.write(f"SIM Avg: {metrics['right_at_1']:.4f}\n")
                f.write(f"SIM Avg: {metrics['right_at_3']:.4f}\n")
                f.write(f"SIM Avg: {metrics['right_at_5']:.4f}\n")
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
    
    args.database_path = "data/vector_database_TCGA"   # 正式实验(cancle for test)

    from src.utils.basic.encoder import WSI_Image_UNI_Encoder, WSI_Image_test_Encoder

    encoder = WSI_Image_UNI_Encoder()
    # encoder = WSI_Image_test_Encoder()    # for test

    basic_retriever = Image2Image_Retriever_Qdrant(encoder, args.database_path)
    slice_retriever = Image2Image_Slice_Retriever(basic_retriever, encoder)

    source_materials_path = "experiment/materials/query_source.json"
    region_materials_path = "experiment/materials/query_slice_infos.json"
    # region_materials_path = "experiment/materials/query_region_infos_sample.json"  # for test

    query_source = sample_query_source(args.target_file, source_materials_path, sample_n=50)
    # query_source = [
    #     "b545c7e6-8318-47e1-8ed4-62a5883a1d1c",
    # ]
    data_prepare = DataPreparationProcess(args.wsi_file_path, 
                                          query_source, 
                                          region_materials_path)
    data_prepare.generation_query_slice()

    exp = Inter_Retrieval_experiment(args.wsi_file_path, 
                                     data_prepare.materials_path,
                                     encoder)

    # exp.main(args, region_retriever)

    # ---------------------------- Ablation Experiment ----------------------------

    # for preprocess in ["spectral", "kmeans", "others"]:
    #     for evaluation in ["boc", "traversal"]:
    for preprocess in ["evolution"]:
        for evaluation in ["boc", "traversal"]:
                args.preprocess = preprocess
                args.evaluation = evaluation
                exp.main(args, slice_retriever)