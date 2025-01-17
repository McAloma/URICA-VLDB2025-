import sys, requests, os, timm, torch, asyncio, json
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from openslide import OpenSlide
from torch.utils.data import DataLoader
import multiprocessing as mp

from src.utils.basic.wsi_dataset import WSIImageDataset
from src.utils.basic.encoder import WSI_Image_UNI_Encoder
from src.utils.open_wsi.backgound import load_wsi_thumbnail, WSI_background_detect, get_patch_background_ratio



class Embedding_loader():
    def __init__(self, step=100):
        self.step = step
        self.wsi_patch_encoder = WSI_Image_UNI_Encoder()
        self.wsi_file_path = "data/Camelyon"

        self.cache_path = f"data/metadata_embedding_Camelyon_{step}"
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        self.loaded_embeddings = os.listdir(self.cache_path)

    async def load_wsi_patches(self, file_name):
        """异步加载 WSI 文件路径"""
        wsi_path = os.path.join(self.wsi_file_path, file_name)
        
        # 准备 slide 和背景缩略图
        slide = OpenSlide(wsi_path)
        _, background, _ = load_wsi_thumbnail(slide)

        levels, patch_size = slide.level_count, (224, 224)
        loaded_infos, loaded_images = [], []
        for level in range(2, levels):     # start from level 2
            ratio = slide.level_downsamples[level]
            width, height = slide.level_dimensions[level][0], slide.level_dimensions[level][1]
            for w in range(0, width, self.step):
                for h in range(0, height, self.step):
                    ture_pos = (int(w * ratio), int(h * ratio))

                    infos = {
                        "wsi_name":wsi_path.split("/")[-1],
                        "position":(w, h),    # basic on current level (左上角)
                        "level":level,
                        "size":patch_size,
                    }
                    
                    _, white_pixel_ratio = get_patch_background_ratio(slide, background, infos)
                    if white_pixel_ratio < 0.95:            # 通过阈值判断是否为背景
                        image = slide.read_region(ture_pos, level, patch_size)
                        image = image.convert('RGB')

                        loaded_infos.append(infos)
                        loaded_images.append(image)

        return loaded_images, loaded_infos
        
    async def loading_wsi_images(self, wsi_name):
        """在 CPU 上异步获取 WSI patch 的 Dataloader。"""
        images, infos = await self.load_wsi_patches(wsi_name)
        wsi_dataset = WSIImageDataset(images, self.wsi_patch_encoder.transform)
        dataloader = DataLoader(wsi_dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True)

        return infos, dataloader

    def loading_worker(self, input_queue, output_queue):
        """加载 WSI 图像的工作进程。"""
        while True:
            wsi_name = input_queue.get()
            if wsi_name is None:
                break

            if wsi_name in self.loaded_embeddings:
                print(f"WSI {wsi_name} cached.")
                output_queue.put((wsi_name, [], []))
            else:
                patch_infos, dataloader = asyncio.run(self.loading_wsi_images(wsi_name))
                output_queue.put((wsi_name, patch_infos, dataloader))

    def encoding_worker(self, input_queue):
        """编码 WSI 图像的工作进程。"""
        while True:
            item = input_queue.get()
            if item is None:
                break

            wsi_name, patch_infos, dataloader = item
            if patch_infos != [] and dataloader != []:
                patch_embeddings = self.wsi_patch_encoder.encode_wsi_patch(wsi_name, dataloader, show=True)       # 这里出来的是一个 tensor 的 list
                patch_embeddings = torch.concat(patch_embeddings, dim=0).tolist()

                dir_path = os.path.join(self.cache_path, wsi_name)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

                info_path = os.path.join(self.cache_path, wsi_name, "patch_info.json")
                with open(info_path, 'w') as file:
                    json.dump(patch_infos, file)

                embedding_path = os.path.join(self.cache_path, wsi_name, "embeddings.json")
                with open(embedding_path, 'w') as file:
                    json.dump(patch_embeddings, file)

    def main(self, wsi_names_list):
        load_workers = 2
        load_queue = mp.Queue(maxsize=8)
        encode_queue = mp.Queue(maxsize=8)

        loading_processes = [mp.Process(target=self.loading_worker, args=(load_queue, encode_queue)) for _ in range(load_workers)]
        encoding_process = mp.Process(target=self.encoding_worker, args=(encode_queue,))

        for p in loading_processes:
            p.start()
        encoding_process.start()

        for wsi_name in wsi_names_list:
            load_queue.put(wsi_name)

        for _ in range(load_workers):
            load_queue.put(None)
        for p in loading_processes:
            p.join()

        encode_queue.put(None)
        encoding_process.join()


if __name__ == "__main__":
    mp.set_start_method('spawn')

    wsis_list = [                   
        "tumor_001.tif",  
        "tumor_002.tif",  
        "tumor_003.tif",  
        "tumor_004.tif",  
        "tumor_005.tif",
        "tumor_006.tif",  
        "tumor_007.tif",  
        "tumor_008.tif",  
        "tumor_009.tif",  
        "tumor_010.tif",
        "tumor_011.tif",  
        "tumor_012.tif",  
        "tumor_013.tif",  
        "tumor_014.tif",  
        "tumor_015.tif",

    ]   # 16 - 20 for test

    loader = Embedding_loader(step=100)  
    loader.main(wsis_list)


