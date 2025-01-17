import os, sys, json, asyncio
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
from torch.utils.data import DataLoader
import multiprocessing as mp

from src.utils.basic.wsi_dataset import WSIImageDataset
from src.utils.basic.encoder import WSI_Image_UNI_Encoder
from src.utils.metadata_wsi.background import load_wsi_thumbnail, get_patch_background_ratio


class Embedding_loader():
    def __init__(self, step=224):
        self.step = step
        self.wsi_patch_encoder = WSI_Image_UNI_Encoder()
        self.cache_path = f"data/metadata_embedding_{self.step}"
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        self.loaded_embeddings = os.listdir(self.cache_path)

    async def load_image_paths(self, folder_path):
        """异步加载图像路径"""
        image_paths = []
        filenames = os.listdir(folder_path)
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(folder_path, filename))
        return image_paths

    async def loading_wsi_image(self, wsi_name):
        """在 CPU 上获取 WSI patch 的 Dataloader。"""
        folder_path = os.path.join(f"data/metadata_patch_{self.step}", wsi_name)
        patch_infos = os.listdir(folder_path)
        image_paths = await self.load_image_paths(folder_path)

        wsi_dataset = WSIImageDataset(image_paths, self.wsi_patch_encoder.transform)
        dataloader = DataLoader(wsi_dataset, batch_size=16, shuffle=False, num_workers=16, pin_memory=True)

        return patch_infos, dataloader

    def loading_worker(self, input_queue, output_queue):
        while True:
            wsi_name = input_queue.get()
            if wsi_name is None:
                break

            if wsi_name in self.loaded_embeddings:
                print(f"WSI {wsi_name} cached.")
                output_queue.put((wsi_name, [], []))
            else:
                patch_infos, dataloader = asyncio.run(self.loading_wsi_image(wsi_name))
                output_queue.put((wsi_name, patch_infos, dataloader))

    def encoding_worker(self, input_queue):
        while True:
            item = input_queue.get()
            if item is None:
                break

            wsi_name, patch_infos, dataloader = item
            patch_embeddings = self.wsi_patch_encoder.encode_wsi_patch(wsi_name, dataloader)

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
    for t in [150, 100, 75, 50]:
        loader = Embedding_loader(t)
        loaded_patches = f"data/metadata_patch_{t}"
        wsi_names_list = [f for f in os.listdir(loaded_patches) if os.path.isdir(os.path.join(loaded_patches, f))]
        loader.main(wsi_names_list)