import os, torch, asyncio, json
from torch.utils.data import DataLoader
from openslide import OpenSlide
from torch.utils.data import DataLoader
import multiprocessing as mp

from src.utils.basic.wsi_dataset import WSIImageDataset
from src.utils.basic.encoder import WSI_Image_UNI_Encoder
from src.utils.open_wsi.backgound import load_wsi_thumbnail, get_patch_background_ratio



class Embedding_loader():
    def __init__(self, step=100):
        self.step = step
        self.wsi_patch_encoder = WSI_Image_UNI_Encoder()
        self.wsi_file_path = "data/TCGA_BRCA"

        self.cache_path = f"data/metadata_embedding_TCGA_{step}"
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        self.loaded_embeddings = os.listdir(self.cache_path)

    async def load_wsi_patches(self, file_name):
        file_path = os.path.join(self.wsi_file_path, file_name)
        subfile_names = os.listdir(file_path)
        wsi_path = ""
        for filename in subfile_names:
            if filename.lower().endswith(('.svs', '.tiff')):
                wsi_path = os.path.join(file_path, filename)
        
        if wsi_path == "":
            print(f"Can not find the WSI file in {file_path}.")
            return None
        
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
                        "position":(w, h),
                        "level":level,
                        "size":patch_size,
                    }
                    
                    _, white_pixel_ratio = get_patch_background_ratio(slide, background, infos)
                    if white_pixel_ratio < 0.95:       
                        image = slide.read_region(ture_pos, level, patch_size)
                        image = image.convert('RGB')

                        loaded_infos.append(infos)
                        loaded_images.append(image)

        return loaded_images, loaded_infos
        
    async def loading_wsi_images(self, wsi_name):
        images, infos = await self.load_wsi_patches(wsi_name)
        wsi_dataset = WSIImageDataset(images, self.wsi_patch_encoder.transform)
        dataloader = DataLoader(wsi_dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True)

        return infos, dataloader

    def loading_worker(self, input_queue, output_queue):
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
        while True:
            item = input_queue.get()
            if item is None:
                break

            wsi_name, patch_infos, dataloader = item
            if patch_infos != [] and dataloader != []:
                patch_embeddings = self.wsi_patch_encoder.encode_wsi_patch(wsi_name, dataloader, show=True)     
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
    
    # import random
    # wsi_names_list = [f for f in os.listdir(wsi_file_path) if os.path.isdir(os.path.join(wsi_file_path, f))]
    # selected_wsi = random.sample(wsi_names_list, 20)
    # print(selected_wsi)

    wsi_names_list = [
        'e986e7c8-32b0-4f46-8721-33172a9147d8', 
        '4ee14819-c89c-48e8-b4ae-57a98757e436', 
        '41f5232a-95f7-409b-a37e-071219c9725c', 
        '489fe1a1-3b63-417f-883f-a8db190c7c78', 
        '22fddedb-f726-4c11-9ce7-080be5658746', 
        '179185d2-5279-4e19-b9da-1f2296973156', 
        '71bc1dee-32ed-40c9-b177-d96fad4eef23', 
        '8d780eb5-29c7-419a-a7e8-70f5fee50b31', 
        'c5fae5f6-db9a-4b48-87b9-e57190c8be3e', 
        'ac76ef0e-eab9-4568-ad64-dd9e17a2103a', 
        '4a843a81-6c41-4c5f-a6de-aff7b600e2dc', 
        'c3200120-e484-45cd-97f4-30d0635c7938', 
        '745c5fd8-def8-46f7-84bd-ea6c573156fd', 
        '915ff10d-452f-46be-9ff1-a0ae127e9dd4', 
        'c0771f84-757a-417d-8841-73735b256322', 
        '7709738b-ba8e-4b20-9175-5db9b5c28556', 
        '7b7166ab-36b5-4f1b-bd82-8b45d8b7ecbd', 
        'bb7d6bc1-4d6a-4ecc-a8ad-ecc31ff4f471', 
        'f4b8f536-2e19-428b-a9f5-2279aab6ba67', 
        'daedfd81-4a6f-4f99-a505-c9363c961f4e'
    ]

    for step in [224, 200, 150, 100, 80, 60, 50]:
        loader = Embedding_loader(step=step)  
        loader.main(wsi_names_list)


