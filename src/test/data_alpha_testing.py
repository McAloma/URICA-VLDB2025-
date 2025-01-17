import os, sys, timm, torch, asyncio, random, aiohttp, math
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
from io import BytesIO
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from openslide import OpenSlide

from src.utils.basic.encoder import WSI_Image_UNI_Encoder
from src.utils.basic.wsi_dataset import WSIImageDataset
from src.utils.open_wsi.backgound import load_wsi_thumbnail, WSI_background_detect, get_patch_background_ratio




class Testing_Encoder_Continuity():
    def __init__(self):
        self.wsi_patch_encoder = WSI_Image_UNI_Encoder()
        self.wsi_file_path = "data/TCGA_BRCA"

    async def wsi_test_data(self, file_name, n=50):
        file_path = os.path.join(self.wsi_file_path, file_name)
        subfile_names = os.listdir(file_path)
        wsi_path = ""
        for filename in subfile_names:
            if filename.lower().endswith(('.svs', '.tiff')):
                wsi_path = os.path.join(file_path, filename)
        
        if not wsi_path:
            print(f"Cannot find the WSI file in {file_path}.")
            return None
        
        # 准备 slide 和背景缩略图
        slide = OpenSlide(wsi_path)
        num_level = slide.level_count

        data_list, delta_list = [], []
        for level in range(num_level):
            ratio = slide.level_downsamples[level]
            width, height = slide.level_dimensions[level][0], slide.level_dimensions[level][1]
                    
            for _ in range(n):
                if width < 896 or height < 896:
                    continue

                x0, y0 = random.randint(448, width - 448), random.randint(488, height - 448)
                x1, y1 = random.randint(448, width - 448), random.randint(448, height - 448)
                xs, ys = random.randint(-224, 224), random.randint(-224, 224)   # 取样范围是 [-2s,2s]

                ture_x0, ture_y0 = int(x0 * ratio), int(y0 * ratio)
                ture_x1, ture_y1 = int(x1 * ratio), int(y1 * ratio)
                ture_xs, ture_ys = int(xs * ratio), int(ys * ratio)

                patch1 = slide.read_region((ture_x0, ture_y0), level, (224, 224))
                patch1 = patch1.convert('RGB')
                patch2 = slide.read_region((ture_x1, ture_y1), level, (224, 224))
                patch2 = patch2.convert('RGB')
                patch3 = slide.read_region((ture_x0+ture_xs, ture_y0+ture_ys), level, (224, 224))
                patch3 = patch3.convert('RGB')
                patch4 = slide.read_region((ture_x1+ture_xs, ture_y1+ture_ys), level, (224, 224))
                patch4 = patch4.convert('RGB')

                if patch1 and patch2 and patch3 and patch4:
                    data = [patch1, patch2, patch3, patch4]
                    data_list.append(data)

                    delta = (1-abs(xs)/224) * (1-abs(ys)/224)
                    delta_list.append(delta)

        return data_list, delta_list


    def testing_alpha_param(self, wsi_names, n=50):
        alpha_results = []
        for name in tqdm(wsi_names):
            try:
                data_list, delta_list = asyncio.run(self.wsi_test_data(name, n=n))
            except:
                continue

            dateset1 = WSIImageDataset([item[0] for item in data_list], transform=self.wsi_patch_encoder.transform)
            dateset2 = WSIImageDataset([item[1] for item in data_list], transform=self.wsi_patch_encoder.transform)
            dateset3 = WSIImageDataset([item[2] for item in data_list], transform=self.wsi_patch_encoder.transform)
            dateset4 = WSIImageDataset([item[3] for item in data_list], transform=self.wsi_patch_encoder.transform)

            loader1 = DataLoader(dateset1, batch_size=n, shuffle=False)
            loader2 = DataLoader(dateset2, batch_size=n, shuffle=False)
            loader3 = DataLoader(dateset3, batch_size=n, shuffle=False)
            loader4 = DataLoader(dateset4, batch_size=n, shuffle=False)

            rep1 = self.wsi_patch_encoder.encode_wsi_patch(name, loader1, show=True)
            rep1 = torch.concat(rep1, dim=0)
            rep2 = self.wsi_patch_encoder.encode_wsi_patch(name, loader2, show=True)
            rep2 = torch.concat(rep2, dim=0)
            rep3 = self.wsi_patch_encoder.encode_wsi_patch(name, loader3, show=True)
            rep3 = torch.concat(rep3, dim=0)
            rep4 = self.wsi_patch_encoder.encode_wsi_patch(name, loader4, show=True)
            rep4 = torch.concat(rep4, dim=0)

            orgin_sim = F.cosine_similarity(rep1, rep2, dim=1)
            shift_sim = F.cosine_similarity(rep3, rep4, dim=1)

            alpha_list = []
            for a, b, c in zip(orgin_sim, shift_sim, delta_list):
                a, b = torch.abs(a), torch.abs(b)
                if b > a:
                    alpha = (math.log(1-b+1e-6) - math.log(1-a+1e-6)) / math.log(c+1e-6)      # In Upper Bound: alpha = (ln(1-b)-ln(1-a))/lnc (b>a)
                elif b < a:
                    alpha = (math.log(b+1e-6) - math.log(a+1e-6)) / math.log(c+1e-6)              # In Lower Bound: alpha = (lnb-lna)/lnc (b<a)
                else:
                    alpha = 0
                alpha_list.append(alpha)

            alpha_result = np.percentile(alpha_list, 99)
            alpha_results.append(alpha_result)

        return alpha_results


if __name__ == "__main__":
    tester = Testing_Encoder_Continuity()
    wsi_names = [
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
    alpha_results = tester.testing_alpha_param(wsi_names, n=100)
        
    file_path = "src/test/results/alpha_results.txt"
    with open(file_path, "w", encoding="utf-8") as file:
        for item in alpha_results:
            file.write(f"{item}\n")  # 每个列表项写入一行

    print(f"列表内容已保存到 {file_path}")
    