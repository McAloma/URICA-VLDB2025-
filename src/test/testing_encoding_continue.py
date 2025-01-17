import os, sys, timm, json, torch, asyncio, requests, random, aiohttp
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import multiprocessing as mp

from src.utils.wsi_background import load_wsi_thumbnail, get_patch_background_ratio



class WSIUNIEncoder():
    def __init__(self, **kwargs):
        self.embed_model =  timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        local_dir = "checkpoints/vit_large_patch16_224.dinov2.uni_mass100k/"
        self._device = self.infer_torch_device()
        print(self._device)
        self.embed_model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), 
                                         map_location="cpu", 
                                         weights_only=True), strict=True)
        self.embed_model = self.embed_model.to(self._device)
        self.embed_model.eval()

    def infer_torch_device(self):
        """Infer the input to torch.device."""
        try:
            has_cuda = torch.cuda.is_available()
        except NameError:
            import torch  # pants: no-infer-dep
            has_cuda = torch.cuda.is_available()
        if has_cuda:
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def encode_wsi_patch(self, wsi_name, dataloader):
        embeddings = []
        with torch.no_grad():
            for images in tqdm(dataloader, desc=f"WSI name: {wsi_name}", ascii=True):
                images = images.to(self._device)
                embedding = self.embed_model(images)
                embeddings.append(embedding.cpu())

        if embeddings == []:
            return []
        else:
            patch_embeddings = torch.cat(embeddings, dim=0).cpu().tolist()
            return patch_embeddings




class TestingDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        patch1, patch2, patch3, ratio = self.data_list[idx]
        if self.transform:
            patch1 = self.transform(patch1)
            patch2 = self.transform(patch2)
            patch3 = self.transform(patch3)

        ratio = torch.tensor(ratio, dtype=torch.float32)

        return patch1, patch2, patch3, ratio




class Testing_Encoder_Continuity():
    def __init__(self):
        self.wsi_patch_encoder = WSIUNIEncoder()
        self.url_head = "http://mdi.hkust-gz.edu.cn/wsi/metaservice/api/region/openslide/"

    async def async_request_patch(self, session, patch_url, angle=0):
        params = {'angle': str(angle)}
        async with session.get(patch_url, params=params) as response:
            img = Image.open(BytesIO(await response.read())).convert("RGB")
            return img

    async def fetch_all_patches(self, urls):
        async with aiohttp.ClientSession() as session:
            tasks = [self.async_request_patch(session, url) for url in urls]
            return await asyncio.gather(*tasks)

    async def wsi_test_data(self, wsi_name, n=100):
        wsi_url = self.url_head + wsi_name
        wsi_info_url = wsi_url.replace("region", "sliceInfo")

        # 获取切片信息
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(wsi_info_url) as response:
                    slide_info = eval(await response.text())
        except:
            print(f"Cannot find the information of {wsi_info_url}.")
            return
        if slide_info == {'error': 'No such file'}:
            print(f"Cannot find useful slide_info: {wsi_info_url}")
            return
        try:
            num_level = int(slide_info["openslide.level-count"])
        except:
            print(f"No useful num_level in wsi {wsi_name}")
            return

        data_list = []
        for level in range(2, num_level):  # start from level 2
            width = int(slide_info[f"openslide.level[{level}].width"])
            height = int(slide_info[f"openslide.level[{level}].height"])
            target = random.choice(["t", "b", "l", "r"])

            for _ in range(n):
                x0, y0 = random.randint(224, width - 224), random.randint(224, height - 224)

                # 根据目标方向生成坐标和比例
                if target == "l":
                    x1, y1 = x0 - 224, y0
                    x2, y2 = random.randint(x1, x0), y0
                    ratio = (x0 - x2) / 224
                elif target == "r":
                    x1, y1 = x0 + 224, y0
                    x2, y2 = random.randint(x0, x1), y0
                    ratio = (x2 - x0) / 224
                elif target == "t":
                    x1, y1 = x0, y0 - 224
                    x2, y2 = x0, random.randint(y1, y0)
                    ratio = (y0 - y2) / 224
                elif target == "b":
                    x1, y1 = x0, y0 + 224
                    x2, y2 = x0, random.randint(y0, y1)
                    ratio = (y2 - y0) / 224

                patch_url1 = os.path.join(wsi_url, f"{x0}/{y0}/{224}/{224}/{level}")
                patch_url2 = os.path.join(wsi_url, f"{x1}/{y1}/{224}/{224}/{level}")
                patch_url3 = os.path.join(wsi_url, f"{x2}/{y2}/{224}/{224}/{level}")

                # 异步请求获取三个图像
                patches = await self.fetch_all_patches([patch_url1, patch_url2, patch_url3])
                patch1, patch2, patch3 = patches[0], patches[1], patches[2]

                # 将图像和比例存储到数据列表
                data = (patch1, patch2, patch3, ratio)
                data_list.append(data)

        return data_list


    def testing_continue(self, wsi_names, n=1000):
        dises, norm_dises = [], []
        for name in wsi_names:
            data_list = asyncio.run(self.wsi_test_data(name, n=n))
            test_dataset = TestingDataset(data_list, transform=self.wsi_patch_encoder.transform)
            test_loader = DataLoader(test_dataset, batch_size=n, shuffle=False)

            for _, (patch1, patch2, patch3, ratio) in enumerate(test_loader):
                rep1 = self.wsi_patch_encoder(patch1)
                rep2 = self.wsi_patch_encoder(patch2)
                rep3 = self.wsi_patch_encoder(patch3)

                rep_idea = rep1 * (1 - ratio) + rep2 * ratio

                dis = torch.sqrt(torch.sum((rep_idea - rep3) ** 2, dim=1))
                norm_dis = dis / torch.norm(rep_idea, dim=1)
            
                dises.append(dis)
                norm_dises.append(norm_dis)

        return dises, norm_dises

if __name__ == "__main__":
    tester = Testing_Encoder_Continuity()
    wsi_names = ["241183-21.tiff", "258992-29.tiff"]
    dises, norm_dises = tester.testing_continue(wsi_names)

    mean_dis = sum(dises) / len(dises)
    mean_norm_dis = sum(norm_dises) / len(norm_dises)
    print(mean_dis, mean_norm_dis)
    