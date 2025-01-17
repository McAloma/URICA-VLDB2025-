import os, sys, json, asyncio, aiohttp, requests, logging
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from datetime import datetime

from src.utils.metadata_wsi.background import load_wsi_thumbnail, get_patch_background_ratio



class ImagePatchDownloader:
    def __init__(self, max_concurrent_downloads=100, step=224):
        self.url_head = "http://mdi.hkust-gz.edu.cn/wsi/metaservice/api/region/openslide/"
        self.max_concurrent_downloads = max_concurrent_downloads
        self.step = step
        self.slice_size = (224, 224)
    
        file_path = f"data/metadata_patch_{self.step}/"
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        self.loaded_wsi_name_path = f"data/metadata_patch_{self.step}/loaded_wsis.json"
        self.image_names = self.load_wsi_name(self.loaded_wsi_name_path)
    
    def load_wsi_name(self, json_file_path):
        # 读取已经缓存过的 WSI 的 Names。
        if not os.path.exists(json_file_path):
            with open(json_file_path, "w") as file:
                json.dump([], file)
            return []

        with open(json_file_path, "r") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return []

    def check_image_name(self, wsi_name):
        # 将新加载的 WSI name 加入到 loaded WSI 中。
        json_file_path = f"data/metadata_patch_{self.step}/loaded_wsis.json"
        image_names = self.load_wsi_name(json_file_path)
        if wsi_name not in image_names:
            image_names.append(wsi_name)
            with open(json_file_path, "w") as file:
                json.dump(image_names, file, indent=4)
            return False
        else:
            return True

    def loading_wsi(self, wsi_name):
        # 按照 Name of WSI 来加载和保存 patch 图像。
        if wsi_name in self.image_names:
            print(f"Patch of WSI {wsi_name} in the Cache.")
            return

        wsi_url = self.url_head + wsi_name
        wsi_info_url = wsi_url.replace("region", "sliceInfo")

        try:
            slide_info = eval(requests.get(wsi_info_url).content)
        except:
            print(f"Can not find the information of {wsi_info_url}.")
            return 
        if slide_info == {'error': 'No such file'}:
            print(f"Can not find usrful slide_info :{wsi_info_url}")
            return
        try:
            num_level = int(slide_info["openslide.level-count"])
        except:
            print(f"None useful num_level in wsi {wsi_name}")
            return

        patch_info_list = []
        for level in range(2, num_level):       # start from level 2
            width = int(slide_info[f"openslide.level[{level}].width"])
            height = int(slide_info[f"openslide.level[{level}].height"])

            # NOTE: 这里取得是固定点为的 patch，用于建立 basic 检索数据库
            for y in range(0, height, self.step):
                for x in range(0, width, self.step):
                    if x + self.slice_size[0] > width or y + self.slice_size[1] > height:
                        continue
                    patch_infos = {
                        "x": str(x),
                        "y": str(y),
                        "width": str(self.slice_size[1]),
                        "height": str(self.slice_size[0]),
                        "level": str(level)
                    }
                    patch_info_list.append(patch_infos)
        
        wsi_dir_path = os.path.join(f"data/metadata_patch_{self.step}", wsi_name)
        os.makedirs(wsi_dir_path, exist_ok=True)
        asyncio.run(self.download_images(wsi_name, patch_info_list))

        self.image_names.append(wsi_name)
        with open(self.loaded_wsi_name_path, "w") as file:
            json.dump(self.image_names, file, indent=4)

    async def download_images(self, wsi_name, patch_infos):
        # 按照 list of patch info 请求并行异步发起多张图像的请求。
        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        async with semaphore:   
            async with aiohttp.ClientSession() as session:
                thumbnail, num_level = load_wsi_thumbnail(wsi_name)
                tasks = [self.asy_download_image(session, wsi_name, patch_info, thumbnail, num_level) for patch_info in patch_infos]
                with tqdm(total=len(tasks), ascii=True) as pbar:
                    pbar.set_description(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | WSI: {wsi_name}")    
                    for task in asyncio.as_completed(tasks):
                        await task
                        pbar.update(1)
                        pbar.refresh() 
                cache_patch_num = len(os.listdir(os.path.join(f"data/metadata_patch_{self.step}", wsi_name)))
                print(f"Final cache patch image {cache_patch_num}")

    async def asy_download_image(self, session, wsi_name, patch_info, thumbnail, num_level): 
        # 按照 patch info 进行单张图像的请求并保存。(NOTE：新加 OTSU 前景筛选功能，只请求背景比例低于0.90的图像。)
        try:
            patch_url = os.path.join(self.url_head, wsi_name, ("/").join([patch_info[key] for key in patch_info]))
            async with session.get(patch_url) as response:
                if response.status == 200:
                    img_data = await response.read()
                    try:
                        image = Image.open(BytesIO(img_data)).convert("RGB")
                        _, back_ratio = get_patch_background_ratio(thumbnail, num_level-1, ("_").join([patch_info[key] for key in patch_info])+".png")
                        if back_ratio < 0.90:
                            patch_info = ("_").join([patch_info[key] for key in patch_info])
                            cache_path = os.path.join(f"data/metadata_patch_{self.step}", wsi_name, f"{patch_info}.png")
                            image.save(cache_path)
                    except Exception as e:
                        logging.error(f"Error opening image from {patch_url}: {e}")
                else:
                    logging.error(f"Error downloading image from {patch_url}: HTTP {response.status}")
        except aiohttp.ClientError as e:
            logging.error(f"Network error occurred while downloading image from {patch_url}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error occurred: {e}")


if __name__ == "__main__":
    for t in [150, 100, 75, 50]:
        downloader = ImagePatchDownloader(step=t)
        file_path = "data/wsi_names_50.json"

        with open(file_path, 'r', encoding='utf-8') as f:
            wsi_name_list = json.load(f)
            for wsi_name in wsi_name_list:
                downloader.loading_wsi(wsi_name)