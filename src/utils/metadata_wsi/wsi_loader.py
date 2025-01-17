import sys, requests, os
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
import numpy as np
from io import BytesIO
from PIL import Image
from multiprocessing import shared_memory
from openslide import OpenSlide





def load_img_url(img_url):
    if "http" in img_url:
        response = requests.get(img_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(img_url).convert("RGB")

    return image


class WSILoader():
    def wsi_loading_patch(self, wsi_path, position=(0,0), level=0, size=(256,256)):
        slide = OpenSlide(wsi_path)

        print(f"文件名: {wsi_path}")
        print(f"图像宽度: {slide.dimensions[0]} 像素")
        print(f"图像高度: {slide.dimensions[1]} 像素")
        print(f"级别数: {slide.level_count}")
        print(f"每级别的尺寸: {slide.level_dimensions}")
        print(f"每级别的降采样因子: {slide.level_downsamples}")

        ture_pos = tuple([int(pos*slide.level_downsamples[level]) for pos in position])
        image = slide.read_region(ture_pos, level, size)
        slide.close()
        image = image.convert('RGB')

        return image

    def wsi_loading_patches(self, wsi_path):
        """Get WSI infos and patches from WSI path"""
        slide = OpenSlide(wsi_path)
        levels, patch_size = slide.level_count, (256, 256)

        loaded_infos, loaded_images = [], []
        for level in range(1, slide.level_count):
            ratio = slide.level_downsamples[level]
            width, height = slide.level_dimensions[level][0], slide.level_dimensions[level][1]
            for w in range(0, width, patch_size[0]):
                for h in range(0, height, patch_size[0]):
                    ture_pos = (int(w * ratio), int(h * ratio))

                    infos = {
                        "wsi_name":wsi_path.split("/")[-1],
                        "position":ture_pos,    # basic on level 0
                        "level":level,
                        "size":patch_size,
                    }

                    image = slide.read_region(ture_pos, level, patch_size)
                    image = image.convert('RGB')
                    loaded_images.append(image)

        return loaded_infos, loaded_images 

    def get_wsi_shared_patches(self, wsi_path):
        """ loading patched image in share memory"""
        loaded_infos, loaded_images = self.wsi_loading_patches(wsi_path)
        img_arrays = np.array([np.array(img, dtype=np.uint8) for img in loaded_images])

        shm = shared_memory.SharedMemory(create=True, size=img_arrays.nbytes)
        shared_array = np.ndarray(img_arrays.shape, dtype=img_arrays.dtype, buffer=shm.buf)
        shared_array[:] = img_arrays[:]     # 将图像数据复制到共享内存

        return loaded_infos, shm, img_arrays.shape, img_arrays.dtype
    

def load_wsi_patch(wsi_url, angle):
    params = {'angle': str(angle)}
    response = requests.get(wsi_url, params=params)

    img = response.content
    img = Image.open(BytesIO(img)).convert("RGB")

    return img

def load_wsi_info(wsi_name):
    url_head = "http://mdi.hkust-gz.edu.cn/wsi/metaservice/api/region/openslide/"
    wsi_url = url_head + wsi_name
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
    
    return wsi_url, num_level, slide_info


def load_wsi_region(wsi_url, x, y, w, h, level, angle):
    """加载 region 的时候用的是中心点和角度值。"""
    x, y, w, h, level, angle = int(x), int(y), int(w), int(h), int(level), int(angle)
    length = max(w, h)
    background_url = f"{wsi_url}/{x-length}/{y-length}/{2*length}/{2*length}/{level}"

    try:
        response = requests.get(background_url)
        background = response.content
        background = Image.open(BytesIO(background)).convert("RGB")
    except:
        return None, None   # 无法正常 load image

    rotated_image = background.rotate(-angle, center=(length, length), expand=False)
    left = length - w / 2
    upper = length - h / 2
    right = length + w / 2
    lower = length + h / 2
    cropped_image = rotated_image.crop((left, upper, right, lower))
    
    return background, cropped_image



if __name__ == "__main__":
    # wsi_path = "data/came/CAMELYON16/images/normal_004.tif"
    # wsi_loader = WSILoader()

    # position, level, size = (0,0), 9, (200,200)
    # wsi_patch_image = wsi_loader.wsi_loading_patch(wsi_path, position, level, size)
    # wsi_patch_image.save("data/cache/"+wsi_path.split("/")[-1].split(".")[0]+".png")

    # position, level, size = (200,200), 8, (200, 200)
    # wsi_patch_image = wsi_loader.wsi_loading_patch(wsi_path, position, level, size)
    # wsi_patch_image.save("data/cache/"+wsi_path.split("/")[-1].split(".")[0]+"1.png")

    wsi_name, x, y, w, h, level, angle = "267136-33.tiff", 20151, 6031, 1014, 1541, 2, 120
    wsi_url = f"http://mdi.hkust-gz.edu.cn/wsi/metaservice/api/region/openslide/{wsi_name}"
    background, cropped_image = load_wsi_region(wsi_url, x, y, w, h, level, angle)
    cropped_image.save("image/region/test_region.png")